#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

static const char *device_type_name(VkPhysicalDeviceType type) {
    switch (type) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU: return "cpu";
        default: return "other";
    }
}

static void print_json_string(const char *value) {
    putchar('"');
    for (const unsigned char *p = (const unsigned char *) value; *p; ++p) {
        switch (*p) {
            case '"': fputs("\\\"", stdout); break;
            case '\\': fputs("\\\\", stdout); break;
            case '\b': fputs("\\b", stdout); break;
            case '\f': fputs("\\f", stdout); break;
            case '\n': fputs("\\n", stdout); break;
            case '\r': fputs("\\r", stdout); break;
            case '\t': fputs("\\t", stdout); break;
            default:
                if (*p < 0x20) {
                    printf("\\u%04x", *p);
                } else {
                    putchar(*p);
                }
        }
    }
    putchar('"');
}

static int supports_memory_budget(VkPhysicalDevice device) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, NULL) != VK_SUCCESS) {
        return 0;
    }
    VkExtensionProperties *extensions = calloc(count, sizeof(*extensions));
    if (extensions == NULL) {
        return 0;
    }
    int supported = 0;
    if (vkEnumerateDeviceExtensionProperties(device, NULL, &count, extensions) == VK_SUCCESS) {
        for (uint32_t i = 0; i < count; ++i) {
            if (strcmp(extensions[i].extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0) {
                supported = 1;
                break;
            }
        }
    }
    free(extensions);
    return supported;
}

int main(void) {
    VkApplicationInfo app = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "transcriber-vulkan-probe",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "none",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_1,
    };
    VkInstanceCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app,
    };
    VkInstance instance = VK_NULL_HANDLE;
    VkResult result = vkCreateInstance(&create_info, NULL, &instance);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkCreateInstance failed: %d\n", result);
        return 1;
    }

    uint32_t count = 0;
    result = vkEnumeratePhysicalDevices(instance, &count, NULL);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "vkEnumeratePhysicalDevices failed: %d\n", result);
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    VkPhysicalDevice *devices = calloc(count, sizeof(*devices));
    if (count > 0 && devices == NULL) {
        fprintf(stderr, "out of memory\n");
        vkDestroyInstance(instance, NULL);
        return 1;
    }
    if (count > 0 && vkEnumeratePhysicalDevices(instance, &count, devices) != VK_SUCCESS) {
        fprintf(stderr, "could not enumerate Vulkan devices\n");
        free(devices);
        vkDestroyInstance(instance, NULL);
        return 1;
    }

    int hardware_devices = 0;
    fputs("{\"schema_version\":1,\"devices\":[", stdout);
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(devices[i], &properties);

        VkPhysicalDeviceMemoryProperties2 memory = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
        };
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT,
        };
        const int has_budget = supports_memory_budget(devices[i]);
        if (has_budget) {
            memory.pNext = &budget;
        }
        vkGetPhysicalDeviceMemoryProperties2(devices[i], &memory);

        uint64_t heap_size = 0;
        uint64_t heap_budget = 0;
        for (uint32_t heap = 0; heap < memory.memoryProperties.memoryHeapCount; ++heap) {
            if ((memory.memoryProperties.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
                continue;
            }
            if (memory.memoryProperties.memoryHeaps[heap].size > heap_size) {
                heap_size = memory.memoryProperties.memoryHeaps[heap].size;
                heap_budget = has_budget ? budget.heapBudget[heap] : 0;
            }
        }

        if (i > 0) putchar(',');
        fputs("{\"index\":", stdout);
        printf("%u,\"name\":", i);
        print_json_string(properties.deviceName);
        fputs(",\"type\":", stdout);
        print_json_string(device_type_name(properties.deviceType));
        printf(",\"vendor_id\":%u,\"heap_size_bytes\":%llu,\"heap_budget_bytes\":",
               properties.vendorID, (unsigned long long) heap_size);
        if (has_budget) {
            printf("%llu", (unsigned long long) heap_budget);
        } else {
            fputs("null", stdout);
        }
        putchar('}');

        if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU) {
            hardware_devices++;
        }
    }
    fputs("]}\n", stdout);

    free(devices);
    vkDestroyInstance(instance, NULL);
    return hardware_devices > 0 ? 0 : 2;
}
