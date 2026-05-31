# Meeting Report

**Source file:** `example_recording.wav`  
**Date:** Unknown  
**Duration:** 31 minutes  
**Requested language:** English (en)  
**Detected language:** English (en) (100% confidence)  

---

## Executive Summary

The discussion centered on the value and implementation of running Microsoft Defender in passive mode. Passive mode allows organizations to coexist with existing third-party endpoint protection tools (like CrowdStrike or SentinelOne) while still leveraging core, valuable Defender functionalities. Key benefits include maximizing existing Microsoft licensing, gaining additional telemetry, and utilizing features like Threat Vulnerability Management and Endpoint DLP. While the process is generally safe, users must be cautious about potential conflicts, especially when considering "block mode."

## Detailed Summary

The podcast featured host Nathan Taylor and guest Nick Ross, CEO of ClockApsil, discussing Microsoft Defender in passive mode.

**What is Passive Mode?**
Passive mode is a mechanism that allows multiple security solutions to coexist. It enables users to utilize the core features of Defender for Business or Defender for Endpoint alongside their primary, existing third-party endpoint protection tool.

**Key Benefits and Features:**
*   **Coexistence:** It allows for the simultaneous use of multiple security layers, addressing the desire to not rely solely on Microsoft's heuristics.
*   **Threat Vulnerability Management:** This feature allows analysis of vulnerabilities on the OS and software layers, providing telemetry on Common Vulnerabilities and Exposures (CVEs) and advising on immediate patching or removal.
*   **Phishing Detection:** Defender can provide additional telemetry on man-in-the-middle attacks and phishing sites, correlating data between the email and the endpoint.
*   **Data Correlation (Attack Graph):** The E5 security bundle pitches an "attack graph" that links evidence from email, endpoint, and user identity to determine high correlation for potential breaches.
*   **Endpoint DLP:** This feature, available in passive mode, allows for enforcing policies like preventing copy/paste of corporate data into uncompliant locations (e.g., personal cloud storage) across browsers.
*   **Telemetry for AI:** Running in passive mode ensures that the Defender XDR portal collects valuable telemetry data, which is crucial for AI-driven incident response and Security Copilot, regardless of whether the organization is on E5 or Business Premium.

**Operational Considerations:**
*   **Management Challenges:** A recurring theme was the difficulty of managing Defender at scale for Managed Service Providers (MSPs), often favoring traditional EDR providers for their ease of use in multi-tenant environments.
*   **Conflict Risk:** Participants advised caution regarding "block mode," as it can create conflicts with existing EDRs.
*   **Deployment:** Ideally, passive mode should activate automatically when another EDR is detected. For Windows Servers, explicit registry configuration is required. A pilot deployment is strongly recommended.
*   **Advanced Features:** The discussion covered leveraging features like Attack Surface Reduction (ASR) rules and Endpoint DLP for data governance at scale.

## Action Items

*   **Pilot Deployment:** It is recommended to conduct a pilot test of passive mode on a small group of endpoints before rolling it out across the entire fleet.
*   **Review Licensing:** Users should review their existing licensing to understand which Defender capabilities are available in passive mode.
*   **Configuration:** For Windows Servers, explicit registry keys must be set to enable passive mode.
*   **Exclusions Management:** Best practices dictate that exclusions must be managed on both the primary EDR and Defender to prevent conflicts.

## Key Decisions

None was explicitly stated.

## Topics Discussed

*   Microsoft Defender Passive Mode functionality.
*   Coexistence of multiple Endpoint Detection and Response (EDR) tools (e.g., Defender with CrowdStrike or SentinelOne).
*   Security features available in passive mode (e.g., Threat Vulnerability Management, Endpoint DLP, Phishing Detection).
*   Operational challenges for MSPs managing Defender at scale.
*   Deployment best practices, including pilot testing and registry configuration for servers.
*   The role of data telemetry in modern AI-driven security and Zero Trust architectures.
