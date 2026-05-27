# Meeting Report

**Source file:** `meeting_20260527_124817.wav`  
**Date:** May 27, 2026  
**Duration:** 31 minutes  

---

## Executive Summary

The meeting discusses the benefits of running Microsoft Defender in Passive Mode, particularly for organizations using third-party endpoint protection tools like CrowdStrike. The participants highlight the integration of Defender's features with existing tools, such as Threatened Bonability Management and Smart Screen Detection, and the potential for enhanced incident response capabilities. They also touch on the challenges of deploying Defender on Windows Servers and the importance of proper configuration to avoid conflicts with other antivirus software.

## Detailed Summary

Nick Ross, CEO of Clock Appsal, introduces the topic of running Microsoft Defender in Passive Mode, explaining that it allows organizations to coexist with third-party endpoint protection tools while leveraging Defender’s core features. He mentions that Passive Mode enables the analysis of vulnerabilities and telemetry from the OS layer, including CBEs and public exploits. Nathan Taylor adds that many of the company’s customers are entitled to Defender but choose to use other tools due to scalability issues with Defender and the desire for multiple layers of protection. 

The discussion then shifts to the concept of Defender in Block Mode, where Defender can override the decisions of other EDR tools. This is seen as a fail-safe mechanism, allowing for additional detection and remediation steps. However, the participants note that this can lead to conflicts between Defender and other EDR tools, requiring careful configuration.

Key points covered include:
- The benefits of passive mode, such as EDR telemetry data and the ability to enforce policies like Endpoint DLP.
- The exclusion of certain features like ASR rules when using Defender in passive mode.
- The importance of proper configuration, especially on Windows Servers, to avoid conflicts with other antivirus software.
- The visibility and data collection capabilities of the Defender XDR portal, which can aid in incident response.

The conversation also touches on the deployment of Defender on Windows Servers, noting that it is less common in small businesses but is beneficial for larger enterprises. Participants discuss the licensing complexities and the potential for increased visibility and data collection through Defender in passive mode.

## Action Items

- None explicitly stated.

## Key Decisions

- Defender in Passive Mode is recommended for organizations using third-party endpoint protection tools to maximize the benefits of both Defender and the third-party tool.
- Defender in Block Mode provides a fail-safe mechanism but requires careful configuration to avoid conflicts with other EDR tools.

## Topics Discussed

- Running Microsoft Defender in Passive Mode.
- Benefits and limitations of Defender in Passive Mode.
- Defender in Block Mode.
- Configuration considerations for Defender in passive mode.
- Deployment of Defender on Windows Servers.
- Incident response capabilities of Defender in passive mode.
- Licensing and management challenges associated with Defender.
```
