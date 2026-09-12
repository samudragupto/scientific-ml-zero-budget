# Security Policy

## Scope & Educational Nature

**Scientific ML on Zero Budget** is an open-source educational and reproducible research demonstration project. It is designed to illustrate memory efficiency, mixed precision, and resilient training patterns on free-tier cloud GPUs (Google Colab, Kaggle).

- **No Network Ingress/Egress Required:** All benchmark datasets in this repository are generated offline deterministically using physical synthetic dynamical equations.
- **No Credentials/Secrets Stored:** The repository contains no API keys, private tokens, or user data.

## Supported Versions

| Version | Supported |
| :--- | :--- |
| `main` (latest release) | :white_check_mark: |
| `< 0.1.0` | :x: |

## Reporting a Vulnerability

If you discover a potential security vulnerability (such as an unsafe serialization flaw, command injection, or denial-of-service condition), please report it responsibly:

1. **GitHub Security Advisory (Preferred):**
   Navigate to the repository's **Security** tab and click **"Report a vulnerability"** to submit a private draft advisory.
2. **Alternative Disclosure:**
   Open a confidential issue or contact the project maintainers directly through GitHub.

### What to Include
- A clear description of the vulnerability.
- Minimal, reproducible steps or proof of concept.
- Affected files, functions, or execution environments.

### Response Timeline
- **Acknowledgement:** Within 48 hours.
- **Assessment & Triage:** Within 5 business days.
- **Remediation & Patch:** A release or mitigation will be published promptly upon verification.
