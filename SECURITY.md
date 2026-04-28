# Security Policy

## Supported Versions

Security fixes target the latest `main` branch unless a release branch is created later.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to the repository owner. Include:

- the affected file, dependency, or workflow;
- steps to reproduce;
- impact and likelihood;
- suggested mitigation if known.

## Model Safety Notes

This model estimates housing prices from a small structured dataset. It should not be used as the sole basis for lending, insurance, taxation, or legal decisions. Production use should include monitoring, human review, privacy checks, and fairness analysis.

## Security Checks

Security checks are documented in [docs/SECURITY_TESTING.md](docs/SECURITY_TESTING.md). The GitHub workflows include Bandit, pip-audit, CodeQL, and Dependabot configuration.
