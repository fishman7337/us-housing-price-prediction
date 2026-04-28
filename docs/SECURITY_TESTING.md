# Security Testing

## Purpose

This project is not a deployed production service, but it still includes security checks so that dependency risks and unsafe coding patterns are caught early.

## Local Security Commands

```bash
python -m bandit -r src -c pyproject.toml
python -m pip_audit -r requirements.txt
```

## GitHub Security Automation

The repository includes:

- Bandit static security scanning in CI.
- pip-audit dependency vulnerability scanning in CI.
- CodeQL analysis through `.github/workflows/codeql.yml`.
- Dependabot updates for Python dependencies and GitHub Actions.

## Scope

Security checks focus on the reusable Python package under `src/` and declared runtime dependencies. Notebooks are excluded from Bandit because they preserve academic narrative and exploratory work.

## Manual Review Notes

Before real deployment, review:

- data provenance and consent;
- secrets management;
- API authentication and rate limiting if a service is added;
- model artifact storage and access controls;
- privacy and fairness risks;
- dependency licenses and vulnerability reports.
