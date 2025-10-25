# GitHub App Authentication Example

This document provides examples of how to use GitHub App installation authentication with the sync service.

We recommend App authentication in production. You can use personal access tokens for testing or small-scale use.

## Prerequisites

1. A GitHub App with the following permissions:
   - Repository permissions:
     - Contents: Read and write
     - Pull requests: Read and write

2. The app installed to your organization/repository

3. The following information:
   - App ID (from GitHub App settings)
   - Installation ID (from the installation URL)
   - Private key (downloaded `.pem` file)

## Environment Variables

### Using Private Key File

```bash
export HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api"
export GITHUB_APP_ID="123456"
export GITHUB_APP_INSTALLATION_ID="12345678"
export GITHUB_APP_PRIVATE_KEY_PATH="/path/to/private-key.pem"
```

### Using Private Key Content

If you store the private key in a secret manager or environment variable:

```bash
export HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api"
export GITHUB_APP_ID="123456"
export GITHUB_APP_INSTALLATION_ID="12345678"
export GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA...
...your private key content...
-----END RSA PRIVATE KEY-----"
```

**Note:** When setting multi-line environment variables, you may need to use literal `\n`:

```bash
export GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
```

The sync service will automatically convert `\n` to actual newlines.

## Programmatic Usage

```python
from horde_model_reference.sync import GitHubSyncClient
from horde_model_reference.sync.config import github_app_settings, github_sync_settings

# Check if GitHub App authentication is configured
if github_app_settings.is_configured():
    print("Using GitHub App authentication")
    print(f"App ID: {github_app_settings.github_app_id}")
    print(f"Installation ID: {github_app_settings.github_installation_id}")
else:
    print("GitHub App authentication not configured")
    if github_sync_settings.github_token:
        print("Using token authentication")
    else:
        print("No authentication configured")

# Create client (will automatically use GitHub App if configured)
with GitHubSyncClient() as client:
    # The client will use GitHub App authentication if available,
    # falling back to token authentication if not
    # Your sync operations here...
    pass
```

## Testing Authentication

You can test that GitHub App authentication is working by running a simple script:

```python
from github import Auth, Github
from horde_model_reference.sync.config import github_app_settings

if not github_app_settings.is_configured():
    print("GitHub App not configured")
    exit(1)

try:
    private_key = github_app_settings.get_private_key_content()

    # Create App authentication
    app_auth = Auth.AppAuth(
        app_id=github_app_settings.github_app_id,
        private_key=private_key,
    )

    # Get installation authentication
    auth = app_auth.get_installation_auth(
        installation_id=github_app_settings.github_installation_id,
    )

    # Create GitHub client
    g = Github(auth=auth)

    # Test by getting authenticated user/app info
    user = g.get_user()
    print(f"Authenticated as: {user.login}")
    print("GitHub App authentication is working!")

except Exception as e:
    print(f"GitHub App authentication failed: {e}")
    exit(1)
```

## Docker/Kubernetes Deployment

### Docker Compose

```yaml
version: '3.8'

services:
  horde-github-sync:
    image: horde-model-reference:latest
    environment:
      - HORDE_GITHUB_SYNC_PRIMARY_API_URL=https://stablehorde.net/api
      - GITHUB_APP_ID=123456
      - GITHUB_APP_INSTALLATION_ID=12345678
      - GITHUB_APP_PRIVATE_KEY_PATH=/secrets/github-app-key.pem
    volumes:
      - ./secrets/github-app-key.pem:/secrets/github-app-key.pem:ro
    command: python scripts/sync/sync_github_references.py
```

### Kubernetes Secret

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: github-app-credentials
type: Opaque
stringData:
  app-id: "123456"
  installation-id: "12345678"
  private-key: |
    -----BEGIN RSA PRIVATE KEY-----
    MIIEpAIBAAKCAQEA...
    ...your private key content...
    -----END RSA PRIVATE KEY-----
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: horde-github-sync
spec:
  replicas: 1
  selector:
    matchLabels:
      app: horde-github-sync
  template:
    metadata:
      labels:
        app: horde-github-sync
    spec:
      containers:
      - name: sync
        image: horde-model-reference:latest
        command: ["python", "scripts/sync/sync_github_references.py"]
        env:
        - name: HORDE_GITHUB_SYNC_PRIMARY_API_URL
          value: "https://stablehorde.net/api"
        - name: GITHUB_APP_ID
          valueFrom:
            secretKeyRef:
              name: github-app-credentials
              key: app-id
        - name: GITHUB_APP_INSTALLATION_ID
          valueFrom:
            secretKeyRef:
              name: github-app-credentials
              key: installation-id
        - name: GITHUB_APP_PRIVATE_KEY
          valueFrom:
            secretKeyRef:
              name: github-app-credentials
              key: private-key
```

## GitHub Actions

For GitHub Actions, you can use the built-in `GITHUB_TOKEN` or create a GitHub App:

```yaml
name: Sync from PRIMARY
on:
  schedule:
    - cron: '0 */6 * * *'
  workflow_dispatch:

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install -e ".[sync]"

      # Option 1: Use GitHub App (recommended)
      - name: Run sync with GitHub App
        env:
          HORDE_GITHUB_SYNC_PRIMARY_API_URL: ${{ secrets.PRIMARY_API_URL }}
          GITHUB_APP_ID: ${{ secrets.GITHUB_APP_ID }}
          GITHUB_APP_INSTALLATION_ID: ${{ secrets.GITHUB_APP_INSTALLATION_ID }}
          GITHUB_APP_PRIVATE_KEY: ${{ secrets.GITHUB_APP_PRIVATE_KEY }}
        run: python scripts/sync/sync_github_references.py

      # Option 2: Use personal access token
      # - name: Run sync with token
      #   env:
      #     HORDE_GITHUB_SYNC_PRIMARY_API_URL: ${{ secrets.PRIMARY_API_URL }}
      #     GITHUB_TOKEN: ${{ secrets.GH_PAT }}
      #   run: python scripts/sync/sync_github_references.py
```

## Troubleshooting

### "GitHub App settings are not fully configured"

Ensure all three environment variables are set:
- `GITHUB_APP_ID`
- `GITHUB_APP_INSTALLATION_ID`
- `GITHUB_APP_PRIVATE_KEY` or `GITHUB_APP_PRIVATE_KEY_PATH`

### "Failed to read private key from file"

- Check that the file path is correct
- Ensure the file exists and is readable
- Verify the file contains a valid PEM-formatted private key

### "GitHub App authentication failed"

- Verify the App ID is correct
- Verify the Installation ID is correct (check the installation URL)
- Ensure the private key matches the GitHub App
- Check that the app is installed to the correct organization/repository
- Verify the app has the required permissions (Contents: Read & Write, Pull requests: Read & Write)

### Getting Installation ID

The Installation ID can be found in the URL when viewing the app installation:

```
https://github.com/organizations/YOUR_ORG/settings/installations/12345678
                                                                   ^^^^^^^^
                                                              Installation ID
```

Or via the GitHub API:

```bash
# Using gh CLI
gh api /app/installations --jq '.[] | {id: .id, account: .account.login}'

# Using curl (requires JWT token)
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     https://api.github.com/app/installations
```
