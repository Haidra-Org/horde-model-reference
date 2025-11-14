# GitHub Model Reference Sync Service

Automated tool for keeping GitHub legacy model reference repositories in sync with the PRIMARY instance.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Operating Modes](#operating-modes)
  - [One-Shot Mode](#one-shot-mode)
  - [Watch Mode](#watch-mode)
- [Configuration](#configuration)
  - [Docker Configuration](#docker-configuration-recommended)
  - [Non-Docker Configuration](#non-docker-configuration)
  - [GitHub Authentication](#github-authentication)
  - [Optional Settings](#optional-settings)
- [Usage](#usage)
- [Deployment](#deployment)
- [Components](#components)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Security Considerations](#security-considerations)
- [Contributing](#contributing)

## Overview

The GitHub sync service compares model references from a PRIMARY instance (v1 API) with GitHub legacy repositories and automatically creates pull requests when drift is detected. This ensures GitHub repos stay up-to-date with the canonical PRIMARY source.

The service can operate in two modes:

1. **One-shot mode** (default): Performs a single sync operation and exits - ideal for scheduled cron jobs or CI/CD pipelines
2. **Watch mode**: Continuously monitors the PRIMARY server for changes and automatically syncs when updates are detected - ideal for development or long-running deployments

## Architecture

```text
PRIMARY (v1 API) ────► Comparator ◄──── GitHub (legacy repos)
                           │
                           ▼
                     Detect Drift
                           │
                           ▼
                    GitHubSyncClient
                           │
                           ├─► Clone repo
                           ├─► Create branch
                           ├─► Update files
                           ├─► Commit changes
                           ├─► Push branch
                           └─► Create PR
```

**Components:**

- **GithubSynchronizer**: Fetches data from PRIMARY and GitHub
- **ModelReferenceComparator**: Compares and detects differences
- **GitHubSyncClient**: Creates PRs with detected changes
- **WatchModeManager**: Monitors metadata and triggers syncs (watch mode only)

## Important Notes

- Ensure your environment variables are set correctly before running the sync service.
- **Git Identity Configuration**: The user running the sync service must have git identity configured for commits to work:

  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "your.email@example.com"
  ```

- **If running the sync service on the same host as the PRIMARY instance**
    - You must set a different `AIWORKER_CACHE_HOME` for each or it will never detect changes. You should always set this variable when using the sync service.

## Installation

Install with sync dependencies:

```bash
# Using pip
pip install -e ".[sync]"

# Using uv
uv pip install -e ".[sync]"

# For development (includes sync dependencies)
uv sync --group sync-dev
```

## Operating Modes

The sync service supports two distinct operating modes to accommodate different use cases.

### One-Shot Mode

One-shot mode is the default behavior - it performs a single sync operation and then exits. This is ideal for:

- Scheduled cron jobs
- CI/CD pipelines
- Manual sync operations
- Serverless deployments

**Basic usage:**

```bash
# Basic sync
python scripts/sync/sync_github_references.py

# Preview changes without creating PRs
python scripts/sync/sync_github_references.py --dry-run

# Sync specific categories only
python scripts/sync/sync_github_references.py --categories image_generation,text_generation

# Override PRIMARY URL
python scripts/sync/sync_github_references.py --primary-url https://api.example.com

# Force PR creation (ignore threshold)
python scripts/sync/sync_github_references.py --force

# Use verbose logging
python scripts/sync/sync_github_references.py --verbose
```

### Watch Mode

Watch mode continuously monitors the PRIMARY server's metadata endpoint for changes and automatically triggers syncs when updates are detected. This is ideal for:

- Development environments
- Long-running Docker containers
- Kubernetes deployments
- Real-time synchronization needs

**How Watch Mode Works:**

1. **Metadata Polling**: Polls the `/model_references/v1/metadata/last_updated` endpoint at regular intervals
2. **Change Detection**: Compares the returned timestamp with the last known timestamp
3. **Automatic Sync**: Triggers a full sync operation when a change is detected
4. **Error Handling**: Retries on network errors with exponential backoff (up to 10 attempts)
5. **Graceful Shutdown**: Handles Ctrl+C and SIGTERM signals cleanly

**Watch mode usage:**

```bash
# Start watch mode with default 60-second polling interval
python scripts/sync/sync_github_references.py --watch

# Use custom polling interval (5 minutes)
python scripts/sync/sync_github_references.py --watch --watch-interval 300

# Run an initial sync on startup, then enter watch mode
python scripts/sync/sync_github_references.py --watch --watch-startup-sync

# Combine with dry-run for testing
python scripts/sync/sync_github_references.py --watch --watch-interval 30 --dry-run
```

**Watch mode configuration options:**

| CLI Argument | Environment Variable | Default | Description |
|--------------|---------------------|---------|-------------|
| `--watch` | `HORDE_GITHUB_SYNC_WATCH_MODE` | `false` | Enable watch mode |
| `--watch-interval` | `HORDE_GITHUB_SYNC_WATCH_INTERVAL_SECONDS` | `60` | Polling interval in seconds |
| `--watch-startup-sync` | `HORDE_GITHUB_SYNC_WATCH_ENABLE_STARTUP_SYNC` | `false` | Run sync on startup |

**Example watch mode output:**

```text
================================================================================
GitHub Model Reference Sync Service - WATCH MODE
================================================================================
Monitoring PRIMARY API: http://localhost:19800
Polling interval: 60 seconds
Initial delay: 0 seconds
Startup sync: disabled
================================================================================
Starting watch loop. Press Ctrl+C to stop.
--------------------------------------------------------------------------------
[2025-10-25 12:00:00] Initialized last known timestamp: 1729864800
[2025-10-25 12:01:00] No changes detected (timestamp: 1729864800)
[2025-10-25 12:02:00] Changes detected! Timestamp changed from 1729864800 to 1729864920
[2025-10-25 12:02:00] Triggering sync operation due to detected changes...
--------------------------------------------------------------------------------
[Sync output...]
--------------------------------------------------------------------------------
[2025-10-25 12:02:30] Sync completed successfully
[2025-10-25 12:02:30] Resuming watch loop...
```

## Configuration

Configure via environment variables with `HORDE_GITHUB_SYNC_` prefix.

### Docker Configuration (Recommended)

For Docker deployments, all configuration is loaded from `.env.sync` file using `env_file` directive.

**1. Copy the example configuration:**

```bash
cp .env.sync.example .env.sync
```

**2. Edit `.env.sync` and set required variables:**

```bash
# Required: PRIMARY API URL
HORDE_GITHUB_SYNC_PRIMARY_API_URL=https://stablehorde.net/api
```

**3. Configure authentication (choose one method):**

See the [GitHub Authentication](#github-authentication) section below for detailed authentication setup.

**4. For GitHub App authentication: Place your private key file:**

```bash
# Save your GitHub App private key as:
./github-app-key.pem
```

**5. For GitHub App: Uncomment volume mount in docker-compose.yml:**

Edit the `github-sync` service in your docker-compose file and uncomment:

```yaml
volumes:
  - github-sync-data:/data
  - ./github-app-key.pem:/app/github-app-key.pem:ro  # Uncomment this line
```

**6. Start with sync enabled:**

```bash
# Single-worker setup
docker-compose --profile sync up -d

# Multi-worker setup
docker-compose -f docker-compose.redis.yml --profile sync up -d
```

**That's it!** No need for `--env-file` parameter - the sync service automatically loads `.env.sync`.

**Alternative: Use the example compose file:**

```bash
# This includes proper GitHub App configuration
docker-compose -f docker-compose.sync.example.yml up -d
```

### Non-Docker Configuration

For non-Docker deployments, set environment variables directly:

**Required variables:**

```bash
# PRIMARY API URL (required)
export HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api"
```

Then configure authentication (see next section).

### GitHub Authentication

The sync service supports two authentication methods. Choose ONE:

#### Option A: Personal Access Token (Simpler)

Best for: Testing, personal use, small-scale deployments

```bash
# Set token via environment variable
export HORDE_GITHUB_SYNC_GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
# or
export GITHUB_TOKEN="ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

**Creating a Personal Access Token:**

1. Go to <https://github.com/settings/tokens>
2. Click "Generate new token" → "Generate new token (classic)"
3. Set a note (e.g., "Horde Model Reference Sync")
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. Copy the token immediately (you won't see it again)

**Required scopes:** `repo` (full control of private repositories)

#### Option B: GitHub App Installation (Recommended)

Best for: Production, organizations, higher rate limits, better security

```bash
# GitHub App credentials
export GITHUB_APP_ID="123456"
export GITHUB_APP_INSTALLATION_ID="12345678"

# Private key can be provided as file path or content
# Option B1: Path to private key file
export GITHUB_APP_PRIVATE_KEY_PATH="/path/to/private-key.pem"

# Option B2: Private key content directly (use \n for newlines)
export GITHUB_APP_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
```

**Note:** If both GitHub token and GitHub App credentials are configured, the GitHub App authentication will take precedence.

**Setting up a GitHub App:**

1. Go to your organization settings → Developer settings → GitHub Apps → New GitHub App
2. Configure the app:
   - Set a name (e.g., "Horde Model Reference Sync")
   - Set Homepage URL (your repo URL)
   - Uncheck "Active" under Webhook (not needed)
   - Set permissions:
     - Repository permissions:
       - Contents: Read and write
       - Pull requests: Read and write
3. Generate a private key and download the `.pem` file
4. Install the app to your organization/repository
5. Note the App ID (from app settings) and Installation ID (from installation URL)
6. Use these values in the environment variables above

**Getting the Installation ID:**

The Installation ID can be found in the URL when viewing the app installation:

```text
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

> **Detailed GitHub App Guide:** See [github_app_auth_example.md](github_app_auth_example.md) for comprehensive examples including Docker, Kubernetes, and GitHub Actions configurations.

### Optional Settings

```bash
# GitHub repositories (defaults shown)
export HORDE_GITHUB_SYNC_GITHUB_IMAGE_REPO="Haidra-Org/AI-Horde-image-model-reference"
export HORDE_GITHUB_SYNC_GITHUB_TEXT_REPO="Haidra-Org/AI-Horde-text-model-reference"
export HORDE_GITHUB_SYNC_GITHUB_BRANCH="main"

# Category filtering (comma-separated)
export HORDE_GITHUB_SYNC_CATEGORIES_TO_SYNC="image_generation,text_generation"

# PR configuration
export HORDE_GITHUB_SYNC_PR_REVIEWERS="user1,user2"
export HORDE_GITHUB_SYNC_PR_LABELS="automated,sync"
export HORDE_GITHUB_SYNC_PR_AUTO_ASSIGN_TEAM="org-name/team-name"

# Sync behavior
export HORDE_GITHUB_SYNC_MIN_CHANGES_THRESHOLD="1"
export HORDE_GITHUB_SYNC_TARGET_CLONE_DIR="/path/to/persistent/clones"

# Timeouts
export HORDE_GITHUB_SYNC_PRIMARY_API_TIMEOUT="30"

# Watch mode settings
export HORDE_GITHUB_SYNC_WATCH_MODE="true"
export HORDE_GITHUB_SYNC_WATCH_INTERVAL_SECONDS="60"
export HORDE_GITHUB_SYNC_WATCH_INITIAL_DELAY_SECONDS="0"
export HORDE_GITHUB_SYNC_WATCH_ENABLE_STARTUP_SYNC="false"

# Debugging
export HORDE_GITHUB_SYNC_DRY_RUN="false"
export HORDE_GITHUB_SYNC_VERBOSE_LOGGING="false"
```

### Testing with Forks

To test the sync service with your own forks:

```bash
# PowerShell
$env:HORDE_MODEL_REFERENCE_IMAGE_GITHUB_REPO_OWNER="your-username"
$env:HORDE_MODEL_REFERENCE_TEXT_GITHUB_REPO_OWNER="your-username"
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800 --dry-run

# Bash
export HORDE_MODEL_REFERENCE_IMAGE_GITHUB_REPO_OWNER="your-username"
export HORDE_MODEL_REFERENCE_TEXT_GITHUB_REPO_OWNER="your-username"
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800 --dry-run
```

### Command-Line Arguments Reference

```text
usage: sync_github_references.py [-h] [--primary-url PRIMARY_URL]
                                 [--categories CATEGORIES] [--dry-run]
                                 [--force] [--verbose] [--target-dir TARGET_DIR]
                                 [--watch] [--watch-interval WATCH_INTERVAL]
                                 [--watch-startup-sync]

options:
  -h, --help            Show help message
  --primary-url         PRIMARY API base URL
  --categories          Comma-separated list of categories to sync
  --dry-run             Preview changes without creating PRs
  --force               Create PR even if changes are below threshold
  --verbose             Enable verbose logging
  --target-dir          Base directory for persistent repo clones
  --watch               Enable watch mode
  --watch-interval      Polling interval in seconds (default: 60)
  --watch-startup-sync  Run sync immediately on startup
```

## Usage

### Common Use Cases

#### Development Server

For local development with automatic syncing:

```bash
# Terminal 1 - Run PRIMARY server
fastapi dev ./src/horde_model_reference/service/app.py --port 19800

# Terminal 2 - Run watch mode with dry-run
python scripts/sync/sync_github_references.py \
  --primary-url http://localhost:19800 \
  --watch \
  --watch-interval 30 \
  --dry-run \
  --verbose
```

#### Scheduled Sync (One-Shot)

For periodic syncs without watch mode:

```bash
# Run every 6 hours via cron
0 */6 * * * cd /path/to/horde-model-reference && /path/to/venv/bin/python scripts/sync/sync_github_references.py
```

#### CI/CD Integration

For automated syncs in CI/CD pipelines:

```bash
# GitHub Actions, GitLab CI, etc.
python scripts/sync/sync_github_references.py \
  --primary-url $PRIMARY_API_URL \
  --categories image_generation,text_generation \
  --verbose
```

#### Production Container (Watch Mode)

For long-running Docker containers:

```dockerfile
# In your Dockerfile
CMD ["python", "scripts/sync/sync_github_references.py", \
     "--watch", \
     "--watch-interval", "300", \
     "--watch-startup-sync"]
```

### Command Line Examples

**One-Shot Mode:**

```bash
# Basic sync
python scripts/sync/sync_github_references.py

# Dry run (preview changes without creating PRs)
python scripts/sync/sync_github_references.py --dry-run

# Sync specific categories
python scripts/sync/sync_github_references.py --categories image_generation,text_generation

# Verbose logging
python scripts/sync/sync_github_references.py --verbose

# Override PRIMARY URL
python scripts/sync/sync_github_references.py --primary-url https://api.example.com

# Force PR creation (ignore threshold)
python scripts/sync/sync_github_references.py --force

# Specify persistent clone directory
python scripts/sync/sync_github_references.py --target-dir /data/git-clones
```

**Watch Mode:**

```bash
# Start watch mode with defaults
python scripts/sync/sync_github_references.py --watch

# Custom polling interval (5 minutes)
python scripts/sync/sync_github_references.py --watch --watch-interval 300

# Run initial sync on startup
python scripts/sync/sync_github_references.py --watch --watch-startup-sync

# Watch mode with verbose logging and dry-run
python scripts/sync/sync_github_references.py --watch --verbose --dry-run

# Full example with all options
python scripts/sync/sync_github_references.py \
  --primary-url http://localhost:19800 \
  --watch \
  --watch-interval 120 \
  --watch-startup-sync \
  --categories image_generation \
  --verbose
```

### Programmatic Usage

```bash
# GitHub repositories (defaults shown)
export HORDE_GITHUB_SYNC_GITHUB_IMAGE_REPO="Haidra-Org/AI-Horde-image-model-reference"
export HORDE_GITHUB_SYNC_GITHUB_TEXT_REPO="Haidra-Org/AI-Horde-text-model-reference"
export HORDE_GITHUB_SYNC_GITHUB_BRANCH="main"

# Category filtering
export HORDE_GITHUB_SYNC_CATEGORIES_TO_SYNC="image_generation,text_generation"

# PR configuration
export HORDE_GITHUB_SYNC_PR_REVIEWERS="user1,user2"
export HORDE_GITHUB_SYNC_PR_LABELS="automated,sync"
export HORDE_GITHUB_SYNC_PR_AUTO_ASSIGN_TEAM="org-name/team-name"

# Sync behavior
export HORDE_GITHUB_SYNC_MIN_CHANGES_THRESHOLD="1"

# Timeouts
export HORDE_GITHUB_SYNC_PRIMARY_API_TIMEOUT="30"

# Debugging
export HORDE_GITHUB_SYNC_DRY_RUN="false"
export HORDE_GITHUB_SYNC_VERBOSE_LOGGING="false"
```

## Usage

### Command Line

Basic usage:

```bash
python scripts/sync/sync_github_references.py
```

With options:

```bash
# Dry run (preview changes without creating PRs)
python scripts/sync/sync_github_references.py --dry-run

# Sync specific categories
python scripts/sync/sync_github_references.py --categories image_generation,text_generation

# Verbose logging
python scripts/sync/sync_github_references.py --verbose

# Override PRIMARY URL
python scripts/sync/sync_github_references.py --primary-url https://api.example.com

# Force PR creation (ignore threshold)
python scripts/sync/sync_github_references.py --force
```

### Programmatic Usage

```python
from horde_model_reference import MODEL_REFERENCE_CATEGORY
from horde_model_reference.sync import (
    GitHubSyncClient,
    ModelReferenceComparator,
)
from horde_model_reference.sync.config import github_app_settings, github_sync_settings

# Check authentication configuration
if github_app_settings.is_configured():
    print("Using GitHub App authentication")
    print(f"App ID: {github_app_settings.github_app_id}")
    print(f"Installation ID: {github_app_settings.github_installation_id}")
elif github_sync_settings.github_token:
    print("Using token authentication")
else:
    print("No authentication configured")

# Create comparator
comparator = ModelReferenceComparator()

# Create GitHub client (automatically selects best auth method)
with GitHubSyncClient() as client:
    # Fetch data
    primary_data = {...}  # Fetch from PRIMARY v1 API
    github_data = {...}   # Fetch from GitHub

    # Compare
    diff = comparator.compare_categories(
        category=MODEL_REFERENCE_CATEGORY.image_generation,
        primary_data=primary_data,
        github_data=github_data,
    )

    # Sync if changes detected
    if diff.has_changes():
        pr_url = client.sync_category_to_github(
            category=MODEL_REFERENCE_CATEGORY.image_generation,
            diff=diff,
            primary_data=primary_data,
        )
        print(f"PR created: {pr_url}")
```

## Deployment

### Scheduled Cron Job (Recommended for One-Shot)

Run every 6 hours:

```bash
# Add to crontab
0 */6 * * * cd /path/to/horde-model-reference && /path/to/venv/bin/python scripts/sync/sync_github_references.py

# Or with explicit configuration
0 */6 * * * cd /path/to/horde-model-reference && \
  HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api" \
  GITHUB_TOKEN="ghp_xxxx" \
  /path/to/venv/bin/python scripts/sync/sync_github_references.py
```

### GitHub Actions

Create `.github/workflows/sync-from-primary.yml`:

```yaml
name: Sync from PRIMARY
on:
  schedule:
    - cron: '0 */6 * * *'  # Every 6 hours
  workflow_dispatch:  # Manual trigger

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

### Systemd Service and Timer

For Linux servers running one-shot mode on a schedule:

**Create `/etc/systemd/system/horde-github-sync.service`:**

```ini
[Unit]
Description=Horde Model Reference GitHub Sync
After=network.target

[Service]
Type=simple
User=horde-sync
WorkingDirectory=/opt/horde-model-reference
ExecStart=/opt/horde-model-reference/.venv/bin/python scripts/sync/sync_github_references.py
Restart=on-failure
RestartSec=3600
EnvironmentFile=/etc/horde-github-sync/env

[Install]
WantedBy=multi-user.target
```

**Create `/etc/systemd/system/horde-github-sync.timer`:**

```ini
[Unit]
Description=Horde GitHub Sync Timer

[Timer]
OnCalendar=*-*-* 0/6:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

**Create `/etc/horde-github-sync/env`:**

```bash
HORDE_GITHUB_SYNC_PRIMARY_API_URL=https://stablehorde.net/api
GITHUB_TOKEN=ghp_your_token_here
```

**Enable and start:**

```bash
sudo systemctl enable horde-github-sync.timer
sudo systemctl start horde-github-sync.timer

# Check timer status
sudo systemctl list-timers horde-github-sync.timer

# Check service logs
sudo journalctl -u horde-github-sync.service -f
```

### Docker Deployment (Watch Mode)

For long-running containers using watch mode:

**docker-compose.yml:**

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
      - HORDE_GITHUB_SYNC_WATCH_MODE=true
      - HORDE_GITHUB_SYNC_WATCH_INTERVAL_SECONDS=300
      - HORDE_GITHUB_SYNC_WATCH_ENABLE_STARTUP_SYNC=true
    volumes:
      - ./github-app-key.pem:/secrets/github-app-key.pem:ro
      - github-sync-data:/data
    command: python scripts/sync/sync_github_references.py --watch
    restart: unless-stopped

volumes:
  github-sync-data:
```

### Kubernetes Deployment (Watch Mode)

**Secret:**

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
    -----END RSA PRIVATE KEY-----
```

**Deployment:**

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
        command: ["python", "scripts/sync/sync_github_references.py", "--watch", "--watch-interval", "300"]
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
        - name: HORDE_GITHUB_SYNC_WATCH_ENABLE_STARTUP_SYNC
          value: "true"
```

## Components

The sync service is composed of four main components:

### 1. ModelReferenceComparator (`src/horde_model_reference/sync/comparator.py`)

Compares PRIMARY and GitHub data to detect differences:

- **Added models**: Present in PRIMARY but not in GitHub
- **Removed models**: Present in GitHub but not in PRIMARY
- **Modified models**: Present in both but with different content

PRIMARY data is the source of truth and replaces GitHub data entirely during sync.

### 2. GitHubSyncClient (`src/horde_model_reference/sync/github_client.py`)

Automates PR creation workflow:

1. Clone GitHub repository
2. Create sync branch
3. Update category files with PRIMARY data
4. Commit changes with detailed message
5. Push branch to remote
6. Create pull request with summary

Features:

- Automatic cleanup of temporary directories
- Context manager support
- Detailed PR descriptions with change summaries
- Configurable reviewers, labels, and team assignments
- Support for both GitHub App and token authentication

### 3. HordeGitHubSyncSettings (`src/horde_model_reference/sync/config.py`)

Pydantic settings for configuration:

- Environment variable support with `HORDE_GITHUB_SYNC_` prefix
- Validation and helpful warnings
- Category filtering
- PR configuration
- Repository mapping
- Watch mode settings

### 4. CLI Script (`scripts/sync/sync_github_references.py`)

Command-line interface for sync operations:

- Fetches data from PRIMARY v1 API
- Fetches data from GitHub
- Compares and detects drift
- Creates PRs when needed
- Supports one-shot and watch modes
- Comprehensive logging and error handling

### 5. WatchModeManager (Part of CLI Script)

Manages continuous monitoring in watch mode:

- Polls PRIMARY metadata endpoint for changes
- Detects timestamp updates
- Triggers automatic syncs
- Handles errors with exponential backoff
- Graceful shutdown handling

## Testing

Run the test suite:

```bash
# Run all sync tests
pytest tests/sync/

# Run specific test file
pytest tests/sync/test_comparator.py

# Run with coverage
pytest tests/sync/ --cov=horde_model_reference.sync --cov-report=html

# Run in verbose mode
pytest tests/sync/ -v
```

Test coverage includes:

- **Comparator**: All diff detection scenarios (added, removed, modified, nested changes)
- **Configuration**: Settings validation, environment variables, repository mapping
- **Integration**: Mock GitHub client interactions (requires manual testing for full integration)

## Troubleshooting

### Common Configuration Issues

#### "PRIMARY API URL is not configured"

Set the `HORDE_GITHUB_SYNC_PRIMARY_API_URL` environment variable:

```bash
export HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api"
```

#### "GitHub token is not configured"

Set a GitHub personal access token with repo write permissions OR configure GitHub App authentication:

##### Option 1: Personal Access Token

```bash
export HORDE_GITHUB_SYNC_GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
# or
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

Create token at: <https://github.com/settings/tokens>

Required scopes: `repo` (full control of private repositories)

##### Option 2: GitHub App (Recommended)

```bash
export GITHUB_APP_ID="123456"
export GITHUB_APP_INSTALLATION_ID="12345678"
export GITHUB_APP_PRIVATE_KEY_PATH="/path/to/private-key.pem"
```

See the [GitHub Authentication](#github-authentication) section above for detailed setup instructions.

### GitHub Operations Issues

#### "Failed to clone repository"

Ensure:

- GitHub token has correct permissions
- Repository name is in `owner/repo` format
- Network connectivity to GitHub

#### "Failed to create PR"

Check:

- Branch was pushed successfully
- Token has PR creation permissions
- No existing PR from the same branch
- Base branch exists in repository

### GitHub App Authentication Issues

#### "GitHub App settings are not fully configured"

Ensure all three environment variables are set:

- `GITHUB_APP_ID`
- `GITHUB_APP_INSTALLATION_ID`
- `GITHUB_APP_PRIVATE_KEY` or `GITHUB_APP_PRIVATE_KEY_PATH`

#### "Failed to read private key from file"

- Check that the file path is correct
- Ensure the file exists and is readable
- Verify the file contains a valid PEM-formatted private key

#### "GitHub App authentication failed"

- Verify the App ID is correct
- Verify the Installation ID is correct (check the installation URL)
- Ensure the private key matches the GitHub App
- Check that the app is installed to the correct organization/repository
- Verify the app has the required permissions (Contents: Read & Write, Pull requests: Read & Write)

### Watch Mode Issues

#### Watch Mode Not Detecting Changes

1. Verify the PRIMARY server is running and accessible
2. Check that metadata tracking is enabled (canonical backend required)
3. Ensure the metadata endpoint returns valid timestamps:

   ```bash
   curl http://localhost:19800/model_references/v1/metadata/last_updated
   ```

#### Connection Errors

Watch mode will retry on network errors up to 10 times with exponential backoff. If errors persist:

1. Check network connectivity to PRIMARY server
2. Verify firewall rules allow connections
3. Check PRIMARY server logs for errors
4. Consider increasing `--watch-interval` to reduce network load

#### Excessive Network Usage

If the polling interval is too aggressive:

```bash
# Increase polling interval to 5 minutes
python scripts/sync/sync_github_references.py --watch --watch-interval 300
```

#### GitHub Rate Limiting

If you're hitting GitHub rate limits:

1. Use a GitHub App instead of personal access token (higher rate limits)
2. Increase `--watch-interval` to reduce API calls
3. Use `--categories` to filter synced categories
4. Check your rate limit status:

   ```bash
   curl -H "Authorization: token YOUR_TOKEN" https://api.github.com/rate_limit
   ```

### Testing and Debugging

#### Dry run for testing

Test sync without creating PRs:

```bash
# One-shot dry run
python scripts/sync/sync_github_references.py --dry-run --verbose

# Watch mode dry run
HORDE_GITHUB_SYNC_DRY_RUN=true python scripts/sync/sync_github_references.py --watch --verbose
```

#### Testing Authentication

Test that GitHub App authentication is working:

```python
from github import Auth, Github
from horde_model_reference.sync.config import github_app_settings

if not github_app_settings.is_configured():
    print("GitHub App not configured")
    exit(1)

try:
    private_key = github_app_settings.get_private_key_content()
    app_auth = Auth.AppAuth(
        app_id=github_app_settings.github_app_id,
        private_key=private_key,
    )
    auth = app_auth.get_installation_auth(
        installation_id=github_app_settings.github_installation_id,
    )
    g = Github(auth=auth)
    user = g.get_user()
    print(f"Authenticated as: {user.login}")
    print("GitHub App authentication is working!")
except Exception as e:
    print(f"GitHub App authentication failed: {e}")
    exit(1)
```

## Security Considerations

1. **GitHub Token**: Store securely, never commit to version control
2. **Environment Variables**: Use `.env` file or secrets management (e.g., Kubernetes Secrets, AWS Secrets Manager)
3. **PR Permissions**: Use least-privilege tokens (repo scope only for tokens, minimal permissions for GitHub Apps)
4. **Review PRs**: Always review automated PRs before merging to catch any unexpected changes
5. **Rate Limits**: GitHub API has rate limits; sync service respects them but monitor usage
6. **Private Keys**: For GitHub App authentication, protect private keys with file permissions (chmod 600) and never commit
7. **Watch Mode**: When running in watch mode, ensure the PRIMARY server is trusted and secure

## Additional Resources

- **GitHub App Authentication Guide**: See [github_app_auth_example.md](github_app_auth_example.md) for detailed examples including Docker, Kubernetes, and GitHub Actions configurations
- **PRIMARY Server Deployment**: See [DEPLOYMENT.md](../../DEPLOYMENT.md) for running a PRIMARY server
- **Development Documentation**: See [CLAUDE.md](../../CLAUDE.md) for development guidelines
- **Sync Implementation Details**: See source code in `src/horde_model_reference/sync/`

## License

AGPL-3.0 - See LICENSE file for details
