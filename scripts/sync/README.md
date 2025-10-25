# GitHub Sync Service

Automated tool for keeping GitHub legacy model reference repositories in sync with the PRIMARY instance.

## Overview

The GitHub sync service compares model references from a PRIMARY instance (v1 API) with GitHub legacy repositories and automatically creates pull requests when drift is detected. This ensures GitHub repos stay up-to-date with the canonical PRIMARY source.

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

## Configuration

Configure via environment variables with `HORDE_GITHUB_SYNC_` prefix.

### Docker Configuration (Recommended)

For Docker deployments, all configuration is loaded from `.env.sync` file using `env_file` directive.

**1. Copy the example configuration:**

```bash
cp .env.sync.example .env.sync
```

**2. Edit `.env.sync` and configure authentication:**

Choose ONE authentication method:

#### Option A: GitHub Personal Access Token

```bash
# In .env.sync:
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Create token at: <https://github.com/settings/tokens>

Required scopes: `repo` (full control of private repositories)

#### Option B: GitHub App (Recommended for Organizations)

> See [a detailed guide for app authentication here](github_app_auth_example.md).

```bash
# In .env.sync:
GITHUB_APP_ID=1234567
GITHUB_APP_INSTALLATION_ID=12345678
GITHUB_APP_PRIVATE_KEY_PATH=/app/github-app-key.pem
```

**3. For GitHub App: Place your private key file:**

```bash
# Save your GitHub App private key as:
./github-app-key.pem
```

**4. For GitHub App: Uncomment volume mount in docker-compose.yml:**

Edit the `github-sync` service in your docker-compose file and uncomment:

```yaml
volumes:
  - github-sync-data:/data
  - ./github-app-key.pem:/app/github-app-key.pem:ro  # Uncomment this line
```

**5. Start with sync enabled:**

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

#### GitHub Personal Access Token

```bash
# PRIMARY API URL (required)
export HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api"

# GitHub token with repo write permissions (required for PR creation)
export HORDE_GITHUB_SYNC_GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
# or
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

#### GitHub App Installation (Recommended for Organizations)

GitHub App authentication provides better security and more granular permissions than personal access tokens. It's especially recommended for organization-wide automation.

```bash
# PRIMARY API URL (required)
export HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api"

# GitHub App credentials
export GITHUB_APP_ID="123456"
export GITHUB_APP_INSTALLATION_ID="12345678"

# Private key can be provided as file path or content
# Option A: Path to private key file
export GITHUB_APP_PRIVATE_KEY_PATH="/path/to/private-key.pem"

# Option B: Private key content directly (use \n for newlines)
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

### Optional Settings

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

# Create comparator
comparator = ModelReferenceComparator()

# Create GitHub client
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

### Scheduled Cron Job (Recommended)

Run every 6 hours:

```bash
# Add to crontab
0 */6 * * * cd /path/to/horde-model-reference && /path/to/venv/bin/python scripts/sync/sync_github_references.py
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
        run: |
          pip install -e ".[sync]"

      - name: Run sync
        env:
          HORDE_GITHUB_SYNC_PRIMARY_API_URL: ${{ secrets.PRIMARY_API_URL }}
          HORDE_GITHUB_SYNC_GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python scripts/sync/sync_github_references.py
```

### Systemd Service

Create `/etc/systemd/system/horde-github-sync.service`:

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

Create `/etc/systemd/system/horde-github-sync.timer`:

```ini
[Unit]
Description=Horde GitHub Sync Timer

[Timer]
OnCalendar=*-*-* 0/6:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:

```bash
sudo systemctl enable horde-github-sync.timer
sudo systemctl start horde-github-sync.timer
```

## Components

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

### 3. HordeGitHubSyncSettings (`src/horde_model_reference/sync/config.py`)

Pydantic settings for configuration:

- Environment variable support with `HORDE_GITHUB_SYNC_` prefix
- Validation and helpful warnings
- Category filtering
- PR configuration
- Repository mapping

### 4. CLI Script (`scripts/sync/sync_github_references.py`)

Command-line interface for sync operations:

- Fetches data from PRIMARY v1 API
- Fetches data from GitHub
- Compares and detects drift
- Creates PRs when needed
- Comprehensive logging and error handling

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

### "PRIMARY API URL is not configured"

Set the `HORDE_GITHUB_SYNC_PRIMARY_API_URL` environment variable:

```bash
export HORDE_GITHUB_SYNC_PRIMARY_API_URL="https://stablehorde.net/api"
```

### "GitHub token is not configured"

Set a GitHub personal access token with repo write permissions OR configure GitHub App authentication:

**Option 1: Personal Access Token**
```bash
export HORDE_GITHUB_SYNC_GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
# or
export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"
```

Create token at: <https://github.com/settings/tokens>

Required scopes: `repo` (full control of private repositories)

**Option 2: GitHub App (Recommended)**
```bash
export GITHUB_APP_ID="123456"
export GITHUB_APP_INSTALLATION_ID="12345678"
export GITHUB_APP_PRIVATE_KEY_PATH="/path/to/private-key.pem"
```

See the "GitHub App Installation" section above for setup instructions.

### "Failed to clone repository"

Ensure:
- GitHub token has correct permissions
- Repository name is in `owner/repo` format
- Network connectivity to GitHub

### "Failed to create PR"

Check:
- Branch was pushed successfully
- Token has PR creation permissions
- No existing PR from the same branch
- Base branch exists in repository

### Dry run for testing

Test sync without creating PRs:

```bash
HORDE_GITHUB_SYNC_DRY_RUN=true python scripts/sync/sync_github_references.py --verbose
```

## Security Considerations

1. **GitHub Token**: Store securely, never commit to version control
2. **Environment Variables**: Use `.env` file or secrets management
3. **PR Permissions**: Use least-privilege tokens (repo scope only)
4. **Review PRs**: Always review automated PRs before merging
5. **Rate Limits**: GitHub API has rate limits; sync service respects them

## Contributing

When contributing to the sync service:

1. Follow existing code style (enforced by ruff/black)
2. Add type hints (mypy strict mode)
3. Write tests for new functionality
4. Update documentation as needed
5. Test with dry run before real PRs

## License

AGPL-3.0 - See LICENSE file for details
