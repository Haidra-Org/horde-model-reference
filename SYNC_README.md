# GitHub Model Reference Sync

This document describes the GitHub synchronization service for keeping legacy repositories in sync with a PRIMARY server instance.

## Overview

The sync service can run in two modes:

1. **One-shot mode** (default): Runs a single sync operation and exits
2. **Watch mode** (`--watch`): Continuously monitors the PRIMARY server for changes and syncs automatically

## One-Shot Mode

One-shot mode is the default behavior. It performs a single sync operation:

```bash
# Basic usage
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800

# Dry run (preview changes without creating PRs)
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800 --dry-run

# Sync specific categories only
python scripts/sync/sync_github_references.py --categories image_generation,text_generation
```

## Watch Mode

Watch mode continuously monitors the PRIMARY server's metadata endpoint for changes and automatically triggers syncs when updates are detected.

### Basic Watch Mode Usage

```bash
# Start watch mode with default 60-second polling interval
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800 --watch

# Use custom polling interval (5 minutes)
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800 --watch --watch-interval 300

# Run an initial sync on startup, then enter watch mode
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800 --watch --watch-startup-sync
```

### How Watch Mode Works

1. **Metadata Polling**: The script polls the `/model_references/v1/metadata/last_updated` endpoint at regular intervals
2. **Change Detection**: Compares the returned timestamp with the last known timestamp
3. **Automatic Sync**: When a change is detected, triggers a full sync operation
4. **Error Handling**: Retries on network errors with exponential backoff
5. **Graceful Shutdown**: Handles Ctrl+C and SIGTERM signals cleanly

### Watch Mode Configuration

Watch mode can be configured via command-line arguments or environment variables:

| CLI Argument | Environment Variable | Default | Description |
|--------------|---------------------|---------|-------------|
| `--watch` | `HORDE_GITHUB_SYNC_WATCH_MODE` | `false` | Enable watch mode |
| `--watch-interval` | `HORDE_GITHUB_SYNC_WATCH_INTERVAL_SECONDS` | `60` | Polling interval in seconds |
| `--watch-startup-sync` | `HORDE_GITHUB_SYNC_WATCH_ENABLE_STARTUP_SYNC` | `false` | Run sync on startup |

### Watch Mode Example

```bash
# PowerShell - set up and run watch mode
$env:HORDE_GITHUB_SYNC_PRIMARY_API_URL="http://localhost:19800"
$env:HORDE_GITHUB_SYNC_GITHUB_TOKEN="ghp_your_token_here"
$env:HORDE_GITHUB_SYNC_WATCH_INTERVAL_SECONDS="120"
python scripts/sync/sync_github_references.py --watch --watch-startup-sync
```

### Watch Mode Output

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

### Required Environment Variables

```bash
# PRIMARY server API URL
HORDE_GITHUB_SYNC_PRIMARY_API_URL=https://your-primary-server.com/api

# GitHub authentication (use one of these methods)
HORDE_GITHUB_SYNC_GITHUB_TOKEN=ghp_your_token_here
# OR
GITHUB_TOKEN=ghp_your_token_here
```

### Optional Environment Variables

```bash
# Filter categories to sync (comma-separated)
HORDE_GITHUB_SYNC_CATEGORIES_TO_SYNC=image_generation,text_generation

# Dry run mode (no PRs created)
HORDE_GITHUB_SYNC_DRY_RUN=true

# Verbose logging
HORDE_GITHUB_SYNC_VERBOSE_LOGGING=true

# Minimum changes required to create PR
HORDE_GITHUB_SYNC_MIN_CHANGES_THRESHOLD=1

# Base directory for persistent repo clones
HORDE_GITHUB_SYNC_TARGET_CLONE_DIR=/path/to/clones

# PR configuration
HORDE_GITHUB_SYNC_PR_REVIEWERS=username1,username2
HORDE_GITHUB_SYNC_PR_LABELS=automated,sync,ready-for-review
HORDE_GITHUB_SYNC_PR_AUTO_ASSIGN_TEAM=org-name/team-name

# Watch mode settings
HORDE_GITHUB_SYNC_WATCH_MODE=true
HORDE_GITHUB_SYNC_WATCH_INTERVAL_SECONDS=60
HORDE_GITHUB_SYNC_WATCH_INITIAL_DELAY_SECONDS=0
HORDE_GITHUB_SYNC_WATCH_ENABLE_STARTUP_SYNC=false
```

### Testing with Forks

To test the sync service with your own forks:

```bash
# PowerShell
$env:HORDE_MODEL_REFERENCE_IMAGE_GITHUB_REPO_OWNER="your-username"
$env:HORDE_MODEL_REFERENCE_TEXT_GITHUB_REPO_OWNER="your-username"
python scripts/sync/sync_github_references.py --primary-url http://localhost:19800 --dry-run
```

### Command-Line Arguments

```texttext
usage: sync_github_references.py [-h] [--primary-url PRIMARY_URL]
                                 [--categories CATEGORIES] [--dry-run]
                                 [--force] [--target-dir TARGET_DIR]
                                 [--watch] [--watch-interval WATCH_INTERVAL]
                                 [--watch-startup-sync]

options:
  -h, --help            Show help message
  --primary-url         PRIMARY API base URL
  --categories          Comma-separated list of categories to sync
  --dry-run             Preview changes without creating PRs
  --force               Create PR even if changes are below threshold
  --target-dir          Base directory for persistent repo clones
  --watch               Enable watch mode
  --watch-interval      Polling interval in seconds (default: 60)
  --watch-startup-sync  Run sync immediately on startup
```

## Use Cases

### Development Server

For local development with automatic syncing:

```bash
# Terminal 1 - Run PRIMARY server
fastapi dev ./src/horde_model_reference/service/app.py --port 19800

# Terminal 2 - Run watch mode
python scripts/sync/sync_github_references.py \
  --primary-url http://localhost:19800 \
  --watch \
  --watch-interval 30 \
  --dry-run
```

### Production Deployment

For production deployments (e.g., Docker container):

```dockerfile
# In your Dockerfile
CMD ["python", "scripts/sync/sync_github_references.py", \
     "--watch", \
     "--watch-interval", "300", \
     "--watch-startup-sync"]
```

### CI/CD Integration

For scheduled syncs in CI/CD pipelines:

```yaml
# GitHub Actions example
- name: Sync Model References
  run: |
    python scripts/sync/sync_github_references.py \
      --primary-url ${{ secrets.PRIMARY_API_URL }} \
      --dry-run
```

## Troubleshooting

### Watch Mode Not Detecting Changes

1. Verify the PRIMARY server is running and accessible
2. Check that metadata tracking is enabled (canonical backend required)
3. Ensure the metadata endpoint returns valid timestamps:

   ```bash
   curl http://localhost:19800/model_references/v1/metadata/last_updated
   ```

### Connection Errors

Watch mode will retry on network errors up to 10 times. If errors persist:

1. Check network connectivity to PRIMARY server
2. Verify firewall rules allow connections
3. Check PRIMARY server logs for errors

### Excessive Network Usage

If the polling interval is too aggressive:

```bash
# Increase polling interval to 5 minutes
python scripts/sync/sync_github_references.py --watch --watch-interval 300
```

### GitHub Rate Limiting

If you're hitting GitHub rate limits:

1. Use a GitHub App instead of personal access token (higher limits)
2. Increase `--watch-interval` to reduce API calls
3. Use `--categories` to filter synced categories

## Architecture

The sync service consists of:

1. **GithubSynchronizer**: Fetches data from PRIMARY and GitHub
2. **ModelReferenceComparator**: Compares and detects differences
3. **GitHubSyncClient**: Creates PRs with detected changes
4. **WatchModeManager**: Monitors metadata and triggers syncs

## See Also

- [DEPLOYMENT.md](DEPLOYMENT.md) - Running a PRIMARY server
- [CLAUDE.md](CLAUDE.md) - Development documentation
- [scripts/sync/](scripts/sync/) - Sync implementation details
