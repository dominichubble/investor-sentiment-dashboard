# GitHub Actions Workflows

## 📊 Automated Data Collection

### `collect-data.yml`

**Purpose:** Automatically collect financial sentiment data from Reddit and News APIs daily.

**Schedule:** 
- Runs daily at 6:00 AM UTC (7:00 AM UK time in winter)
- Can be manually triggered from the Actions tab

**What it does:**
1. Collects up to 100 Reddit posts from financial subreddits
2. Collects news articles from the past 24 hours
3. Preprocesses all data with FinBERT-optimized configuration
4. Commits processed data back to the repository

**Requirements:**

You need to set up the following secrets in your GitHub repository:

**Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Description | How to Get |
|-------------|-------------|------------|
| `REDDIT_CLIENT_ID` | Reddit API client ID | https://www.reddit.com/prefs/apps |
| `REDDIT_CLIENT_SECRET` | Reddit API secret | Same as above |
| `REDDIT_USER_AGENT` | Reddit API user agent | Use: `investor-sentiment-dashboard/1.0` |
| `NEWS_API_KEY` | NewsAPI.org API key | https://newsapi.org/register |

#### Setting Up Secrets

1. Go to your repository on GitHub
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Add each secret with the exact name and value

#### Manual Trigger

To run the workflow manually:

1. Go to **Actions** tab in your GitHub repository
2. Select **Daily Financial Data Collection** workflow
3. Click **Run workflow** button
4. Select branch (usually `main` or `dev`)
5. Click **Run workflow**

#### Monitoring

- View workflow runs in the **Actions** tab
- Each run shows a summary of files collected
- Workflow continues even if one source fails
- Data is automatically committed to the `data/` directory

#### Benefits

✅ **Free** - 2,000 minutes/month on GitHub Free  
✅ **Automated** - No PC needed  
✅ **Reliable** - Runs in the cloud  
✅ **Versioned** - All data changes tracked in Git  
✅ **Transparent** - Full logs of each collection  

---

# GitHub Actions Workflows

This directory contains automated workflows for CI/CD and code quality.

## Workflows

### 🧪 Backend CI (`backend-ci.yml`)
**Triggers:** PRs and pushes to `dev`/`main` affecting backend code

**What it does:**
- Runs all backend tests with coverage reporting
- Checks code formatting (Black)
- Checks import sorting (isort)
- Lints code for errors (flake8)
- Type checks with mypy
- Uploads coverage to Codecov

**Requirements:** All tests must pass and code must be properly formatted

---

### ✅ PR Checks (`pr-checks.yml`)
**Triggers:** All pull requests to `dev`/`main`

**What it does:**
- Validates PR title follows conventional commits format:
  - `feat: description` - New feature
  - `fix: description` - Bug fix
  - `docs: description` - Documentation
  - `test: description` - Tests
  - `refactor: description` - Code refactoring
  - `chore: description` - Maintenance
- Ensures PR has meaningful description (min 20 characters)
- Checks branch naming convention (warning only)

**Requirements:** PR title must follow format, PR must have description

---

### 🎨 Auto-format (`auto-format.yml`)
**Triggers:** PRs opened/updated to `dev`/`main`

**What it does:**
- Automatically formats Python code with Black (88 char line length)
- Sorts imports with isort
- Commits changes back to PR branch

**Note:** Only works on PRs from branches in same repository (not forks)

---

### 🔒 Dependency Review (`dependency-review.yml`)
**Triggers:** All pull requests

**What it does:**
- Reviews dependency changes for security vulnerabilities
- Fails on moderate or higher severity issues
- Posts summary comment on PR

---

## Local Development

### Install formatting tools
```bash
cd backend
pip install black isort flake8 mypy
```

### Format code before committing
```bash
# Format with Black
black app/ tests/

# Sort imports
isort app/ tests/

# Check for issues
flake8 app/ tests/
mypy app/ --ignore-missing-imports
```

### Run tests locally
```bash
cd backend
python -m pytest tests/ -v --cov=app
```

## Configuration Files

- `.flake8` - Flake8 linting configuration
- `pyproject.toml` - Black and isort configuration (if added)

## Best Practices

1. **Before creating a PR:**
   - Run tests locally
   - Format code with Black and isort
   - Check for linting issues

2. **PR Title Examples:**
   - ✅ `feat: add Twitter data ingestion pipeline`
   - ✅ `fix(reddit): handle deleted authors correctly`
   - ✅ `test: add unit tests for sentiment analysis`
   - ❌ `updated files` (too vague)
   - ❌ `Fixed bug` (missing conventional commit type)

3. **Branch Naming:**
   - Use Jira ticket format: `FYP-146-Implement-Reddit-data-pipeline`
   - Or conventional format: `feat/add-twitter-pipeline`

## Troubleshooting

### Tests fail in CI but pass locally
- Ensure all dependencies are in `requirements.txt`
- Check Python version matches (3.11)
- Verify environment variables aren't needed

### Auto-format doesn't commit
- Only works on branches in same repository (not forks)
- Check GitHub Actions permissions are enabled

### Type checking fails
- Add `# type: ignore` for unavoidable issues
- Or use `--ignore-missing-imports` flag
