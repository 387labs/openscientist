# Contributing to OpenScientist

## Prerequisites

- Python 3.12+
- Docker
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Environment Setup

⚠️ **Required before running tests or the application**

Create a `.env` file with database configuration:

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and uncomment the DATABASE_URL line
# For local development, use:
# DATABASE_URL=postgresql+asyncpg://openscientist:openscientist_dev_password@localhost:5434/openscientist

# Start the database with Docker
docker compose -f docker-compose.yml up -d postgres
```

See the README for provider-specific settings if you need to configure Claude API access.

## Local Development

```bash
# Install all dependencies (including dev tools)
uv sync

# Run tests (requires .env to be configured)
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/openscientist --cov-report=term-missing

# Run webapp tests
uv run pytest tests/webapp/

# Type checking
uv run mypy src/openscientist/ tests/

# Lint and format
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Legacy Job Migration (Filesystem -> DB)

Use this when migrating old on-disk jobs (from pre-user versions) into the
database-backed model.

### Prerequisites

- Database is running (for local dev: `docker compose -f docker-compose.yml up -d postgres`)
- Dependencies are installed (`uv sync`)
- Legacy job folders exist (default path: `jobs/`)

### Trigger Migration

```bash
# Safe preview (no database writes)
uv run python -m openscientist.job_manager bootstrap --jobs-dir jobs --dry-run

# Apply migration
uv run python -m openscientist.job_manager bootstrap --jobs-dir jobs
```

The bootstrap command currently supports:
- `--jobs-dir`
- `--dry-run`


## Docker

```bash
# Build images
make build

# Start services
make start

# View logs
make logs

# Restart / stop
make restart
make stop
```

The app runs at <http://localhost:8080>.

## Docker (Production / Deploy)

```bash
# Build and start
make rebuild

# Deploy to remote server (pulls latest code, rebuilds, restarts)
make deploy                          # default host
make deploy DEPLOY_HOST=myserver     # custom host
```

The remote server must have the repo cloned and a `.env` file configured (see `.env.example`).

## Code Quality

All PRs must pass:

```bash
uv run ruff check src/ tests/   # lint
uv run mypy src/openscientist/ tests/  # types
uv run pytest                   # tests (67% coverage minimum, see pyproject.toml)
```

CI (`.github/workflows/ci.yml`) also runs on every PR and blocks merging on:

- **Secret scanning** (gitleaks) over the PR's commits
- **Dependency vulnerability scanning** — `pip-audit` against installed Python packages, plus `dependency-review-action` (fails on high/critical severity)
- **Docker build validation** — hadolint on all Dockerfiles, plus full builds of `Dockerfile.base`, `Dockerfile.executor`, and `Dockerfile.agent` when Docker-relevant files change
- **Coverage delta** — fails if this branch's coverage drops more than 0.5 points below `main`'s

Coverage reports (XML, JSON, HTML) are uploaded as a workflow artifact on every run.

## Git Hooks

Hooks are managed by [pre-commit](https://pre-commit.com) (see `.pre-commit-config.yaml`). Install both hook types once after cloning:

```bash
uv run pre-commit install                     # runs lint/type/test checks on commit
uv run pre-commit install --hook-type pre-push # blocks direct pushes to main
```
