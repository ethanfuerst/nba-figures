# Contributing

## Setup

```bash
git clone https://github.com/ethanfuerst/nba-figures.git
cd nba-figures
uv sync --extra dev
uv run pre-commit install --hook-type commit-msg --hook-type pre-commit
```

## Commit Messages

This project uses [conventional commits](https://www.conventionalcommits.org/). All commit messages must follow the format:

```
type(scope): description
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`

The commitizen pre-commit hook validates this automatically.

## Running Tests

```bash
uv run pytest
```

## Linting

```bash
uv run pre-commit run --all-files
```
