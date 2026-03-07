# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0]

### Added

- CLI (`nbafigs/cli.py`) with `player` and `team` subcommands
- Python API: `player_shot_chart()` and `team_shot_chart()` in `nbafigs/__init__.py`
- Normal and hex shot chart modes
- Zone grouping via `shots_grouper()` in `nbafigs/viz/chart.py`
- Retry and rate limiting for all nba_api calls via `sources/nba_api_call()`
- Player shot chart filters: opponent, location, date range, period, season type, clutch, game segment, and more
- Custom exceptions: `PlayerNotFoundError`, `TeamNotFoundError`, `SeasonNotFoundError`
- GitHub Actions CI with ruff lint and pytest across Python 3.10-3.13
- Full test suite with mocked nba_api calls
