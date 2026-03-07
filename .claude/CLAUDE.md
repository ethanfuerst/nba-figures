# nbafigs

## Project structure

```
nbafigs/           # Installable package
├── core/          # NBAPlayer, NBATeam classes
├── sources/       # nba_api wrappers (all calls go through nba_api_call())
├── viz/           # Rendering: court.py (draw_court, make_shot_fig), chart.py (make_shot_chart, shots_grouper)
├── data/          # Bundled assets (basketball-floor-texture.png)
├── cli.py         # Click-based CLI
├── errors.py      # Custom exceptions
└── __init__.py    # Public API: player_shot_chart(), team_shot_chart()
tests/             # pytest tests (all nba_api calls mocked)
```

## Commands

```bash
uv sync --extra dev    # Install dependencies
uv run pytest          # Run tests
uv run nbafigs --help  # CLI usage
```

## Conventions

- All nba_api calls go through `sources/nba_api_call()` for retry + rate limiting
- Texture loaded via `importlib.resources` (not relative paths)
- Use `matplotlib.colormaps['name']` (not deprecated `cm.get_cmap()`)
- Use `pd.concat()` (not deprecated `df.append()`)
- Mock all nba_api calls in tests — never hit the real API
