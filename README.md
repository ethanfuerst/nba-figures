# nbafigs

NBA shot chart generator with a CLI and Python API. Built on top of `nba_api`.

## Installation

pip:

```
pip install nbafigs
```

From source (development):

```
git clone https://github.com/ethanfuerst/nba-figures.git
cd nba-figures
uv sync --extra dev
```

## CLI usage

Player shot chart (normal):

```
nbafigs player "LeBron James" --seasons 2024 -o lebron.png
```

Player shot chart (hex, filtered):

```
nbafigs player "Stephen Curry" -s 2023 --kind hex --scale P_PPS --location Home -o curry_home.png
```

Team shot chart for a specific game:

```
nbafigs team DAL --game-id 0042300401 -o dal_game.png
```

Run `nbafigs --help`, `nbafigs player --help`, or `nbafigs team --help` for full option lists.

## Python API

```python
from nbafigs import player_shot_chart, team_shot_chart

# Player shot chart
df, fig = player_shot_chart('Cooper Flagg', seasons=[2025], chart_params={'kind': 'hex'})
fig.savefig('flagg.png')

# Team shot chart
df, fig = team_shot_chart('DAL', game_id='0042300401')
fig.savefig('dal.png')
```

## Available filters (player)

`--opponent`, `--location`, `--date-from`, `--date-to`, `--period`, `--season-type`,
`--clutch`, `--ahead-behind`, `--game-segment`, `--season-segment`, `--last-n-games`,
`--vs-conference`, `--vs-division`

All filters map directly to nba_api ShotChartDetail parameters.

## Development

```
git clone https://github.com/ethanfuerst/nba-figures.git
cd nba-figures
uv sync --extra dev
uv run pytest
```

## Contributing

1. Fork the repo and create a feature branch
2. Make changes and add tests
3. Run `uv run pytest` to verify
4. Open a PR against `main`

## License

MIT
