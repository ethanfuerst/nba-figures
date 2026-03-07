# nbafigs

NBA shot chart generator with CLI and Python API.

## Overview

nbafigs makes it easy to generate NBA shot charts for any player or team. It wraps the `nba_api` package and renders publication-quality hexbin shot charts using matplotlib.

## Quick Example

```python
from nbafigs import player_shot_chart, team_shot_chart

# Generate a player shot chart
player_shot_chart('Stephen Curry', seasons=['2023-24'])

# Generate a team shot chart
team_shot_chart('Golden State Warriors', seasons=['2023-24'])
```

## Features

- Player and team shot charts with hexbin visualization
- CLI for quick chart generation
- Customizable chart parameters (scale, kind, limiters)
- Rate-limited NBA API calls with automatic retries
