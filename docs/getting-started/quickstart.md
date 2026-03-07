# Quickstart

## Python API

```python
from nbafigs import player_shot_chart, team_shot_chart

# Player shot chart
player_shot_chart('LeBron James', seasons=['2023-24'])

# Team shot chart
team_shot_chart('Los Angeles Lakers', seasons=['2023-24'])
```

## CLI

```bash
# Player shot chart
nbafigs player "Stephen Curry" --seasons 2023-24

# Team shot chart
nbafigs team "Golden State Warriors" --seasons 2023-24

# See all options
nbafigs --help
```
