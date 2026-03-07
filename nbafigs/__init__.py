from nbafigs.core.player import NBAPlayer as NBAPlayer
from nbafigs.core.team import NBATeam as NBATeam
from nbafigs.errors import PlayerNotFoundError as PlayerNotFoundError
from nbafigs.errors import SeasonNotFoundError as SeasonNotFoundError
from nbafigs.errors import TeamNotFoundError as TeamNotFoundError
from nbafigs.viz.chart import make_shot_chart as make_shot_chart
from nbafigs.viz.court import draw_court as draw_court

__version__ = '0.1.0'


def player_shot_chart(player_name, seasons=None, chart_params=None, **limiters):
    '''Generate a player shot chart.

    Args:
        player_name: Player name string.
        seasons: List of season start years (e.g. [2023] or [2020, 2023]).
        chart_params: Dict of parameters for make_shot_chart().
        **limiters: Filters for the shot chart API call.

    Returns:
        Tuple of (DataFrame, Figure).
    '''
    if chart_params is None:
        chart_params = {}
    player = NBAPlayer(player_name, print_name=False)
    return player.get_shot_chart(seasons=seasons, chart_params=chart_params, **limiters)


def team_shot_chart(team_abbrev, game_id, playoffs=False, chart_params=None):
    '''Generate a team shot chart.

    Args:
        team_abbrev: Team abbreviation (e.g. 'DAL').
        game_id: NBA game ID string.
        playoffs: Whether the game is a playoff game.
        chart_params: Dict of parameters for make_shot_chart().

    Returns:
        Tuple of (DataFrame, Figure).
    '''
    if chart_params is None:
        chart_params = {}
    team = NBATeam(team_abbrev)
    return team.get_shot_chart(game_id=game_id, playoffs=playoffs, chart_params=chart_params)
