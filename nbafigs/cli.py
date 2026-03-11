import matplotlib

matplotlib.use('Agg')

import click
import matplotlib.pyplot as plt

from nbafigs import player_shot_chart, team_shot_chart


@click.group()
@click.version_option()
def main():
    """NBA shot chart generator."""
    pass


@main.command()
@click.argument('player_name')
@click.option(
    '--seasons',
    '-s',
    type=int,
    multiple=True,
    help='Season year(s), e.g. 2023 or 2020 2023',
)
@click.option('--kind', type=click.Choice(['normal', 'hex']), default='normal')
@click.option(
    '--scale', type=click.Choice(['P_PPS', 'PCT_DIFF', 'D_PPS']), default='P_PPS'
)
@click.option('--no-misses', is_flag=True, help='Hide missed shots')
@click.option('--no-pct', is_flag=True, help='Hide shooting percentages')
@click.option('--title', type=str, default=None)
@click.option('--context', type=str, default=None)
@click.option(
    '--output',
    '-o',
    type=click.Path(),
    default=None,
    help='Save to file instead of displaying',
)
@click.option(
    '--opponent', type=str, default=None, help='Opponent team abbreviation (e.g. DAL)'
)
@click.option('--location', type=click.Choice(['Home', 'Road']), default=None)
@click.option('--date-from', type=str, default=None, help='Start date (MM-DD-YYYY)')
@click.option('--date-to', type=str, default=None, help='End date (MM-DD-YYYY)')
@click.option('--period', type=int, default=None, help='Game period (1-4, 5=OT)')
@click.option('--season-type', type=str, default=None)
@click.option('--clutch', type=str, default=None)
@click.option('--ahead-behind', type=str, default=None)
@click.option(
    '--game-segment',
    type=click.Choice(['First Half', 'Overtime', 'Second Half']),
    default=None,
)
@click.option(
    '--season-segment',
    type=click.Choice(['Post All-Star', 'Pre All-Star']),
    default=None,
)
@click.option('--last-n-games', type=int, default=None)
@click.option('--vs-conference', type=click.Choice(['East', 'West']), default=None)
@click.option('--vs-division', type=str, default=None)
def player(
    player_name,
    seasons,
    kind,
    scale,
    no_misses,
    no_pct,
    title,
    context,
    output,
    **filters,
):
    """Generate a player shot chart."""
    chart_params = {
        'kind': kind,
        'scale': scale,
        'show_misses': not no_misses,
        'show_pct': not no_pct,
    }
    if title:
        chart_params['title'] = title
    if context:
        chart_params['context'] = context

    # Map CLI filter names to nba_api limiter keys.
    # To add a new filter: add a @click.option above and a key here.
    filter_map = {
        'opponent': 'OpponentTeam',
        'location': 'Location',
        'date_from': 'DateFrom',
        'date_to': 'DateTo',
        'period': 'Period',
        'season_type': 'SeasonType',
        'clutch': 'ClutchTime',
        'ahead_behind': 'AheadBehind',
        'game_segment': 'GameSegment',
        'season_segment': 'SeasonSegment',
        'last_n_games': 'LastNGames',
        'vs_conference': 'VsConference',
        'vs_division': 'VsDivision',
    }
    limiters = {}
    for cli_key, api_key in filter_map.items():
        val = filters.get(cli_key)
        if val is not None:
            limiters[api_key] = val

    seasons_list = list(seasons) if seasons else None
    df, fig = player_shot_chart(
        player_name, seasons=seasons_list, chart_params=chart_params, **limiters
    )

    if output:
        fig.savefig(output, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
        click.echo(f'Saved to {output}')
    else:
        plt.show()


@main.command()
@click.argument('team_abbrev')
@click.option('--game-id', required=True, help='NBA game ID')
@click.option('--playoffs', is_flag=True)
@click.option('--kind', type=click.Choice(['normal', 'hex']), default='normal')
@click.option(
    '--scale', type=click.Choice(['P_PPS', 'PCT_DIFF', 'D_PPS']), default='P_PPS'
)
@click.option('--no-misses', is_flag=True)
@click.option('--no-pct', is_flag=True)
@click.option('--output', '-o', type=click.Path(), default=None)
def team(team_abbrev, game_id, playoffs, kind, scale, no_misses, no_pct, output):
    """Generate a team shot chart for a specific game."""
    chart_params = {
        'kind': kind,
        'scale': scale,
        'show_misses': not no_misses,
        'show_pct': not no_pct,
    }

    df, fig = team_shot_chart(
        team_abbrev, game_id, playoffs=playoffs, chart_params=chart_params
    )

    if output:
        fig.savefig(output, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=150)
        click.echo(f'Saved to {output}')
    else:
        plt.show()
