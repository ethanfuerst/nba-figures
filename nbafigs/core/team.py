import datetime

import pandas as pd
from nba_api.stats.static import teams

from nbafigs.errors import TeamNotFoundError
from nbafigs.sources.player import fetch_shot_chart
from nbafigs.sources.team import fetch_box_score, fetch_team_game_log
from nbafigs.viz.chart import make_shot_chart, shots_grouper


class NBATeam:
    """Represents an NBA team and provides methods for shot chart generation.

    Args:
        team_abbrev: Team abbreviation (e.g. 'DAL', 'LAL').

    Raises:
        TeamNotFoundError: If no team matches the given abbreviation.
    """

    def __init__(self, team_abbrev):
        self.league = pd.DataFrame(teams.get_teams())

        if team_abbrev not in self.league['abbreviation'].to_list():
            raise TeamNotFoundError('Team not found. Check the abbreviation.')

        self.abbrev = team_abbrev
        team_row = self.league[self.league['abbreviation'] == team_abbrev].iloc[0]
        self.id = team_row['id']
        self.full_name = team_row['full_name']
        self.nickname = team_row['nickname']
        self.city = team_row['city']
        self.state = team_row['state']
        self.year_founded = team_row['year_founded']

    def get_season(self, season, season_type):
        """Fetch a team game log for a season.

        Args:
            season: Season start year.
            season_type: Season type string (e.g. 'Regular Season').

        Returns:
            DataFrame of game logs.
        """
        return fetch_team_game_log(self.id, season, season_type)

    def get_shot_chart(self, game_id, playoffs=False, chart_params=None):
        """Generate a team shot chart for a specific game.

        Args:
            game_id: NBA game ID string.
            playoffs: Whether the game is a playoff game.
            chart_params: Dict of parameters for make_shot_chart().

        Returns:
            Tuple of (DataFrame, Figure).
        """
        if chart_params is None:
            chart_params = {}

        df = fetch_box_score(game_id)
        opp_team = df[df['TEAM_ABBREVIATION'] != self.abbrev]['TEAM_ID'].iloc[0]
        df = df[df['TEAM_ABBREVIATION'] == self.abbrev].copy()
        player_list = df[df['MIN'] != None]['PLAYER_ID'].to_list()  # noqa: E711

        if playoffs:
            season_type = 'Playoffs'
        else:
            season_type = 'Regular Season'

        shots = pd.DataFrame()
        avgs = pd.DataFrame()
        for player_id in player_list:
            df_1, df_2 = fetch_shot_chart(
                player_id,
                team_id=self.id,
                game_id_nullable=game_id,
                season_type_all_star=season_type,
            )
            shots = pd.concat([shots, df_1])
            avgs = pd.concat([avgs, df_2])

        shots = shots.reset_index()
        avgs = avgs.reset_index()

        shots[
            ['SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG']
        ] = shots[
            ['SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG']
        ].astype('int32')

        to_plot = shots_grouper(shots, avgs)

        opponent = self.league[self.league['id'] == opp_team]['full_name'].iloc[0]
        game_date = datetime.datetime.strptime(
            shots.iloc[0]['GAME_DATE'], '%Y%m%d'
        ).strftime('%B %-d, %Y')
        title = f'The {self.full_name} against the {opponent} on {game_date}'
        chart_params['title'] = title

        fig = make_shot_chart(to_plot, **chart_params)
        return to_plot, fig
