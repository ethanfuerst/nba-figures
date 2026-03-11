import datetime

import numpy as np
import pandas as pd

from nbafigs.errors import PlayerNotFoundError, SeasonNotFoundError
from nbafigs.sources.player import (
    fetch_player_career_stats,
    fetch_player_game_log,
    fetch_shot_chart,
    find_player,
)
from nbafigs.sources.team import get_team_id
from nbafigs.viz.chart import make_shot_chart, shots_grouper


class NBAPlayer:
    """Represents an NBA player and provides methods for shot chart generation.

    Args:
        player_name: The name of the player. If the name is too general,
            the first match will be returned.
        print_name: If True, prints the player name on init.

    Raises:
        PlayerNotFoundError: If no player matches the given name.
    """

    def __init__(self, player_name, print_name=True):
        player_search = find_player(player_name)

        if player_search is None:
            raise PlayerNotFoundError(
                'Name not found in database. Try being more specific or look for the '
                'player here: https://stats.nba.com/players/'
            )

        self.player_id = player_search['id']
        self.name = player_search['full_name']
        self.first_name = player_search['first_name']
        self.last_name = player_search['last_name']
        self.is_active = player_search['is_active']
        self.print_name = print_name

        df = self.get_career()
        df = df[df['Team'] != 'TOT'][['Season', 'Team', 'TEAM_ID']].copy()
        df['start'] = df['Season'].apply(lambda x: int(x[:4]))
        df['end'] = df['start'] + 1
        self._career = df.rename({'TEAM_ID': 'Team ID', 'start': 'season'}, axis=1)[
            ['Team ID', 'season']
        ]
        cond = df.end.sub(df.end.shift()).ne(1) | (df.Team.ne(df.Team.shift()))
        no_year_end_change = df.end.shift(-1).sub(df.end).eq(0)
        df['change'] = df.loc[cond, 'start']
        df['end_edit'] = np.where(no_year_end_change, df.start, df.end)
        df['change'] = df.change.ffill().astype('Int64')
        df = df.groupby(['Team', 'TEAM_ID', 'change']).end_edit.max().reset_index()
        df['Years'] = df.change.astype(str).str.cat(df.end_edit.astype(str), sep='-')
        df = df.sort_values(['change', 'end_edit'])
        df = df.drop(['change', 'end_edit'], axis=1)
        df = df.rename({'TEAM_ID': 'Team ID'}, axis='columns')
        df = df.reset_index(drop=True)
        self.career = df

        if print_name:
            print(self.name)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f'NBA_Player(player_name={self.name}, print_name={self.print_name})'

    def get_season(self, season=None, season_type='regular'):
        """Fetch a single-season game log for the player.

        Args:
            season: Season start year (e.g. 2023). Defaults to current year - 1.
            season_type: One of 'regular', 'preseason', 'playoffs', 'allstar'.

        Returns:
            DataFrame of game logs.

        Raises:
            SeasonNotFoundError: If no data exists for the given season.
        """
        if season is None:
            season = datetime.datetime.today().year - 1

        if season_type not in ['regular', 'preseason', 'playoffs', 'allstar']:
            season_type = 'regular'

        s_types = {
            'regular': 'Regular Season',
            'preseason': 'Pre-Season',
            'playoffs': 'Playoffs',
            'allstar': 'All-Star',
        }
        s_type = s_types[season_type]

        df = fetch_player_game_log(self.player_id, season, s_type)

        if len(df) == 0:
            raise SeasonNotFoundError(
                f"{self.name} doesn't have data recorded for the {season} season."
            )

        df['Player'] = self.name
        df['Season'] = f'{season}-{str(season + 1)[2:]}'
        ts_denom = 2 * (df['FGA'] + (0.44 * df['FTA']))
        df['TS_PCT'] = round(df['PTS'] / ts_denom.replace(0, np.nan), 3)

        df = df[
            [
                'Season',
                'Player',
                'Game_ID',
                'GAME_DATE',
                'MATCHUP',
                'WL',
                'MIN',
                'FGM',
                'FGA',
                'FG_PCT',
                'FG3M',
                'FG3A',
                'FG3_PCT',
                'FTM',
                'FTA',
                'FT_PCT',
                'TS_PCT',
                'OREB',
                'DREB',
                'REB',
                'AST',
                'STL',
                'BLK',
                'TOV',
                'PF',
                'PTS',
                'PLUS_MINUS',
            ]
        ]

        return df

    def get_career(self):
        """Fetch career totals for the player.

        Returns:
            DataFrame of career stats by season.
        """
        df = fetch_player_career_stats(self.player_id)

        df['Player'] = self.name
        df['Season'] = df['SEASON_ID'].copy()
        df['Team'] = df['TEAM_ABBREVIATION'].copy()
        ts_denom = 2 * (df['FGA'] + (0.44 * df['FTA']))
        df['TS_PCT'] = round(df['PTS'] / ts_denom.replace(0, np.nan), 3)

        df = df[
            [
                'Player',
                'Season',
                'Team',
                'TEAM_ID',
                'PLAYER_AGE',
                'GP',
                'GS',
                'MIN',
                'FGM',
                'FGA',
                'FG_PCT',
                'FG3M',
                'FG3A',
                'FG3_PCT',
                'FTM',
                'FTA',
                'FT_PCT',
                'TS_PCT',
                'OREB',
                'DREB',
                'REB',
                'AST',
                'STL',
                'BLK',
                'TOV',
                'PF',
                'PTS',
            ]
        ].copy()

        return df

    def get_full_career(self, season_type='regular'):
        """Fetch game logs for all seasons in the player's career.

        Args:
            season_type: One of 'regular', 'preseason', 'playoffs', 'allstar',
                'full', 'all', 'no_all_star'.

        Returns:
            DataFrame of game logs across all requested seasons.
        """
        career_seasons = self.get_career()
        seasons = [int(i[:4]) for i in career_seasons['Season'].values.astype(str)]

        frames = []
        for i in seasons:
            if season_type in ('preseason', 'all', 'no_all_star'):
                try:
                    df_1 = self.get_season(i, season_type='preseason')
                    df_1['Season Type'] = 'PRE'
                    frames.append(df_1)
                except SeasonNotFoundError:
                    pass
            if season_type in ('regular', 'full', 'all', 'no_all_star'):
                try:
                    df_2 = self.get_season(i, season_type='regular')
                    df_2['Season Type'] = 'REG'
                    frames.append(df_2)
                except SeasonNotFoundError:
                    pass
            if season_type in ('playoffs', 'full', 'all', 'no_all_star'):
                try:
                    df_3 = self.get_season(i, season_type='playoffs')
                    df_3['Season Type'] = 'PLAY'
                    frames.append(df_3)
                except SeasonNotFoundError:
                    pass
            if season_type in ('allstar', 'all'):
                try:
                    df_4 = self.get_season(i, season_type='allstar')
                    df_4['Season Type'] = 'ALLSTAR'
                    frames.append(df_4)
                except SeasonNotFoundError:
                    pass

        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if df.empty:
            return df

        df = df[
            [
                'Season',
                'Season Type',
                'Player',
                'Game_ID',
                'GAME_DATE',
                'MATCHUP',
                'WL',
                'MIN',
                'FGM',
                'FGA',
                'FG_PCT',
                'FG3M',
                'FG3A',
                'FG3_PCT',
                'FTM',
                'FTA',
                'FT_PCT',
                'TS_PCT',
                'OREB',
                'DREB',
                'REB',
                'AST',
                'STL',
                'BLK',
                'TOV',
                'PF',
                'PTS',
                'PLUS_MINUS',
            ]
        ].copy()
        cols_as_int = [
            'MIN',
            'FGM',
            'FGA',
            'FG3M',
            'FG3A',
            'FTM',
            'FTA',
            'OREB',
            'DREB',
            'REB',
            'AST',
            'STL',
            'BLK',
            'TOV',
            'PF',
            'PTS',
            'PLUS_MINUS',
        ]
        df[cols_as_int] = df[cols_as_int].astype(int)

        return df

    def get_shot_chart(self, seasons=None, chart_params=None, **limiters):
        """Generate a shot chart for the player.

        Args:
            seasons: List of 1 or 2 season start years (e.g. [2023] or [2020, 2023]).
                None defaults to the most recent season.
            chart_params: Dict of parameters for make_shot_chart().
            **limiters: Filters for the shot chart (e.g. OpponentTeam='DAL').

        Returns:
            Tuple of (DataFrame, Figure).

        Raises:
            SeasonNotFoundError: If no shot data is found.
        """
        if chart_params is None:
            chart_params = {}
        to_plot = self.format_shots(seasons, chart_params, **limiters)
        fig = make_shot_chart(to_plot, **chart_params)
        return to_plot, fig

    def format_shots(self, seasons, chart_params, **limiters):
        """Pull and format shot data for chart generation.

        Args:
            seasons: List of season start years.
            chart_params: Dict of chart parameters (may be mutated to add title).
            **limiters: Filters for the shot chart API call.

        Returns:
            DataFrame ready for make_shot_chart().

        Raises:
            SeasonNotFoundError: If no shot data is found.
        """
        reassign_dict = dict(
            zip(
                [
                    'GameID',
                    'AheadBehind',
                    'ClutchTime',
                    'DateFrom',
                    'DateTo',
                    'GameSegment',
                    'LastNGames',
                    'Location',
                    'Month',
                    'OpponentTeam',
                    'Outcome',
                    'Period',
                    'PlayerPosition',
                    'PointDiff',
                    'RookieYear',
                    'SeasonSegment',
                    'SeasonType',
                    'VsConference',
                    'VsDivision',
                ],
                [
                    'game_id_nullable',
                    'ahead_behind_nullable',
                    'clutch_time_nullable',
                    'date_from_nullable',
                    'date_to_nullable',
                    'game_segment_nullable',
                    'last_n_games',
                    'location_nullable',
                    'month',
                    'opponent_team_id',
                    'outcome_nullable',
                    'period',
                    'player_position_nullable',
                    'point_diff_nullable',
                    'rookie_year_nullable',
                    'season_segment_nullable',
                    'season_type_all_star',
                    'vs_conference_nullable',
                    'vs_division_nullable',
                ],
                strict=False,
            )
        )

        new_limiters = {reassign_dict[key]: value for key, value in limiters.items()}

        if 'opponent_team_id' in new_limiters:
            new_limiters['opponent_team_id'] = get_team_id(
                new_limiters['opponent_team_id']
            )

        title = self.name
        if 'date_to_nullable' in new_limiters:
            d_from = datetime.datetime.strptime(
                new_limiters['date_from_nullable'], '%m-%d-%Y'
            ).strftime('%B %-d, %Y')
            d_to = datetime.datetime.strptime(
                new_limiters['date_to_nullable'], '%m-%d-%Y'
            ).strftime('%B %-d, %Y')
            title += f' from {d_from} to {d_to}'
        else:
            if seasons is None:
                l_seas = int(self.career['Years'].iloc[-1][5:]) - 1
                seasons = [l_seas]
            if len(seasons) == 1:
                title += f' in the {seasons[0]}-{str(seasons[0] + 1)[2:]} season'
            elif len(seasons) == 2 and seasons[1] - seasons[0] == 1:
                title += (
                    f' in the {seasons[0]}-{str(seasons[0] + 1)[2:]} and '
                    f'{seasons[1]}-{str(seasons[1] + 1)[2:]} seasons'
                )
            else:
                title += (
                    f' from the {seasons[0]}-{str(seasons[0] + 1)[2:]} to '
                    f'{seasons[-1]}-{str(seasons[-1] + 1)[2:]} seasons'
                )
        if 'title' not in chart_params:
            chart_params['title'] = title

        shots = pd.DataFrame()
        avgs = pd.DataFrame()

        if 'date_to_nullable' in new_limiters:
            df_1, df_2 = fetch_shot_chart(self.player_id, **new_limiters)
            shots = pd.concat([shots, df_1])
            avgs = pd.concat([avgs, df_2])
        else:
            first = seasons[0]
            last = seasons[-1]
            season_df = (
                self._career[
                    (self._career['season'].astype(int) >= first)
                    & (self._career['season'].astype(int) <= last)
                ]
                .reset_index(drop=True)
                .copy()
            )

            season_df['season'] = season_df['season'].apply(
                lambda x: f'{x}-{str(x + 1)[2:]}'
            )

            for i in range(len(season_df)):
                df_1, df_2 = fetch_shot_chart(
                    self.player_id,
                    season_nullable=season_df.iloc[i]['season'],
                    **new_limiters,
                )
                df_1['Season'] = season_df.iloc[i]['season']
                shots = pd.concat([shots, df_1])
                avgs = pd.concat([avgs, df_2])

        shots.reset_index(inplace=True, drop=True)

        if len(shots) == 0:
            if len(seasons) == 1:
                raise SeasonNotFoundError(
                    f'{self.name} has no data recorded for the {seasons[0]} season '
                    'with those limiters'
                )
            else:
                raise SeasonNotFoundError(
                    f'{self.name} has no data recorded for the '
                    f'{seasons[0]}-{seasons[-1]} seasons with those limiters'
                )

        return shots_grouper(shots, avgs)
