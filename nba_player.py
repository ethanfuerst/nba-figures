import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.patches as mpatches
import datetime
import html5lib
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo, playergamelog, playercareerstats, shotchartdetail, shotchartlineupdetail
from nba_season import NBA_Season
from nba_methods import *


class NBA_Player:
    def __init__(self, player_name, print_name=True):
        '''
        Player object from nba_api

        Parameters:

        player_name (string, required)
            The name of the player. If name is too general, then first name in the search will be returned.
                Ex. 'luka doncic' or 'harden'
            If the search does not return a player, a PlayerNotFoundError will be thrown.

        print_name (boolean, default: True)
            Will print the name of the player on default to make sure that you're querying the player that you want.
            If you don't want this done, you can pass it False
        
        Attributes:
            self.player_id
            self.name
            self.first_name
            self.last_name
            self.is_active
            self.career
            self.league - df of teams in the league
        '''

        # - Search for the player to get the id
        player_search = players.find_players_by_full_name(player_name)

        # - If no results for the player, throw an error
        if len(player_search) == 0:
            raise PlayerNotFoundError('Name not found in database. Try being more specific or look for the player here: https://stats.nba.com/players/')

        # - Get the id from the result of my search
        self.player_id = player_search[0]['id']
        self.name = player_search[0]['full_name']
        self.first_name = player_search[0]['first_name']
        self.last_name = player_search[0]['last_name']
        self.is_active = player_search[0]['is_active']
        self.print_name = print_name
        # - Create a df that outlines the players career
        df = self.get_career()
        df = df[df['Team'] != 'TOT'][['Season', 'Team', 'TEAM_ID']].copy()
        df['start'] = df['Season'].apply(lambda x: int(x[:4]))
        df['end'] = df['start'] + 1
        self._career = df.rename({'TEAM_ID': 'Team ID', 'start': 'season'}, axis=1)[['Team ID', 'season']]
        cond = df.end.sub(df.end.shift()).ne(1) | (df.Team.ne(df.Team.shift()))
        no_year_end_change = df.end.shift(-1).sub(df.end).eq(0)
        df['change'] = df.loc[cond,'start']
        df['end_edit'] = np.where(no_year_end_change,df.start,df.end)
        df['change'] = df.change.ffill().astype('Int64')
        df = df.groupby(['Team','TEAM_ID','change']).end_edit.max().reset_index()
        df['Years'] = df.change.astype(str).str.cat(df.end_edit.astype(str),sep='-')
        df = df.sort_values(['change', 'end_edit'])
        df = df.drop(['change','end_edit'],axis = 1)
        df = df.rename({'TEAM_ID': 'Team ID'}, axis='columns')
        df = df.reset_index(drop=True)
        self.career = df

        if print_name:
            print(self.name)
        
        return
    
    def __str__(self):
        return self.name
    
    def __repr__(self):
        return f"NBA_Player(player_name={self.name}, print_name={self.print_name})"
    
    def get_season(self, season=datetime.datetime.today().year - 1, season_type='regular'):
        '''
        Returns a df of player game logs from the 'season to season+1' season.
        Specify 'regular', 'preseason', 'playoffs' or 'allstar'.

        Parameters:

        season (int, default: current year - 1)
            The season that you want to pull data from. 
                Ex. 2003
            If the player you specified doesn't have date from the season inputted, a SeasonNotFoundError will be thrown.
            
        season_type (string, default: 'regular')
            The period of games from which you'd like the data from.
            Must be one of the following:
                'regular' - Regular Season
                'preseason' - Pre-Season
                'playoffs' - Playoffs
                'allstar' - All-Star
            If season_type is not one of the values above, it will be changed to 'regular'.

        Returns:

        df
            A pd.DataFrame() containing the player data with the following columns:
                ['Season', 'Player', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL',
                'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
                'FT_PCT', 'TS_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                'PTS', 'PLUS_MINUS']
        '''
        # - Change season_type to 'regular' if not specified correctly
        if season_type not in ['regular', 'preseason', 'playoffs', 'allstar']:
            season_type = 'regular'
        
        s_types = {'regular':'Regular Season', 'preseason':'Pre-Season', 'playoffs':'Playoffs', 'allstar':'All-Star'}
        s_type = s_types[season_type]

        # - playergamelog is a nba_api endpoint that contains the dataframes
        try:
            log = playergamelog.PlayerGameLog(player_id=self.player_id, season=season, season_type_all_star=s_type)
        except:
            return pd.DataFrame()

        # - If no results for the season, throw an error
        if len(log.get_data_frames()[0]) == 0:
            raise SeasonNotFoundError(self.name + " doesn't have data recorded for the " +  str(season) + " season." )

        df = log.get_data_frames()[0]
        df['Player'] = self.name
        df['Season'] = str(season) + "-" + str(season + 1)[2:]
        df['TS_PCT'] = round(df['PTS'] / (2*(df['FGA'] + (.44 * df['FTA']))),3)

        # - Drop the 'video available' column and reorder
        df = df[['Season', 'Player', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL',
                'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
                'FT_PCT', 'TS_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'PF',
                'PTS', 'PLUS_MINUS']]
        
        return df

    def get_career(self):
        '''
        Returns a df of the player's totals and percentages for all season in the player's career.

        Parameters:

        season (int, default: current year - 1)
            The season that you want to pull data from. 
                Ex. 2003
            If the player you specified doesn't have date from the season inputted, a SeasonNotFoundError will be thrown.
        
        Returns:

        df
            A pd.DataFrame() containing the player data with the following columns:
                ['Player', 'Season', 'Team', 'TEAM_ID', 
                'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TS_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                'STL', 'BLK', 'TOV', 'PF', 'PTS']
        '''
        # - see more on https://github.com/swar/nba_api/blob/master/docs/nba_api/stats/endpoints/playercareerstats.md
        log = playercareerstats.PlayerCareerStats(player_id=self.player_id, per_mode36='Totals')
        df = log.get_data_frames()[0]

        df['Player'] = self.name
        df['Season'] = df['SEASON_ID'].copy()
        df['Team'] = df['TEAM_ABBREVIATION'].copy()
        df['TS_PCT'] = round(df['PTS'] / (2*(df['FGA'] + (.44 * df['FTA']))),3)

        # - Specify column order
        df = df[['Player', 'Season', 'Team', 'TEAM_ID', 
                'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TS_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                'STL', 'BLK', 'TOV', 'PF', 'PTS']].copy()
                
        return df

    def get_full_career(self, season_type='regular'):
        '''
        Returns a df of the player's game logs for all season in the player's career.

        Parameters:

        season_type (string, default: 'regular')
            The period of games from which you'd like the data from.
            Must be one of the following:
                'regular' - Regular Season
                'preseason' - Pre-Season
                'playoffs' - Playoffs
                'allstar' - All-Star
                'full' - Regular Season and Playoffs
                'all' - Pre-Season, Regular Season, Playoffs and All-Star
                'no_all_star' - Pre-Season, Regular Season and Playoffs
            If season_type is not one of the values above, it will be changed to 'regular'.
        
        Returns:

        df
            A pd.DataFrame() containing the player data with the following columns:
                ['Player', 'Season', 'Team', 'TEAM_ID', 
                'PLAYER_AGE', 'GP', 'GS', 'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A',
                'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'TS_PCT', 'OREB', 'DREB', 'REB', 'AST', 
                'STL', 'BLK', 'TOV', 'PF', 'PTS']
        '''
        # - Get list of seasons with data
        career_seasons = self.get_career()
        seasons = [int(i[:4]) for i in career_seasons['Season'].values.astype(str)]

        # - Get data from each season
        df = pd.DataFrame()
        for i in seasons:
            if season_type == 'preseason' or season_type == 'all' or season_type == 'no_all_star':
                try:
                    df_1 = self.get_season(i, season_type='preseason')
                    df_1['Season Type'] = 'PRE'
                    df = df.append(df_1)
                except SeasonNotFoundError:
                    pass
            if season_type == 'regular' or season_type == 'full' or season_type == 'all' or season_type == 'no_all_star':
                try:
                    df_2 = self.get_season(i, season_type='regular')
                    df_2['Season Type'] = 'REG'
                    df = df.append(df_2)
                except SeasonNotFoundError:
                    pass
            if season_type == 'playoffs' or season_type == 'full' or season_type == 'all' or season_type == 'no_all_star':
                # Add flag if player had no data in playoffs
                try:
                    df_3 = self.get_season(i, season_type='playoffs')
                    df_3['Season Type'] = 'PLAY'
                    df = df.append(df_3)
                except SeasonNotFoundError:
                    pass
            if season_type == 'allstar' or season_type == 'all':
                try:
                    df_4 = self.get_season(i, season_type='allstar')
                    df_4['Season Type'] = 'ALLSTAR'
                    df = df.append(df_4)
                except SeasonNotFoundError:
                    pass

        df.reset_index(inplace=True)
        df = df[['Season', 'Season Type', 'Player', 'Game_ID', 'GAME_DATE', 'MATCHUP', 'WL', 
                'MIN', 'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA',
                'FT_PCT', 'TS_PCT', 'OREB', 'DREB', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                'PF', 'PTS', 'PLUS_MINUS']].copy()
        cols_as_int = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 
                        'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', 'PLUS_MINUS']
        # ! error with all star data here
        df[cols_as_int] = df[cols_as_int].astype(int)

        return df
    
    def get_shot_chart(self, seasons=None, chart_params={}, **limiters):
        '''
        Returns a matplotlib fig and a pandas DataFrame showing the player's shot chart given certain parameters.
        Can filter that df and pass to make_shot_chart to limit the shots plotted by shot type (ex. Jump shot, Dunk, etc.)

        Parameters:

        seasons (list of integers, default: None (most recent season of career))
            The seasons (inclusive) for which you'd like to get data from.
            Must be a list of length 1 or 2 containing integers of the seasons.
            Example: [2005, 2018]
                Will return data for the 05-06 season to the 18-19 season
            If seasons, DateFrom and DateTo are passed make sure that the seasons cover the dates.
            If nothing is passed, it will return just the most recent season of the players career.
            If the player doesn't have data recorded for a year passed, then a SeasonNotFoundError will be thrown.
            If the seasons range contains years that the player didn't play then only years with data will be shown.

        chart_params (dict)
            See the make_shot_chart() method for list of paramters

        **limiters (assorted data types)
            These will filter the shots on the shot chart.
            AheadBehind - One of 'Ahead or Behind', 'Ahead or Tied', 'Behind or Tied'
            ClutchTime - One of 'Last (1-5) Minute(s)' or 'Last (30 or 10) Seconds'
            DateFrom - ex. 12-14-2019
            DateTo - ex. 01-24-2020
                Must specify both date_from_nullable and date_to_nullable
            GameSegment - One of 'First Half', 'Overtime', 'Second Half'
            LastNGames - integer
                Will return the last n games from the season specified
                Default is most recent season played
            Location - One of 'Home', 'Road'
            OpponentTeam - Abbreviation, like DAL or LAL
            Outcome - One of 'W' or 'L'
            Period - 1, 2, 3, 4 or 5 (OT)
            PlayerPosition - One of 'Guard', 'Center', 'Forward'
            PointDiff - integer
            SeasonSegment - One of 'Post All-Star', 'Pre All-Star'
            SeasonType - One of 'Regular Season', 'Pre Season', 'Playoffs', 'All Star'
            GameID - Use self.career and the NBA_Team class to get the full schedule with game_ids
            VsConference - One of 'East', 'West'
            VsDivision  - One of 'Atlantic', 'Central', 'Northwest', 'Pacific', 'Southeast', 'Southwest', 'East', 'West'
        
        
        Returns:

        df
            df of data from API
        
        fig
            fig object of the shotchart
        '''
        # todo add 'all' parameter to seasons that returns whole career
        to_plot = self.format_shots(seasons, chart_params, **limiters)

        fig = make_shot_chart(to_plot, **chart_params)
        return to_plot, fig

    def format_shots(self, seasons, chart_params, **limiters):
        '''
        Pulls and formats data for get_shot_chart and get_shot_dist (with those parameters)
        '''
        reassign_dict = dict(zip(['GameID', 'AheadBehind', 'ClutchTime', 'DateFrom', 'DateTo', 'GameSegment', 'LastNGames', 'Location', 
                        'Month', 'OpponentTeam', 'Outcome', 'Period', 'PlayerPosition', 'PointDiff', 'RookieYear', 
                        'SeasonSegment', 'SeasonType', 'VsConference', 'VsDivision'], 
                        ['game_id_nullable','ahead_behind_nullable', 'clutch_time_nullable', 'date_from_nullable', 
                        'date_to_nullable', 'game_segment_nullable', 'last_n_games', 'location_nullable', 
                        'month', 'opponent_team_id', 'outcome_nullable', 'period', 'player_position_nullable', 
                        'point_diff_nullable', 'rookie_year_nullable', 'season_segment_nullable', 
                        'season_type_all_star', 'vs_conference_nullable', 'vs_division_nullable']))
        
        new_limiters = {reassign_dict[key]: value for key, value in limiters.items()}

        if 'opponent_team_id' in new_limiters.keys():
            new_limiters['opponent_team_id'] = get_team_id(new_limiters['opponent_team_id'])
        
        # - Create title
        title = self.name
        if 'date_to_nullable' in new_limiters.keys():
            d_from = datetime.datetime.strptime(new_limiters['date_from_nullable'], '%m-%d-%Y').strftime("%B %-d, %Y")
            d_to = datetime.datetime.strptime(new_limiters['date_to_nullable'], '%m-%d-%Y').strftime("%B %-d, %Y")
            title += ' from ' + d_from + ' to ' + d_to
        else:
            if seasons is None:
                # - Will get most recent season
                l_seas = int(self.career['Years'].iloc[-1][5:]) - 1
                seasons = [l_seas]
            if len(seasons) == 1:
                title += ' in the ' + str(str(seasons[0]) + "-" + str(seasons[0] + 1)[2:]) + ' season'
            elif seasons[1] - seasons[0] == 1:
                title += ' in the ' + str(str(seasons[0]) + "-" + str(seasons[0] + 1)[2:]) + ' and ' +str(str(seasons[1]) + "-" + str(seasons[1] + 1)[2:]) + ' seasons'
            else:
                title += ' from the ' + str(str(seasons[0]) + "-" + str(seasons[0] + 1)[2:]) + ' to ' +str(str(seasons[1]) + "-" + str(seasons[1] + 1)[2:]) + ' seasons'
        if 'title' not in chart_params.keys():
            chart_params['title'] = title

        shots = pd.DataFrame()
        avgs = pd.DataFrame()

        # - if dates are not null
        if 'date_to_nullable' in new_limiters.keys():
            # - Query with dates
            log = shotchartdetail.ShotChartDetail(team_id=0, player_id=self.player_id, 
                                                    context_measure_simple=['FGA', 'FG3A'], **new_limiters)
            df_1 = log.get_data_frames()[0]
            df_2 = log.get_data_frames()[1]
            # df_1['Season'] = season_df.iloc[i]['season']
            shots = pd.concat([shots, df_1])
            avgs = pd.concat([avgs, df_2])
        # - else when seasons not null
        else:
            # - Query with seasons
            if len(seasons) > 2:
                raise TypeError('The seasons variable must be a list of length 2 or 1 with years in integer form. Example: [2005, 2018]')
            else:
                # - Get the seasons from ref between two dates
                first = seasons[0]
                if len(seasons) == 1:
                    last = seasons[0]
                else:
                    last = seasons[1]
                # - Get all seasons and team ID between first and last
                season_df = self._career[(self._career['season'].astype(int) >= first) & (self._career['season'].astype(int) <= last)].reset_index(drop=True).copy()

            # - Change format of season column to work with the API
            season_df['season'] = season_df['season'].apply(lambda x: str(x) + "-" + str(x + 1)[2:])
            # - Now create the df for the shot chart creation with the dfs given
            for i in range(len(season_df)):
                log = shotchartdetail.ShotChartDetail(team_id=0, player_id=self.player_id, \
                    season_nullable=season_df.iloc[i]['season'], context_measure_simple=['FGA', 'FG3A'], **new_limiters)
                df_1 = log.get_data_frames()[0]
                df_2 = log.get_data_frames()[1]
                df_1['Season'] = season_df.iloc[i]['season']
                shots = pd.concat([shots, df_1])
                avgs = pd.concat([avgs, df_2])
        
        shots.reset_index(inplace=True, drop=True)

        if len(shots) == 0:
            if len(seasons) == 1:
                raise SeasonNotFoundError(str(self.name) + ' has no data recorded for the ' + str(seasons[0]) + ' season with those limiters')
            else:
                raise SeasonNotFoundError(str(self.name) + ' has no data recorded for the ' + str(seasons[0]) + '-' + str(seasons[1]) + ' seasons with those limiters')
        
        return shots_grouper(shots,avgs)

    def get_shot_dist(self, seasons=None, chart_params={}, **limiters):
        to_plot = self.format_shots(seasons, chart_params, **limiters)

        fig = make_shot_dist(to_plot, **chart_params)
        return to_plot, fig
    
    def get_ani_shot_chart(self, seasons=None, 
                            interval=750, repeat_delay=0,
                            chart_params={}, **limiters):
        '''
        Returns a matplotlib ani showing the player's shot chart given certain parameters.
        Saved as player_firstname_player_lastname_season-season.gif

        Parameters:

        seasons (list of integers, default: None (most recent season of career))
            The seasons (inclusive) for which you'd like to get data from.
            Must be a list of length 1 or 2 containing integers of the seasons.
            Example: [2005, 2018]
                Will return data for the 05-06 season to the 18-19 season
            If seasons, DateFrom and DateTo are passed make sure that the seasons cover the dates.
            If nothing is passed, it will return just the most recent season of the players career.
            If the player doesn't have data recorded for a year passed, then a SeasonNotFoundError will be thrown.
            If the seasons range contains years that the player didn't play then only years with data will be shown.

        interval (int, default 750):
            Miliseconds between frame switch
        
        repeat_delay (int, default 0):
            Miliseconds between gif repeat

        chart_params (dict)
            See the make_shot_chart() method for list of paramters

        **limiters (assorted data types)
            These will filter the shots on the shot chart.
            AheadBehind - One of 'Ahead or Behind', 'Ahead or Tied', 'Behind or Tied'
            ClutchTime - One of 'Last (1-5) Minute(s)' or 'Last (30 or 10) Seconds'
            DateFrom - ex. 12-14-2019
            DateTo - ex. 01-24-2020
                Must specify both date_from_nullable and date_to_nullable
            GameSegment - One of 'First Half', 'Overtime', 'Second Half'
            LastNGames - integer
                Will return the last n games from the season specified
                Default is most recent season played
            Location - One of 'Home', 'Road'
            OpponentTeam - Abbreviation, like DAL or LAL
            Outcome - One of 'W' or 'L'
            Period - 1, 2, 3, 4 or 5 (OT)
            PlayerPosition - One of 'Guard', 'Center', 'Forward'
            PointDiff - integer
            SeasonSegment - One of 'Post All-Star', 'Pre All-Star'
            SeasonType - One of 'Regular Season', 'Pre Season', 'Playoffs', 'All Star'
            GameID - Use self.career and the NBA_Team class to get the full schedule with game_ids
            VsConference - One of 'East', 'West'
            VsDivision  - One of 'Atlantic', 'Central', 'Northwest', 'Pacific', 'Southeast', 'Southwest', 'East', 'West'
        
        
        Returns:

        ani
            ani figure from matplotlib
        '''

        fig_defaults = dict(title=None, title_size=22, context=None, context_size=14,
                            scale='P_PPS', show_misses=True, hex_grid=50, scale_factor=5,
                            min_factor=0)

        for i in fig_defaults.keys():
            if i not in chart_params.keys():
                chart_params.update({i: fig_defaults[i]})

        # make fig and ax
        background_color = '#d9d9d9'
        fig, ax = plt.subplots(facecolor=background_color, figsize=(10,10))
        
        if seasons == None:
            seasons = [self._career['season'].min(), self._career['season'].max()]

        if len(seasons) == 2:
            seasons = [i for i in range(seasons[0], seasons[1] + 1)]

        # make list of params in dict
        frames = []
        for i in seasons:
            # add the data with limiters
            
            df = self.format_shots(seasons=[i], chart_params=chart_params, **limiters)

            # and the chart params because this whole item in the iterable will be passed to the function
            season_data = dict(
                data=df,
                params=chart_params,
                season=i
            )

            frames.append(season_data)

        def hex_animate(i):
            if i != seasons[0]:
                ax.clear()
            frame = frames[i]
            df = frame['data']
            df_t = df.copy()

            scale = frame['params']['scale']
            show_misses = frame['params']['show_misses']
            hex_grid = frame['params']['hex_grid']
            scale_factor = frame['params']['scale_factor']
            min_factor = frame['params']['min_factor']

            if scale == 'P_PPS':
                # - error if highest val is 1
                df_t['P_PPS'] = df_t['P_PPS']/3

            if not show_misses:
                df_t = df_t[df_t['SHOT_MADE'] == 1].copy()
            hexbin = ax.hexbin(df_t['X'], df_t['Y'], C=df_t[scale].values
                , gridsize=hex_grid, edgecolors='black',cmap=cm.get_cmap('RdYlBu_r'), extent=[-275, 275, -50, 425]
                , reduce_C_function=np.sum)
            # - color
            hexbin2 = ax.hexbin(df_t['X'], df_t['Y'], C=df_t[scale].values, gridsize=hex_grid, edgecolors='black',
                cmap=cm.get_cmap('RdYlBu_r'), extent=[-275, 275, -50, 425], reduce_C_function=np.mean)

            if chart_params['title'] is not None:
                ax.set_title(chart_params['title'], pad=10, fontdict={'fontsize': chart_params['title_size'], 'fontweight':'semibold'})

            court_elements = draw_court()
            for element in court_elements:
                ax.add_patch(element)

            img = plt.imread("basketball-floor-texture.png")
            ax.imshow(img,zorder=0, extent=[-275, 275, -50, 425])

            ax.set_xlim(-250,250)
            ax.set_ylim(422.5, -47.5)
            ax.axis(False)

            if chart_params['context'] is not None:
                # - If multiple lines then add context size to second variable for each additional line
                ax.text(0, 435 + (chart_params['context_size'] * chart_params['context'].count('\n')), s=chart_params['context'], 
                                fontsize=chart_params['context_size'], ha='center')

            # - gets the color for the legend on the bottom left using the first season of data
            offsets = hexbin.get_offsets()
            orgpath = hexbin.get_paths()[0]
            verts = orgpath.vertices
            values1 = hexbin.get_array()
            values1 = np.array([scale_factor if i > scale_factor else 0 if i < min_factor else i for i in values1])
            values1 = ((values1 - 1.0)/(scale_factor-1.0))*(1.0-.4) + .4
            values2 = hexbin2.get_array()
            patches = []

            for offset, val in zip(offsets,values1):
                v1 =  verts*val + offset
                path = Path(v1, orgpath.codes)
                patch = PathPatch(path)
                patches.append(patch)

            pc = PatchCollection(patches, cmap=cm.get_cmap('RdYlBu_r'), edgecolors='black')
            if scale == 'PCT_DIFF':
                if pc.get_clim()[0] is None:
                    bottom = abs(df_t[scale].min())
                    top = abs(df_t[scale].max())
                else:
                    top = abs(pc.get_clim()[1])
                    bottom = abs(pc.get_clim()[0])
                m = min(top, bottom)
                # - Need one extreme of the comparison to be at least 1.5 percent off from average
                if m < .025:
                    m = .025
                pc.set_clim([-1 * m, m])
            # - pps is .4 to 1.something 
            elif scale in ['P_PPS', 'L_PPS']:
                # - for 2: 20% to 60%
                # - for 3: 13% to 40%
                pc.set_clim([0.13333, .4])
            else:
                pc.set_clim([-.05,.05])
            
            pc.set_array(values2)

            ax.add_collection(pc)
            hexbin.remove()
            hexbin2.remove()
            
            ax.text(200, 375, str(frame['season']) + "-" + str(frame['season'] + 1)[2:] + ' season',
                        horizontalalignment='center', fontsize=12, bbox=dict(facecolor=background_color, boxstyle='round'))

            return hexbin,

        fig.patch.set_facecolor(background_color)
        ax.patch.set_facecolor(background_color)
        plt.rcParams["figure.figsize"] = (10, 10)

        hx = ax.hexbin(frames[0]['data']['X'], frames[0]['data']['Y'], C=frames[0]['data'][frames[0]['params']['scale']].values
                , gridsize=frames[0]['params']['hex_grid'], edgecolors='black',cmap=cm.get_cmap('RdYlBu_r')
                , reduce_C_function=np.sum)
        axins1 = inset_axes(ax, width="16%", height="2%", loc='lower left')
        cbar = fig.colorbar(hx, cax=axins1, orientation="horizontal", ticks=[-1, 1])
        interval = hx.get_clim()[1] - hx.get_clim()[0]
        ltick = hx.get_clim()[0] + (interval * .2)
        rtick = hx.get_clim()[1] - (interval * .2)
        cbar.set_ticks([ltick, rtick])
        axins1.xaxis.set_ticks_position('top')
        if frames[0]['params']['scale'] == 'PCT_DIFF':
            legend_text = '% Compared to \nLeague Average'
            tick_labels = ['Below', 'Above']
        elif frames[0]['params']['scale'] == 'P_PPS':
            legend_text = 'Efficiency by Zone'
            tick_labels = ['Lower', 'Higher']
        else:
            legend_text = 'Efficiency compared to \nLeague Average'
            tick_labels = ['Lower', 'Higher']
        cbar.ax.set_title(legend_text, fontsize=10)
        cbar.set_ticklabels(tick_labels)

        # show pct
        plt.text(222, 20, 'The larger hexagons\nrepresent a higher\ndensity of shots',
                        horizontalalignment='center', bbox=dict(facecolor=background_color, boxstyle='round'))

        ani = animation.FuncAnimation(fig, func=hex_animate, frames=len(seasons), blit=True,
                                        interval=interval, repeat_delay=repeat_delay, save_count=1)

        ani.save(self.first_name + '_' + self.last_name + str(seasons[0])[2:] + '-' + str(seasons[-1])[2:] + '.gif', 
                fps=1, writer='PillowWriter', 
                savefig_kwargs={'facecolor':background_color, 'bbox_inches' : 'tight', 'pad_inches': .05})
        
        return ani
