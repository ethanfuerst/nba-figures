import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Rectangle, Arc
import datetime
import html5lib
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo, boxscoretraditionalv2, playergamelog, playercareerstats, teamgamelog, shotchartdetail, shotchartlineupdetail
from nba_season import NBA_Season
from nba_methods import make_shot_chart, shots_grouper

# - Custom errors
class PlayerNotFoundError(Exception):
    pass

class SeasonNotFoundError(Exception):
    pass

class TeamNotFoundError(Exception):
    pass


class NBA_Team():
    def __init__(self, team_abbrev):
        
        
        self.league = pd.DataFrame(teams.get_teams())
        if team_abbrev not in self.league['abbreviation'].to_list():
            raise TeamNotFoundError('Team not found. Check the abbreviation.')
        self.abbrev = team_abbrev
        self.id = self.league[self.league['abbreviation'] == self.abbrev]['id'].iloc[0]
        self.full_name = self.league[self.league['abbreviation'] == self.abbrev]['full_name'].iloc[0]
        self.nickname = self.league[self.league['abbreviation'] == self.abbrev]['nickname'].iloc[0]
        self.city = self.league[self.league['abbreviation'] == self.abbrev]['city'].iloc[0]
        self.state = self.league[self.league['abbreviation'] == self.abbrev]['state'].iloc[0]
        self.year_founded = self.league[self.league['abbreviation'] == self.abbrev]['year_founded'].iloc[0]

        return

    def get_season(self, season, season_type):
        '''
        
        '''
        # - Overview of games including ids
        # - team game log
        log = teamgamelog.TeamGameLog(team_id=self.id,season=season,season_type_all_star=season_type)
        return log.get_data_frames()[0]
    
    def get_season_games(self):
        return

    def get_shot_chart(self, game_id, playoffs=False, chart_params={}):
        '''
        game_id (string, required)
            The id of the game for the shotchart

        playoffs (boolean, default is False)
            Must be True if game referenced is in the playoffs
        
        chart_params (dict)
            See the make_shot_chart() method for list of paramters

        Returns:

        df
            df of data from API
        mavs.get_shot_chart('0041900156')
        plt
            plt object of the shotchart
        '''
        # - Query data
        log = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
        df = log.get_data_frames()[0]
        opp_team = df[df['TEAM_ABBREVIATION'] != self.abbrev]['TEAM_ID'].iloc[0]
        df = df[df['TEAM_ABBREVIATION'] == self.abbrev].copy()
        # - get list of all players that logged game minutes
        player_list = df[df['MIN'] != None]['PLAYER_ID'].to_list()
        if playoffs:
            season_type = 'Playoffs'
        else:
            season_type = 'Regular Season'
        # - get all shot charts from that game
        shots = pd.DataFrame()
        avgs = pd.DataFrame()
        for i in player_list:
            log = shotchartdetail.ShotChartDetail(team_id=self.id, game_id_nullable=game_id, 
                                                    player_id=i, context_measure_simple=['FGA', 'FG3A'],
                                                    season_type_all_star='Playoffs')
            df_1 = log.get_data_frames()[0]
            shots = pd.concat([shots, df_1])
            df_2 = log.get_data_frames()[1]
            avgs = pd.concat([avgs, df_2])
        
        shots = shots.reset_index()
        avgs = avgs.reset_index()

        # - fix data types
        shots[['SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG']] = shots[['SHOT_DISTANCE', 'LOC_X', 'LOC_Y', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG']].astype('int32')

        to_plot = shots_grouper(shots,avgs)

        opponent = self.league[self.league['id'] == opp_team]['full_name'].iloc[0]
        game_date = datetime.datetime.strptime(shots.iloc[0]['GAME_DATE'], '%Y%m%d').strftime("%B %-d, %Y")
        title = 'The ' + self.full_name + ' against the ' + opponent + ' on ' + game_date
        chart_params['title'] = title

        # - Make shot chart
        plt = make_shot_chart(to_plot, **chart_params)
        plt.show()
        return to_plot, plt
