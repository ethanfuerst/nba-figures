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
from nba_api.stats.endpoints import commonplayerinfo, playergamelog, playercareerstats, shotchartdetail, leaguegamelog, shotchartlineupdetail


# - Custom errors
class PlayerNotFoundError(Exception):
    pass

class SeasonNotFoundError(Exception):
    pass

class NBA_Season:
    def __init__(self, season=datetime.datetime.today().year - 1):
        '''
        Season object using data from basketball-reference.com

        Parameters:

        season (int, default: current year - 1)
            The season that you want to pull data from. 
                Ex. 2008
            If the season you inputted isn't an integer, a TypeError will be thrown.

        Attributes:
            self.season
            self.season_str
            self.games - df of all games in a season
            self.league - df of teams in the league
        '''

        try:
            season = int(season)
        except:
            # - This is probably because they inputted a string for season, and we need an int
            raise TypeError("Wrong variable type for season. Integer expected.")
        self.season = season
        self.season_str = str(season) + "-" + str(season + 1)[2:]

        # - basketball-reference references the season by the second year in each season, so we need to add 1 to the season
        season = self.season + 1
        # - The season goes from October to June usually, so we will go from July to June to capture all data
        # todo change when NBA season is changed to dec-aug
        # - see how months are formatted on bball ref during october and other months for 2020 season
        # - see how other seasons are formatted as well 
        months = [datetime.date(2019, i, 1).strftime('%B').lower() for i in list(range(10, 13)) + list(range(1,10))]

        # - Getting the list of URLs with our months list
        urls = []
        for i in months:
            urls.append('https://www.basketball-reference.com/leagues/NBA_' + str(season) + '_games-' + str(i) + '.html')
        
        games = pd.DataFrame()
        for url in urls:
            try:
                month = pd.read_html(url)[0]
                month.drop(['Notes','Unnamed: 6'], axis=1, inplace=True)
                month.dropna(subset=['PTS'], inplace=True)
                games = pd.concat([games, month], sort=False)
            except:
                pass
        
        # - Reset the index and rename the overtime column
        games.reset_index(inplace=True, drop=True)
        games.rename(columns={'Unnamed: 7': 'OT'}, inplace=True)

        self.games = self.__clean_games(games)
        self.league = pd.DataFrame(teams.get_teams())

        try:
            self.playoff_start = self.games[self.games['Date'] == 'playoffs'].index[0]
        except:
            # - The specified season doeesn't contain playoff games
            self.playoff_start = None
        
        log = leaguegamelog.LeagueGameLog(counter=0, direction='ASC', league_id='00', 
                player_or_team_abbreviation='T', season=self.season_str, season_type_all_star='Regular Season')

        self.game_log = log.get_data_frames()[0]

        log = leaguegamelog.LeagueGameLog(counter=0, direction='ASC', league_id='00', 
                player_or_team_abbreviation='T', season=self.season_str, season_type_all_star='Playoffs')
        
        self.playoffs = log.get_data_frames()[0]

        log = leaguegamelog.LeagueGameLog(counter=0, direction='ASC', league_id='00', 
                player_or_team_abbreviation='P', season=self.season_str, season_type_all_star='Regular Season')
        
        self.reg_player_scoring = log.get_data_frames()[0]

        log = leaguegamelog.LeagueGameLog(counter=0, direction='ASC', league_id='00', 
                player_or_team_abbreviation='P', season=self.season_str, season_type_all_star='Playoffs')
        
        self.po_player_scoring = log.get_data_frames()[0]

    def __str__(self):
        return self.season_str + ' NBA Season'
    
    def __repr__(self):
        return f"NBA_Season(season={self.season})"
    
    # - Private method (I think)
    def __clean_games(self, df):
        df['Season'] = self.season_str
        df['Year'] = pd.to_datetime(df['Date']).dt.year
        df['PTS'] = df['PTS'].astype(int)
        df['PTS.1'] = df['PTS.1'].astype(int)
        df['MOV'] = abs(df['PTS'] - df['PTS.1'])
        df['Date'] = pd.to_datetime(df['Date'])
        df['Start (ET)'] = pd.to_datetime(df['Start (ET)'])
        df['OT'] = df['OT'].fillna('No overtime')
        return df
    
    def get_season(self):
        if self.playoff_start == None:
            return self.__clean_games(self.games.copy())
        else:
            return self.__clean_games(self.games[self.games.index < self.playoff_start].copy())
    
    def get_play(self):
        if self.playoff_start == None:
            raise SeasonNotFoundError("There are no playoffs recorded for the " + self.season_str + " season.")
        else:
            return self.__clean_games(self.games[self.games.index > self.playoff_start].copy())
