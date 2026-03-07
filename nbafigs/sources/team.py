import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv2, teamgamelog
from nba_api.stats.static import teams

from nbafigs.sources import nba_api_call


def find_team(team_abbrev):
    '''Look up a team by abbreviation. Return the team dict or None.'''
    all_teams = pd.DataFrame(teams.get_teams())
    matches = all_teams[all_teams['abbreviation'] == team_abbrev]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def get_team_id(team_abbrev):
    '''Return the team_id for a given abbreviation.'''
    team = find_team(team_abbrev)
    return team['id'] if team else None


def fetch_team_game_log(team_id, season, season_type_all_star):
    '''Fetch a team game log for a season.'''
    log = nba_api_call(
        teamgamelog.TeamGameLog,
        team_id=team_id,
        season=season,
        season_type_all_star=season_type_all_star,
    )
    return log.get_data_frames()[0]


def fetch_box_score(game_id):
    '''Fetch traditional box score for a game.'''
    log = nba_api_call(
        boxscoretraditionalv2.BoxScoreTraditionalV2,
        game_id=game_id,
    )
    return log.get_data_frames()[0]
