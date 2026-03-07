import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use('Agg')


@pytest.fixture
def sample_shots_df():
    '''Sample shots DataFrame mimicking nba_api ShotChartDetail output.'''
    return pd.DataFrame({
        'LOC_X': [10, -50, 200, -220, 0, 100],
        'LOC_Y': [20, 100, 50, 80, 5, 300],
        'SHOT_DISTANCE': [3, 12, 22, 24, 1, 25],
        'SHOT_TYPE': ['2PT Field Goal', '2PT Field Goal', '3PT Field Goal',
                      '3PT Field Goal', '2PT Field Goal', '3PT Field Goal'],
        'ACTION_TYPE': ['Layup', 'Jump Shot', 'Jump Shot',
                        'Jump Shot', 'Dunk', 'Jump Shot'],
        'SHOT_MADE_FLAG': [1, 0, 1, 0, 1, 1],
        'SHOT_ATTEMPTED_FLAG': [1, 1, 1, 1, 1, 1],
        'SHOT_ZONE_BASIC': ['Restricted Area', 'Mid-Range', 'Above the Break 3',
                            'Left Corner 3', 'Restricted Area', 'Above the Break 3'],
        'SHOT_ZONE_AREA': ['Center(C)', 'Left Side(L)', 'Center(C)',
                           'Left Side(L)', 'Center(C)', 'Right Side Center(LC)'],
        'SHOT_ZONE_RANGE': ['Less Than 8 ft.', '8-16 ft.', '24+ ft.',
                            '24+ ft.', 'Less Than 8 ft.', '24+ ft.'],
    })


@pytest.fixture
def sample_avgs_df():
    '''Sample league averages DataFrame.'''
    return pd.DataFrame({
        'SHOT_ZONE_BASIC': ['Restricted Area', 'Mid-Range', 'Above the Break 3',
                            'Left Corner 3'],
        'SHOT_ZONE_AREA': ['Center(C)', 'Left Side(L)', 'Center(C)',
                           'Left Side(L)'],
        'SHOT_ZONE_RANGE': ['Less Than 8 ft.', '8-16 ft.', '24+ ft.',
                            '24+ ft.'],
        'FGM': [100, 40, 35, 20],
        'FGA': [150, 100, 100, 50],
        'FG_PCT': [0.667, 0.400, 0.350, 0.400],
    })


@pytest.fixture
def sample_grouped_df():
    '''Sample DataFrame from shots_grouper(), ready for make_shot_chart().'''
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        'X': np.random.randint(-250, 250, n),
        'Y': np.random.randint(-47, 400, n),
        'SHOT_DISTANCE': np.random.randint(0, 30, n),
        'PTS': np.random.choice([2, 3], n),
        'SHOT_TYPE': ['Jump Shot'] * n,
        'SHOT_MADE': np.random.choice([0, 1], n),
        'ZONE': np.random.choice(['Less Than 8 ft.', '8-16 ft. (L)', '3 Pointer (C)'], n),
        'PLAYER_PCT': np.random.uniform(0.3, 0.6, n),
        'LEAGUE_PCT': np.random.uniform(0.3, 0.5, n),
        'PCT_DIFF': np.random.uniform(-0.1, 0.1, n),
        'P_PPS': np.random.uniform(0.5, 1.5, n),
        'L_PPS': np.random.uniform(0.5, 1.2, n),
        'D_PPS': np.random.uniform(-0.3, 0.3, n),
    })


@pytest.fixture
def mock_player_search():
    '''Mock result from players.find_players_by_full_name().'''
    return {
        'id': 201566,
        'full_name': 'Russell Westbrook',
        'first_name': 'Russell',
        'last_name': 'Westbrook',
        'is_active': True,
    }


@pytest.fixture
def mock_career_df():
    '''Mock career stats DataFrame.'''
    return pd.DataFrame({
        'SEASON_ID': ['2008-09', '2009-10', '2010-11'],
        'PLAYER_ID': [201566, 201566, 201566],
        'LEAGUE_ID': ['00', '00', '00'],
        'TEAM_ID': [1610612760, 1610612760, 1610612760],
        'TEAM_ABBREVIATION': ['OKC', 'OKC', 'OKC'],
        'PLAYER_AGE': [20.0, 21.0, 22.0],
        'GP': [82, 82, 82],
        'GS': [82, 82, 82],
        'MIN': [2600, 2800, 2900],
        'FGM': [500, 550, 600],
        'FGA': [1200, 1250, 1300],
        'FG_PCT': [0.417, 0.440, 0.462],
        'FG3M': [50, 55, 60],
        'FG3A': [180, 190, 200],
        'FG3_PCT': [0.278, 0.289, 0.300],
        'FTM': [300, 350, 400],
        'FTA': [400, 450, 500],
        'FT_PCT': [0.750, 0.778, 0.800],
        'OREB': [50, 55, 60],
        'DREB': [300, 320, 340],
        'REB': [350, 375, 400],
        'AST': [500, 550, 600],
        'STL': [100, 110, 120],
        'BLK': [20, 22, 24],
        'TOV': [200, 210, 220],
        'PF': [150, 155, 160],
        'PTS': [1350, 1500, 1660],
    })
