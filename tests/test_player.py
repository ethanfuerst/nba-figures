import matplotlib.pyplot as plt
import pandas as pd
import pytest

from nbafigs.core.player import NBAPlayer
from nbafigs.errors import PlayerNotFoundError, SeasonNotFoundError


def test_player_init(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )

    player = NBAPlayer('Russell Westbrook', print_name=False)

    assert player.name == 'Russell Westbrook'
    assert player.player_id == 201566
    assert player.is_active is True


def test_player_init_not_found(mocker):
    mocker.patch('nbafigs.core.player.find_player', return_value=None)

    with pytest.raises(PlayerNotFoundError):
        NBAPlayer('Fake Player', print_name=False)


def test_player_str(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    assert str(player) == 'Russell Westbrook'


def test_player_repr(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    assert 'Russell Westbrook' in repr(player)


def test_player_get_career(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    career = player.get_career()

    assert 'Player' in career.columns
    assert 'Season' in career.columns
    assert len(career) == 3


def test_player_get_season(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    game_log = pd.DataFrame(
        {
            'Game_ID': ['001'],
            'GAME_DATE': ['2023-01-01'],
            'MATCHUP': ['OKC vs. LAL'],
            'WL': ['W'],
            'MIN': [30],
            'FGM': [10],
            'FGA': [20],
            'FG_PCT': [0.5],
            'FG3M': [3],
            'FG3A': [8],
            'FG3_PCT': [0.375],
            'FTM': [5],
            'FTA': [6],
            'FT_PCT': [0.833],
            'OREB': [1],
            'DREB': [5],
            'REB': [6],
            'AST': [8],
            'STL': [2],
            'BLK': [0],
            'TOV': [3],
            'PF': [2],
            'PTS': [28],
            'PLUS_MINUS': [10],
        }
    )
    mocker.patch('nbafigs.core.player.fetch_player_game_log', return_value=game_log)
    player = NBAPlayer('Russell Westbrook', print_name=False)

    df = player.get_season(2023)

    assert len(df) == 1
    assert 'TS_PCT' in df.columns


def test_player_get_season_empty_raises(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    mocker.patch(
        'nbafigs.core.player.fetch_player_game_log', return_value=pd.DataFrame()
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    with pytest.raises(SeasonNotFoundError):
        player.get_season(1990)


def test_player_get_shot_chart(
    mocker, mock_player_search, mock_career_df, sample_shots_df, sample_avgs_df
):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    mocker.patch(
        'nbafigs.core.player.fetch_shot_chart',
        return_value=(sample_shots_df, sample_avgs_df),
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    to_plot, fig = player.get_shot_chart(seasons=[2010])

    assert isinstance(fig, plt.Figure)
    assert 'X' in to_plot.columns
    plt.close(fig)
