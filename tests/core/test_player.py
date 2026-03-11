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


def test_player_get_shot_chart_default_params(
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

    _, fig = player.get_shot_chart(seasons=[2010], chart_params=None)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_player_print_name(mocker, mock_player_search, mock_career_df, capsys):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )

    NBAPlayer('Russell Westbrook', print_name=True)

    captured = capsys.readouterr()
    assert 'Russell Westbrook' in captured.out


def test_player_get_season_default_season(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    game_log = pd.DataFrame(
        {
            'Game_ID': ['001'],
            'GAME_DATE': ['2025-01-01'],
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

    df = player.get_season()

    assert len(df) == 1


def test_player_get_season_invalid_type(mocker, mock_player_search, mock_career_df):
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

    # invalid season_type falls back to 'regular'
    df = player.get_season(2023, season_type='invalid')

    assert len(df) == 1


def test_player_get_season_exception(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    mocker.patch(
        'nbafigs.core.player.fetch_player_game_log',
        side_effect=Exception('API error'),
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    with pytest.raises(Exception, match='API error'):
        player.get_season(2023)


def test_player_get_full_career_regular(mocker, mock_player_search, mock_career_df):
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

    df = player.get_full_career(season_type='regular')

    assert len(df) == 3
    assert 'Season Type' in df.columns


def test_player_get_full_career_all(mocker, mock_player_search, mock_career_df):
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

    df = player.get_full_career(season_type='all')

    assert len(df) > 0
    assert 'Season Type' in df.columns


def test_player_get_full_career_empty(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    mocker.patch(
        'nbafigs.core.player.fetch_player_game_log',
        side_effect=SeasonNotFoundError('no data'),
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    df = player.get_full_career(season_type='regular')

    assert df.empty


def test_player_get_full_career_no_all_star(mocker, mock_player_search, mock_career_df):
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

    df = player.get_full_career(season_type='no_all_star')

    assert len(df) > 0


def test_player_get_full_career_preseason_and_allstar_not_found(
    mocker, mock_player_search, mock_career_df
):
    """Covers SeasonNotFoundError catches for preseason, playoffs, and allstar."""
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    # Return empty DF to trigger SeasonNotFoundError for all season types
    mocker.patch(
        'nbafigs.core.player.fetch_player_game_log',
        return_value=pd.DataFrame(),
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    df = player.get_full_career(season_type='all')

    assert df.empty


def test_player_format_shots_default_season(
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

    chart_params = {}
    to_plot = player.format_shots(None, chart_params)

    assert 'X' in to_plot.columns
    assert 'title' in chart_params


def test_player_format_shots_two_seasons(
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

    chart_params = {}
    to_plot = player.format_shots([2008, 2010], chart_params)

    assert 'X' in to_plot.columns
    assert 'to' in chart_params['title']


def test_player_format_shots_consecutive_seasons(
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

    chart_params = {}
    to_plot = player.format_shots([2009, 2010], chart_params)

    assert 'X' in to_plot.columns
    assert 'and' in chart_params['title']


def test_player_format_shots_with_date_range(
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
    mocker.patch('nbafigs.core.player.get_team_id', return_value=1610612742)
    player = NBAPlayer('Russell Westbrook', print_name=False)

    chart_params = {}
    to_plot = player.format_shots(
        None,
        chart_params,
        DateFrom='01-01-2023',
        DateTo='03-01-2023',
    )

    assert 'X' in to_plot.columns
    assert 'from' in chart_params['title']


def test_player_format_shots_with_opponent(
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
    mocker.patch('nbafigs.core.player.get_team_id', return_value=1610612742)
    player = NBAPlayer('Russell Westbrook', print_name=False)

    chart_params = {}
    to_plot = player.format_shots([2010], chart_params, OpponentTeam='DAL')

    assert 'X' in to_plot.columns


def test_player_format_shots_many_seasons(
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

    chart_params = {}
    to_plot = player.format_shots([2008, 2009, 2010], chart_params)

    assert not to_plot.empty
    assert 'to' in chart_params['title']


def test_player_format_shots_no_data(mocker, mock_player_search, mock_career_df):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    mocker.patch(
        'nbafigs.core.player.fetch_shot_chart',
        return_value=(pd.DataFrame(), pd.DataFrame()),
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    with pytest.raises(SeasonNotFoundError):
        player.format_shots([2010], {})


def test_player_format_shots_no_data_multi_season(
    mocker, mock_player_search, mock_career_df
):
    mocker.patch('nbafigs.core.player.find_player', return_value=mock_player_search)
    mocker.patch(
        'nbafigs.core.player.fetch_player_career_stats', return_value=mock_career_df
    )
    mocker.patch(
        'nbafigs.core.player.fetch_shot_chart',
        return_value=(pd.DataFrame(), pd.DataFrame()),
    )
    player = NBAPlayer('Russell Westbrook', print_name=False)

    with pytest.raises(SeasonNotFoundError):
        player.format_shots([2008, 2010], {})


def test_player_format_shots_custom_title(
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

    chart_params = {'title': 'Custom Title'}
    player.format_shots([2010], chart_params)

    assert chart_params['title'] == 'Custom Title'
