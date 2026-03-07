import pandas as pd

from nbafigs.sources import nba_api_call
from nbafigs.sources.player import (
    fetch_player_career_stats,
    fetch_player_game_log,
    fetch_shot_chart,
    find_player,
)
from nbafigs.sources.team import (
    fetch_box_score,
    fetch_team_game_log,
    find_team,
    get_team_id,
)


def test_nba_api_call_invokes_func(mocker):
    mock_func = mocker.MagicMock(return_value='result')
    mocker.patch('nbafigs.sources.time.sleep')

    result = nba_api_call(mock_func, 'arg1', key='val')

    mock_func.assert_called_once_with('arg1', key='val')
    assert result == 'result'


def test_nba_api_call_sleeps_after(mocker):
    mock_func = mocker.MagicMock(return_value='ok')
    mock_sleep = mocker.patch('nbafigs.sources.time.sleep')

    nba_api_call(mock_func)

    mock_sleep.assert_called_once_with(0.6)


def test_find_player_returns_dict(mocker):
    mocker.patch(
        'nbafigs.sources.player.players.find_players_by_full_name',
        return_value=[{'id': 123, 'full_name': 'Test Player'}],
    )

    result = find_player('Test Player')

    assert result == {'id': 123, 'full_name': 'Test Player'}


def test_find_player_returns_none_when_not_found(mocker):
    mocker.patch(
        'nbafigs.sources.player.players.find_players_by_full_name',
        return_value=[],
    )

    result = find_player('Nobody')

    assert result is None


def test_fetch_player_game_log(mocker):
    mock_df = pd.DataFrame({'PTS': [20, 30]})
    mock_endpoint = mocker.MagicMock()
    mock_endpoint.get_data_frames.return_value = [mock_df]
    mocker.patch('nbafigs.sources.player.nba_api_call', return_value=mock_endpoint)

    result = fetch_player_game_log(123, '2023-24', 'Regular Season')

    assert len(result) == 2


def test_fetch_player_career_stats(mocker):
    mock_df = pd.DataFrame({'GP': [82]})
    mock_endpoint = mocker.MagicMock()
    mock_endpoint.get_data_frames.return_value = [mock_df]
    mocker.patch('nbafigs.sources.player.nba_api_call', return_value=mock_endpoint)

    result = fetch_player_career_stats(123)

    assert len(result) == 1


def test_fetch_shot_chart_returns_tuple(mocker):
    shots_df = pd.DataFrame({'LOC_X': [10]})
    avgs_df = pd.DataFrame({'FGM': [100]})
    mock_endpoint = mocker.MagicMock()
    mock_endpoint.get_data_frames.return_value = [shots_df, avgs_df]
    mocker.patch('nbafigs.sources.player.nba_api_call', return_value=mock_endpoint)

    shots, avgs = fetch_shot_chart(123)

    assert len(shots) == 1
    assert len(avgs) == 1


def test_find_team_returns_dict(mocker):
    mocker.patch(
        'nbafigs.sources.team.teams.get_teams',
        return_value=[{'id': 1, 'abbreviation': 'DAL', 'full_name': 'Dallas Mavericks'}],
    )

    result = find_team('DAL')

    assert result['abbreviation'] == 'DAL'


def test_find_team_returns_none(mocker):
    mocker.patch(
        'nbafigs.sources.team.teams.get_teams',
        return_value=[{'id': 1, 'abbreviation': 'DAL', 'full_name': 'Dallas Mavericks'}],
    )

    result = find_team('XXX')

    assert result is None


def test_get_team_id_returns_id(mocker):
    mocker.patch(
        'nbafigs.sources.team.find_team',
        return_value={'id': 42, 'abbreviation': 'DAL'},
    )

    result = get_team_id('DAL')

    assert result == 42


def test_get_team_id_returns_none_when_not_found(mocker):
    mocker.patch('nbafigs.sources.team.find_team', return_value=None)

    result = get_team_id('XXX')

    assert result is None


def test_fetch_team_game_log(mocker):
    mock_df = pd.DataFrame({'WL': ['W', 'L']})
    mock_endpoint = mocker.MagicMock()
    mock_endpoint.get_data_frames.return_value = [mock_df]
    mocker.patch('nbafigs.sources.team.nba_api_call', return_value=mock_endpoint)

    result = fetch_team_game_log(1, '2023-24', 'Regular Season')

    assert len(result) == 2


def test_fetch_box_score(mocker):
    mock_df = pd.DataFrame({'PLAYER_ID': [123, 456]})
    mock_endpoint = mocker.MagicMock()
    mock_endpoint.get_data_frames.return_value = [mock_df]
    mocker.patch('nbafigs.sources.team.nba_api_call', return_value=mock_endpoint)

    result = fetch_box_score('0042300401')

    assert len(result) == 2
