import matplotlib.pyplot as plt
import pandas as pd
import pytest

from nbafigs.core.team import NBATeam
from nbafigs.errors import TeamNotFoundError


def _mock_teams_static(mocker):
    mocker.patch(
        'nbafigs.core.team.teams.get_teams',
        return_value=[
            {
                'id': 1610612742,
                'abbreviation': 'DAL',
                'full_name': 'Dallas Mavericks',
                'nickname': 'Mavericks',
                'city': 'Dallas',
                'state': 'Texas',
                'year_founded': 1980,
            },
            {
                'id': 1610612747,
                'abbreviation': 'LAL',
                'full_name': 'Los Angeles Lakers',
                'nickname': 'Lakers',
                'city': 'Los Angeles',
                'state': 'California',
                'year_founded': 1948,
            },
        ],
    )


def test_team_init(mocker):
    _mock_teams_static(mocker)

    team = NBATeam('DAL')

    assert team.abbrev == 'DAL'
    assert team.full_name == 'Dallas Mavericks'
    assert team.id == 1610612742


def test_team_init_not_found(mocker):
    _mock_teams_static(mocker)

    with pytest.raises(TeamNotFoundError):
        NBATeam('XXX')


def test_team_get_season(mocker):
    _mock_teams_static(mocker)
    mock_log = pd.DataFrame({'game': ['data']})
    mocker.patch('nbafigs.core.team.fetch_team_game_log', return_value=mock_log)

    team = NBATeam('DAL')
    result = team.get_season(2023, 'Regular Season')

    assert len(result) == 1


def test_team_get_shot_chart_playoffs(mocker, sample_shots_df, sample_avgs_df):
    _mock_teams_static(mocker)

    box_score_df = pd.DataFrame(
        {
            'PLAYER_ID': [101],
            'TEAM_ABBREVIATION': ['DAL'],
            'TEAM_ID': [1610612742],
            'MIN': ['30:00'],
        }
    )
    opp_rows = pd.DataFrame(
        {
            'PLAYER_ID': [201],
            'TEAM_ABBREVIATION': ['LAL'],
            'TEAM_ID': [1610612747],
            'MIN': ['30:00'],
        }
    )
    full_box = pd.concat([box_score_df, opp_rows], ignore_index=True)

    shots_with_date = sample_shots_df.copy()
    shots_with_date['GAME_DATE'] = '20230101'

    mocker.patch('nbafigs.core.team.fetch_box_score', return_value=full_box)
    mocker.patch(
        'nbafigs.core.team.fetch_shot_chart',
        return_value=(shots_with_date, sample_avgs_df),
    )

    team = NBATeam('DAL')
    _, fig = team.get_shot_chart(game_id='0042300401', playoffs=True)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_team_get_shot_chart_default_params(mocker, sample_shots_df, sample_avgs_df):
    _mock_teams_static(mocker)

    box_score_df = pd.DataFrame(
        {
            'PLAYER_ID': [101],
            'TEAM_ABBREVIATION': ['DAL'],
            'TEAM_ID': [1610612742],
            'MIN': ['30:00'],
        }
    )
    opp_rows = pd.DataFrame(
        {
            'PLAYER_ID': [201],
            'TEAM_ABBREVIATION': ['LAL'],
            'TEAM_ID': [1610612747],
            'MIN': ['30:00'],
        }
    )
    full_box = pd.concat([box_score_df, opp_rows], ignore_index=True)

    shots_with_date = sample_shots_df.copy()
    shots_with_date['GAME_DATE'] = '20230101'

    mocker.patch('nbafigs.core.team.fetch_box_score', return_value=full_box)
    mocker.patch(
        'nbafigs.core.team.fetch_shot_chart',
        return_value=(shots_with_date, sample_avgs_df),
    )

    team = NBATeam('DAL')
    _, fig = team.get_shot_chart(game_id='0042300401', chart_params=None)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_team_get_shot_chart(mocker, sample_shots_df, sample_avgs_df):
    _mock_teams_static(mocker)

    box_score_df = pd.DataFrame(
        {
            'PLAYER_ID': [101, 102],
            'TEAM_ABBREVIATION': ['DAL', 'DAL'],
            'TEAM_ID': [1610612742, 1610612742],
            'MIN': ['30:00', '25:00'],
        }
    )
    # Add opponent rows
    opp_rows = pd.DataFrame(
        {
            'PLAYER_ID': [201, 202],
            'TEAM_ABBREVIATION': ['LAL', 'LAL'],
            'TEAM_ID': [1610612747, 1610612747],
            'MIN': ['30:00', '25:00'],
        }
    )
    full_box = pd.concat([box_score_df, opp_rows], ignore_index=True)

    # Add GAME_DATE to shots so the title can be generated
    shots_with_date = sample_shots_df.copy()
    shots_with_date['GAME_DATE'] = '20230101'

    mocker.patch('nbafigs.core.team.fetch_box_score', return_value=full_box)
    mocker.patch(
        'nbafigs.core.team.fetch_shot_chart',
        return_value=(shots_with_date, sample_avgs_df),
    )

    team = NBATeam('DAL')
    to_plot, fig = team.get_shot_chart(game_id='0042300401')

    assert isinstance(fig, plt.Figure)
    assert 'X' in to_plot.columns
    plt.close(fig)
