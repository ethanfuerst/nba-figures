import matplotlib.pyplot as plt

from nbafigs import player_shot_chart, team_shot_chart


def test_player_shot_chart(mocker, sample_grouped_df):
    fig, _ = plt.subplots()
    mocker.patch(
        'nbafigs.NBAPlayer',
        return_value=mocker.Mock(
            get_shot_chart=mocker.Mock(return_value=(sample_grouped_df, fig))
        ),
    )

    df, result_fig = player_shot_chart('Test Player', seasons=[2023])

    assert 'X' in df.columns
    assert isinstance(result_fig, plt.Figure)
    plt.close(fig)


def test_player_shot_chart_default_params(mocker, sample_grouped_df):
    fig, _ = plt.subplots()
    mock_player = mocker.Mock(
        get_shot_chart=mocker.Mock(return_value=(sample_grouped_df, fig))
    )
    mocker.patch('nbafigs.NBAPlayer', return_value=mock_player)

    player_shot_chart('Test Player')

    mock_player.get_shot_chart.assert_called_once_with(
        seasons=None, chart_params={},
    )
    plt.close(fig)


def test_player_shot_chart_none_chart_params(mocker, sample_grouped_df):
    """Covers the chart_params is None branch."""
    fig, _ = plt.subplots()
    mock_player = mocker.Mock(
        get_shot_chart=mocker.Mock(return_value=(sample_grouped_df, fig))
    )
    mocker.patch('nbafigs.NBAPlayer', return_value=mock_player)

    player_shot_chart('Test Player', chart_params=None)

    mock_player.get_shot_chart.assert_called_once_with(
        seasons=None, chart_params={},
    )
    plt.close(fig)


def test_team_shot_chart(mocker, sample_grouped_df):
    fig, _ = plt.subplots()
    mocker.patch(
        'nbafigs.NBATeam',
        return_value=mocker.Mock(
            get_shot_chart=mocker.Mock(return_value=(sample_grouped_df, fig))
        ),
    )

    df, result_fig = team_shot_chart('DAL', '0042300401')

    assert 'X' in df.columns
    assert isinstance(result_fig, plt.Figure)
    plt.close(fig)


def test_team_shot_chart_default_params(mocker, sample_grouped_df):
    fig, _ = plt.subplots()
    mock_team = mocker.Mock(
        get_shot_chart=mocker.Mock(return_value=(sample_grouped_df, fig))
    )
    mocker.patch('nbafigs.NBATeam', return_value=mock_team)

    team_shot_chart('DAL', '0042300401')

    mock_team.get_shot_chart.assert_called_once_with(
        game_id='0042300401', playoffs=False, chart_params={},
    )
    plt.close(fig)
