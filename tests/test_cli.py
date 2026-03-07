import matplotlib.pyplot as plt
from click.testing import CliRunner

from nbafigs.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])

    assert result.exit_code == 0
    assert 'NBA shot chart generator' in result.output


def test_cli_player_help():
    runner = CliRunner()
    result = runner.invoke(main, ['player', '--help'])

    assert result.exit_code == 0
    assert 'PLAYER_NAME' in result.output


def test_cli_team_help():
    runner = CliRunner()
    result = runner.invoke(main, ['team', '--help'])

    assert result.exit_code == 0
    assert 'TEAM_ABBREV' in result.output


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ['--version'])

    assert result.exit_code == 0
    assert '0.1.0' in result.output


def test_cli_player_with_output(mocker, tmp_path, sample_grouped_df):
    fig, _ = plt.subplots()
    mocker.patch(
        'nbafigs.cli.player_shot_chart',
        return_value=(sample_grouped_df, fig),
    )
    output_path = str(tmp_path / 'test.png')
    runner = CliRunner()

    result = runner.invoke(
        main, ['player', 'Test Player', '-s', '2023', '-o', output_path]
    )

    assert result.exit_code == 0
    assert f'Saved to {output_path}' in result.output
    plt.close(fig)


def test_cli_team_with_output(mocker, tmp_path, sample_grouped_df):
    fig, _ = plt.subplots()
    mocker.patch(
        'nbafigs.cli.team_shot_chart',
        return_value=(sample_grouped_df, fig),
    )
    output_path = str(tmp_path / 'test.png')
    runner = CliRunner()

    result = runner.invoke(
        main, ['team', 'DAL', '--game-id', '0042300401', '-o', output_path]
    )

    assert result.exit_code == 0
    assert f'Saved to {output_path}' in result.output
    plt.close(fig)


def test_cli_player_with_title_and_context(mocker, tmp_path, sample_grouped_df):
    fig, _ = plt.subplots()
    mock_fn = mocker.patch(
        'nbafigs.cli.player_shot_chart',
        return_value=(sample_grouped_df, fig),
    )
    output_path = str(tmp_path / 'test.png')
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            'player',
            'Test Player',
            '--title',
            'My Title',
            '--context',
            'My Context',
            '-o',
            output_path,
        ],
    )

    assert result.exit_code == 0
    call_kwargs = mock_fn.call_args
    assert call_kwargs[1]['chart_params']['title'] == 'My Title'
    assert call_kwargs[1]['chart_params']['context'] == 'My Context'
    plt.close(fig)


def test_cli_player_with_limiters(mocker, tmp_path, sample_grouped_df):
    fig, _ = plt.subplots()
    mock_fn = mocker.patch(
        'nbafigs.cli.player_shot_chart',
        return_value=(sample_grouped_df, fig),
    )
    output_path = str(tmp_path / 'test.png')
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            'player',
            'Test Player',
            '--opponent',
            'DAL',
            '--location',
            'Home',
            '-o',
            output_path,
        ],
    )

    assert result.exit_code == 0
    call_kwargs = mock_fn.call_args
    assert call_kwargs[1]['OpponentTeam'] == 'DAL'
    assert call_kwargs[1]['Location'] == 'Home'
    plt.close(fig)


def test_cli_player_show(mocker, sample_grouped_df):
    fig, _ = plt.subplots()
    mocker.patch(
        'nbafigs.cli.player_shot_chart',
        return_value=(sample_grouped_df, fig),
    )
    mock_show = mocker.patch('nbafigs.cli.plt.show')
    runner = CliRunner()

    result = runner.invoke(main, ['player', 'Test Player'])

    assert result.exit_code == 0
    mock_show.assert_called_once()
    plt.close(fig)


def test_cli_team_show(mocker, sample_grouped_df):
    fig, _ = plt.subplots()
    mocker.patch(
        'nbafigs.cli.team_shot_chart',
        return_value=(sample_grouped_df, fig),
    )
    mock_show = mocker.patch('nbafigs.cli.plt.show')
    runner = CliRunner()

    result = runner.invoke(main, ['team', 'DAL', '--game-id', '0042300401'])

    assert result.exit_code == 0
    mock_show.assert_called_once()
    plt.close(fig)
