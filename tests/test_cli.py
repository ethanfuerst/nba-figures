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
    fig, ax = plt.subplots()
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
    fig, ax = plt.subplots()
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
