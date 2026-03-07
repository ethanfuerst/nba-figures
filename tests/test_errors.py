import pytest

from nbafigs.errors import PlayerNotFoundError, SeasonNotFoundError, TeamNotFoundError


def test_player_not_found_error_is_exception():
    with pytest.raises(PlayerNotFoundError):
        raise PlayerNotFoundError('not found')


def test_season_not_found_error_is_exception():
    with pytest.raises(SeasonNotFoundError):
        raise SeasonNotFoundError('no data')


def test_team_not_found_error_is_exception():
    with pytest.raises(TeamNotFoundError):
        raise TeamNotFoundError('bad abbreviation')


def test_errors_have_message():
    err = PlayerNotFoundError('test message')

    assert str(err) == 'test message'


def test_errors_are_catchable_as_exception():
    with pytest.raises(Exception):
        raise PlayerNotFoundError('caught as Exception')
