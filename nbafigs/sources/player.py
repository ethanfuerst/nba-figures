from nba_api.stats.endpoints import playercareerstats, playergamelog, shotchartdetail
from nba_api.stats.static import players

from nbafigs.sources import nba_api_call


def find_player(player_name):
    """Look up a player by name. Return the first match dict or None."""
    results = players.find_players_by_full_name(player_name)
    return results[0] if results else None


def fetch_player_game_log(player_id, season, season_type_all_star):
    """Fetch a single-season game log for a player."""
    log = nba_api_call(
        playergamelog.PlayerGameLog,
        player_id=player_id,
        season=season,
        season_type_all_star=season_type_all_star,
    )
    return log.get_data_frames()[0]


def fetch_player_career_stats(player_id):
    """Fetch career totals for a player."""
    log = nba_api_call(
        playercareerstats.PlayerCareerStats,
        player_id=player_id,
        per_mode36='Totals',
    )
    return log.get_data_frames()[0]


def fetch_shot_chart(player_id, team_id=0, **kwargs):
    """Fetch shot chart detail. Returns (shots_df, league_avg_df)."""
    log = nba_api_call(
        shotchartdetail.ShotChartDetail,
        team_id=team_id,
        player_id=player_id,
        context_measure_simple=['FGA', 'FG3A'],
        **kwargs,
    )
    return log.get_data_frames()[0], log.get_data_frames()[1]
