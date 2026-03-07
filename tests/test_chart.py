import matplotlib.pyplot as plt
import pytest

from nbafigs.viz.chart import make_shot_chart, shots_grouper, zone_label


@pytest.mark.parametrize(
    ('shot_range', 'area', 'basic', 'expected'),
    [
        ('Less Than 8 ft.', 'Center(C)', 'Restricted Area', 'Less Than 8 ft.'),
        ('8-16 ft.', 'Left Side(L)', 'Mid-Range', '8-16 ft. (L)'),
        ('8-16 ft.', 'Right Side(L)', 'Mid-Range', '8-16 ft. (R)'),
        ('8-16 ft.', 'Center(C)', 'Mid-Range', '8-16 ft. (C)'),
        ('16-24 ft.', 'Left Side(L)', 'Mid-Range', '16-24 ft. (L)'),
        ('16-24 ft.', 'Right Side(L)', 'Mid-Range', '16-24 ft. (R)'),
        ('16-24 ft.', 'Left Side Center(LC)', 'Mid-Range', '16-24 ft. (LC)'),
        ('16-24 ft.', 'Right Side Center(LC)', 'Mid-Range', '16-24 ft. (RC)'),
        ('16-24 ft.', 'Center(C)', 'Mid-Range', 'Mid Range (C)'),
        ('24+ ft.', 'Left Side(L)', 'Left Corner 3', 'Left Corner 3'),
        ('24+ ft.', 'Right Side(R)', 'Right Corner 3', 'Right Corner 3'),
        ('24+ ft.', 'Center(C)', 'Above the Break 3', '3 Pointer (C)'),
        ('24+ ft.', 'Left Side Center(LC)', 'Above the Break 3', '3 Pointer (LC)'),
        ('24+ ft.', 'Right Side Center(LC)', 'Above the Break 3', '3 Pointer (RC)'),
        ('Back Court Shot', 'Back Court(BC)', 'Backcourt', 'Backcourt'),
    ],
)
def test_zone_label(shot_range, area, basic, expected):
    row = {
        'SHOT_ZONE_RANGE': shot_range,
        'SHOT_ZONE_AREA': area,
        'SHOT_ZONE_BASIC': basic,
    }

    assert zone_label(row) == expected


def test_shots_grouper_returns_expected_columns(sample_shots_df, sample_avgs_df):
    result = shots_grouper(sample_shots_df, sample_avgs_df)

    expected_cols = [
        'X',
        'Y',
        'SHOT_DISTANCE',
        'PTS',
        'SHOT_TYPE',
        'SHOT_MADE',
        'ZONE',
        'PLAYER_PCT',
        'LEAGUE_PCT',
        'PCT_DIFF',
        'P_PPS',
        'L_PPS',
        'D_PPS',
    ]
    assert list(result.columns) == expected_cols


def test_shots_grouper_calculates_pps(sample_shots_df, sample_avgs_df):
    result = shots_grouper(sample_shots_df, sample_avgs_df)

    assert 'P_PPS' in result.columns
    assert 'L_PPS' in result.columns
    assert 'D_PPS' in result.columns


def test_make_shot_chart_normal(sample_grouped_df):
    fig = make_shot_chart(sample_grouped_df, kind='normal')

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_shot_chart_normal_no_misses(sample_grouped_df):
    fig = make_shot_chart(sample_grouped_df, kind='normal', show_misses=False)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_shot_chart_hex(sample_grouped_df):
    fig = make_shot_chart(sample_grouped_df, kind='hex')

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_shot_chart_hex_pct_diff(sample_grouped_df):
    fig = make_shot_chart(sample_grouped_df, kind='hex', scale='PCT_DIFF')

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_shot_chart_hex_d_pps(sample_grouped_df):
    fig = make_shot_chart(sample_grouped_df, kind='hex', scale='D_PPS')

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_shot_chart_with_title(sample_grouped_df):
    fig = make_shot_chart(sample_grouped_df, title='Test Chart', context='Some context')

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_make_shot_chart_no_pct(sample_grouped_df):
    fig = make_shot_chart(sample_grouped_df, show_pct=False)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)
