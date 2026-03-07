import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

from nbafigs.viz.chart import make_shot_chart, shots_grouper, zone_label


def test_zone_label_less_than_8ft():
    row = {
        'SHOT_ZONE_RANGE': 'Less Than 8 ft.',
        'SHOT_ZONE_AREA': 'Center(C)',
        'SHOT_ZONE_BASIC': 'Restricted Area',
    }

    assert zone_label(row) == 'Less Than 8 ft.'


def test_zone_label_8_16_left():
    row = {
        'SHOT_ZONE_RANGE': '8-16 ft.',
        'SHOT_ZONE_AREA': 'Left Side(L)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == '8-16 ft. (L)'


def test_zone_label_8_16_right():
    row = {
        'SHOT_ZONE_RANGE': '8-16 ft.',
        'SHOT_ZONE_AREA': 'Right Side(L)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == '8-16 ft. (R)'


def test_zone_label_8_16_center():
    row = {
        'SHOT_ZONE_RANGE': '8-16 ft.',
        'SHOT_ZONE_AREA': 'Center(C)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == '8-16 ft. (C)'


def test_zone_label_16_24_left():
    row = {
        'SHOT_ZONE_RANGE': '16-24 ft.',
        'SHOT_ZONE_AREA': 'Left Side(L)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == '16-24 ft. (L)'


def test_zone_label_16_24_right():
    row = {
        'SHOT_ZONE_RANGE': '16-24 ft.',
        'SHOT_ZONE_AREA': 'Right Side(L)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == '16-24 ft. (R)'


def test_zone_label_16_24_left_center():
    row = {
        'SHOT_ZONE_RANGE': '16-24 ft.',
        'SHOT_ZONE_AREA': 'Left Side Center(LC)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == '16-24 ft. (LC)'


def test_zone_label_16_24_right_center():
    row = {
        'SHOT_ZONE_RANGE': '16-24 ft.',
        'SHOT_ZONE_AREA': 'Right Side Center(LC)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == '16-24 ft. (RC)'


def test_zone_label_16_24_center():
    row = {
        'SHOT_ZONE_RANGE': '16-24 ft.',
        'SHOT_ZONE_AREA': 'Center(C)',
        'SHOT_ZONE_BASIC': 'Mid-Range',
    }

    assert zone_label(row) == 'Mid Range (C)'


def test_zone_label_left_corner_3():
    row = {
        'SHOT_ZONE_RANGE': '24+ ft.',
        'SHOT_ZONE_AREA': 'Left Side(L)',
        'SHOT_ZONE_BASIC': 'Left Corner 3',
    }

    assert zone_label(row) == 'Left Corner 3'


def test_zone_label_right_corner_3():
    row = {
        'SHOT_ZONE_RANGE': '24+ ft.',
        'SHOT_ZONE_AREA': 'Right Side(R)',
        'SHOT_ZONE_BASIC': 'Right Corner 3',
    }

    assert zone_label(row) == 'Right Corner 3'


def test_zone_label_above_break_3_center():
    row = {
        'SHOT_ZONE_RANGE': '24+ ft.',
        'SHOT_ZONE_AREA': 'Center(C)',
        'SHOT_ZONE_BASIC': 'Above the Break 3',
    }

    assert zone_label(row) == '3 Pointer (C)'


def test_zone_label_above_break_3_left_center():
    row = {
        'SHOT_ZONE_RANGE': '24+ ft.',
        'SHOT_ZONE_AREA': 'Left Side Center(LC)',
        'SHOT_ZONE_BASIC': 'Above the Break 3',
    }

    assert zone_label(row) == '3 Pointer (LC)'


def test_zone_label_above_break_3_right_center():
    row = {
        'SHOT_ZONE_RANGE': '24+ ft.',
        'SHOT_ZONE_AREA': 'Right Side Center(LC)',
        'SHOT_ZONE_BASIC': 'Above the Break 3',
    }

    assert zone_label(row) == '3 Pointer (RC)'


def test_zone_label_backcourt():
    row = {
        'SHOT_ZONE_RANGE': 'Back Court Shot',
        'SHOT_ZONE_AREA': 'Back Court(BC)',
        'SHOT_ZONE_BASIC': 'Backcourt',
    }

    assert zone_label(row) == 'Backcourt'


def test_shots_grouper_returns_expected_columns(sample_shots_df, sample_avgs_df):
    result = shots_grouper(sample_shots_df, sample_avgs_df)

    expected_cols = [
        'X', 'Y', 'SHOT_DISTANCE', 'PTS', 'SHOT_TYPE', 'SHOT_MADE', 'ZONE',
        'PLAYER_PCT', 'LEAGUE_PCT', 'PCT_DIFF', 'P_PPS', 'L_PPS', 'D_PPS',
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
