import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from nbafigs.viz.court import make_shot_fig


def zone_label(row):
    """Create a standardized zone label from shot zone columns.

    Args:
        row: A row from the shot chart DataFrame with SHOT_ZONE_RANGE,
            SHOT_ZONE_AREA, and SHOT_ZONE_BASIC columns.

    Returns:
        A string zone label.
    """
    if row['SHOT_ZONE_RANGE'] == '8-16 ft.':
        if row['SHOT_ZONE_AREA'] == 'Left Side(L)':
            return '8-16 ft. (L)'
        elif row['SHOT_ZONE_AREA'] == 'Right Side(L)':
            return '8-16 ft. (R)'
        else:
            return '8-16 ft. (C)'
    if row['SHOT_ZONE_RANGE'] == '16-24 ft.':
        if row['SHOT_ZONE_AREA'] == 'Left Side(L)':
            return '16-24 ft. (L)'
        elif row['SHOT_ZONE_AREA'] == 'Right Side(L)':
            return '16-24 ft. (R)'
        elif row['SHOT_ZONE_AREA'] == 'Left Side Center(LC)':
            return '16-24 ft. (LC)'
        elif row['SHOT_ZONE_AREA'] == 'Right Side Center(LC)':
            return '16-24 ft. (RC)'
        else:
            return 'Mid Range (C)'
    elif row['SHOT_ZONE_BASIC'] == 'Left Corner 3':
        return 'Left Corner 3'
    elif row['SHOT_ZONE_BASIC'] == 'Right Corner 3':
        return 'Right Corner 3'
    elif row['SHOT_ZONE_BASIC'] == 'Above the Break 3':
        if row['SHOT_ZONE_AREA'] == 'Left Side Center(LC)':
            return '3 Pointer (LC)'
        elif row['SHOT_ZONE_AREA'] == 'Right Side Center(LC)':
            return '3 Pointer (RC)'
        elif row['SHOT_ZONE_AREA'] == 'Center(C)':
            return '3 Pointer (C)'
        else:
            return 'Backcourt'
    elif row['SHOT_ZONE_RANGE'] == 'Less Than 8 ft.':
        return 'Less Than 8 ft.'
    else:
        return 'Backcourt'


def shots_grouper(shots, avgs):
    """Group shots by zone and merge with league averages.

    Args:
        shots: DataFrame of individual shot records from the nba_api.
        avgs: DataFrame of league average stats by zone.

    Returns:
        DataFrame with columns: X, Y, SHOT_DISTANCE, PTS, SHOT_TYPE,
        SHOT_MADE, ZONE, PLAYER_PCT, LEAGUE_PCT, PCT_DIFF, P_PPS, L_PPS, D_PPS.
    """
    shots['ZONE'] = shots.apply(lambda row: zone_label(row), axis=1)
    avgs['ZONE'] = avgs.apply(lambda row: zone_label(row), axis=1)

    shots_group = (
        shots.groupby(by=['ZONE'])
        .sum(numeric_only=True)
        .reset_index()[['ZONE', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG']]
        .copy()
    )
    shots_group['AVG_FG_PCT'] = round(
        shots_group['SHOT_MADE_FLAG'] / shots_group['SHOT_ATTEMPTED_FLAG'], 3
    )

    avgs = avgs.groupby(by=['ZONE']).sum(numeric_only=True).reset_index()
    avgs['AVG_FG_PCT'] = round(avgs['FGM'] / avgs['FGA'], 3)
    avgs = avgs.drop('FG_PCT', axis=1)

    merged = pd.merge(shots_group, avgs, on=['ZONE']).copy()
    merged = merged.rename(
        {'AVG_FG_PCT_x': 'PLAYER_PCT', 'AVG_FG_PCT_y': 'LEAGUE_PCT'}, axis=1
    ).copy()
    merged['PCT_DIFF'] = merged['PLAYER_PCT'] - merged['LEAGUE_PCT']

    to_plot = pd.merge(shots, merged, on=['ZONE'])[
        [
            'LOC_X',
            'LOC_Y',
            'SHOT_DISTANCE',
            'SHOT_TYPE',
            'ACTION_TYPE',
            'SHOT_MADE_FLAG_x',
            'ZONE',
            'PLAYER_PCT',
            'LEAGUE_PCT',
            'PCT_DIFF',
        ]
    ]
    to_plot['SHOT_TYPE'] = to_plot['SHOT_TYPE'].astype(str).str[0].astype(int)
    to_plot.columns = [
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
    ]

    to_plot['P_PPS'] = to_plot['PLAYER_PCT'] * to_plot['PTS']
    to_plot['L_PPS'] = to_plot['LEAGUE_PCT'] * to_plot['PTS']
    to_plot['D_PPS'] = to_plot['P_PPS'] - to_plot['L_PPS']

    return to_plot


def make_shot_chart(
    df,
    kind='normal',
    show_misses=True,
    title=None,
    title_size=22,
    context=None,
    context_size=14,
    show_pct=True,
    make_marker='o',
    miss_marker='x',
    make_marker_size=90,
    miss_marker_size=86,
    make_marker_color='#007A33',
    miss_marker_color='#C80A18',
    make_width=1,
    miss_width=3,
    hex_grid=50,
    scale_factor=5,
    min_factor=0,
    scale='P_PPS',
):
    """Render a shot chart figure.

    Args:
        df: DataFrame from shots_grouper() with required columns.
        kind: 'normal' for scatter plot or 'hex' for hexbin heatmap.
        show_misses: Whether to show missed shots.
        title: Title text for the figure.
        title_size: Font size for the title.
        context: Context text below the figure.
        context_size: Font size for the context text.
        show_pct: Whether to show shooting percentages.
        make_marker: Marker style for made shots (normal mode).
        miss_marker: Marker style for missed shots (normal mode).
        make_marker_size: Marker size for made shots.
        miss_marker_size: Marker size for missed shots.
        make_marker_color: Color for made shots.
        miss_marker_color: Color for missed shots.
        make_width: Line width for made shot markers.
        miss_width: Line width for missed shot markers.
        hex_grid: Number of hexes in each axis (hex mode).
        scale_factor: Max shot count for hex scaling.
        min_factor: Min shot count for hex scaling.
        scale: Color scale metric ('P_PPS', 'PCT_DIFF', or 'D_PPS').

    Returns:
        A matplotlib Figure.
    """
    fig, ax = make_shot_fig(title, title_size, context, context_size)

    df_t = df.copy()

    if scale == 'P_PPS':
        df_t['P_PPS'] = df_t['P_PPS'] / 3

    if show_pct:
        att_2 = len(df[(df['PTS'] == 2)])
        att_3 = len(df[(df['PTS'] == 3)])

        if att_2 != 0:
            made_2 = len(df[(df['PTS'] == 2) & (df['SHOT_MADE'] == 1)])
            if made_2 != 0:
                _2pt = round(round(made_2 / att_2, 4) * 100, 2)
            else:
                _2pt = 0
            _2_str = f'2pt%: {made_2}/{att_2} for {_2pt}%'

        if att_3 != 0:
            made_3 = len(df[(df['PTS'] == 3) & (df['SHOT_MADE'] == 1)])
            if made_3 != 0:
                _3pt = round(round(made_3 / att_3, 4) * 100, 2)
            else:
                _3pt = 0
            _3_str = f'3pt%: {made_3}/{att_3} for {_3pt}%'

        if kind == 'hex':
            txt_x = 245
            txt_b = 382.5
            txt_t = 370
            f_size = 12
        else:
            txt_x = 245
            txt_b = 417.5
            txt_t = 400
            f_size = 15

        if (att_2 == 0) and (att_3 == 0):
            pass
        elif (att_2 != 0) and (att_3 == 0):
            plt.text(
                txt_x,
                txt_b,
                _2_str,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=f_size,
            )
        elif (att_3 != 0) and (att_2 == 0):
            plt.text(
                txt_x,
                txt_b,
                _3_str,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=f_size,
            )
        else:
            plt.text(
                txt_x,
                txt_t,
                _2_str,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=f_size,
            )
            plt.text(
                txt_x,
                txt_b,
                _3_str,
                horizontalalignment='right',
                verticalalignment='bottom',
                fontsize=f_size,
            )

    if kind == 'normal':
        df_1 = df_t[df_t['SHOT_MADE'] == 1].copy()
        plt.scatter(
            df_1['X'],
            df_1['Y'],
            s=make_marker_size,
            marker=make_marker,
            c=make_marker_color,
            linewidth=make_width,
        )
        if show_misses:
            df_2 = df[df['SHOT_MADE'] == 0].copy()
            plt.scatter(
                df_2['X'],
                df_2['Y'],
                s=miss_marker_size,
                marker=miss_marker,
                c=miss_marker_color,
                linewidth=miss_width,
            )
    else:
        plt.text(
            196,
            414,
            'The larger hexagons\nrepresent a higher\ndensity of shots',
            horizontalalignment='center',
            bbox=dict(facecolor='#d9d9d9', boxstyle='round'),
        )

        if not show_misses:
            df_t = df_t[df_t['SHOT_MADE'] == 1].copy()
        hexbin = ax.hexbin(
            df_t['X'],
            df_t['Y'],
            C=df_t[scale].values,
            gridsize=hex_grid,
            edgecolors='black',
            cmap=matplotlib.colormaps['RdYlBu_r'],
            extent=[-275, 275, -50, 425],
            reduce_C_function=np.sum,
        )
        hexbin2 = ax.hexbin(
            df_t['X'],
            df_t['Y'],
            C=df_t[scale].values,
            gridsize=hex_grid,
            edgecolors='black',
            cmap=matplotlib.colormaps['RdYlBu_r'],
            extent=[-275, 275, -50, 425],
            reduce_C_function=np.mean,
        )

        axins1 = inset_axes(ax, width='16%', height='2%', loc='lower left')
        cbar = fig.colorbar(hexbin, cax=axins1, orientation='horizontal', ticks=[-1, 1])
        interval = hexbin.get_clim()[1] - hexbin.get_clim()[0]
        ltick = hexbin.get_clim()[0] + (interval * 0.2)
        rtick = hexbin.get_clim()[1] - (interval * 0.2)
        cbar.set_ticks([ltick, rtick])
        axins1.xaxis.set_ticks_position('top')
        if scale == 'PCT_DIFF':
            legend_text = '% Compared to \nLeague Average'
            tick_labels = ['Below', 'Above']
        elif scale == 'P_PPS':
            legend_text = 'Efficiency by Zone'
            tick_labels = ['Lower', 'Higher']
        else:
            legend_text = 'Efficiency compared to \nLeague Average'
            tick_labels = ['Lower', 'Higher']
        cbar.ax.set_title(legend_text, fontsize=10)
        cbar.set_ticklabels(tick_labels)

        offsets = hexbin.get_offsets()
        orgpath = hexbin.get_paths()[0]
        verts = orgpath.vertices
        values1 = hexbin.get_array()
        values1 = np.array(
            [
                scale_factor if i > scale_factor else 0 if i < min_factor else i
                for i in values1
            ]
        )
        # Guard against division by zero when scale_factor == 1
        if scale_factor != 1:
            values1 = ((values1 - 1.0) / (scale_factor - 1.0)) * (1.0 - 0.4) + 0.4
        else:
            values1 = np.where(values1 > 0, 1.0, 0.4)
        values2 = hexbin2.get_array()
        patches = []

        for offset, val in zip(offsets, values1, strict=False):
            v1 = verts * val + offset
            path = Path(v1, orgpath.codes)
            patch = PathPatch(path)
            patches.append(patch)

        pc = PatchCollection(
            patches, cmap=matplotlib.colormaps['RdYlBu_r'], edgecolors='black'
        )
        if scale == 'PCT_DIFF':
            if pc.get_clim()[0] is None:
                bottom = abs(df_t[scale].min())
                top = abs(df_t[scale].max())
            else:
                top = abs(pc.get_clim()[1])
                bottom = abs(pc.get_clim()[0])
            m = min(top, bottom)
            if m < 0.025:
                m = 0.025
            pc.set_clim([-1 * m, m])
        elif scale in ['P_PPS', 'L_PPS']:
            pc.set_clim([0.13333, 0.4])
        else:
            pc.set_clim([-0.05, 0.05])

        pc.set_array(values2)

        ax.add_collection(pc)
        hexbin.remove()
        hexbin2.remove()

    return fig
