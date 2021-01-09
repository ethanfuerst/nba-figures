import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Rectangle, Arc, PathPatch
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import html5lib
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import commonplayerinfo, playergamelog, playercareerstats, shotchartdetail, shotchartlineupdetail
from nba_season import NBA_Season


# - Custom errors
class PlayerNotFoundError(Exception):
    pass

class SeasonNotFoundError(Exception):
    pass


def thres_games(startyear=2000, endyear=datetime.datetime.today().year - 1, thres = 40):
    '''
    Returns a df detailing the number of games won by a number >= the threshold specified.

    Parameters:


    startyear (int, default: 2001)
        The first season that you want to be in the df.
            ex. 2015

    endyear (int, default: current year - 1)
        The last season that you want to be in the df.
            ex. 2019
    
    thres (int, default: 40)
        The margin of victory threshold for the df.
            ex. 30
    

    Returns:

    df
        A pd.DataFrame() containing the season data with the following columns:
            ['Season', 'Count', 'Game Nums', 'Projected']
            'Count' is the number of games over thres.
            'Projected' is for current seasons in play only.
    '''

    # - Need to make sure that startyear, endyear and thres are all integers
    try:
        startyear = int(startyear)
        endyear = int(endyear)
        thres = int(thres)
    except:
        # - This is probably because they inputted a string for season, and we need an int
        raise TypeError("Wrong variable type for startyear, endyear or thres. Integer expected.")
    # - Need to check that endyear season exists
    try:
        # - If there is data for endyear, we are good.
        Season(endyear)
    except:
        # - If we can't get data from endyear, then raise SeasonNotFoundError
        raise SeasonNotFoundError("There is no data for the " + str(endyear) + " season yet.")

    years = [i for i in range(startyear, endyear + 1)]
    tot = []
    for i in years:
        curr_season = NBA_Season(i)
        year = curr_season.get_season()
        num_games = len(year)
        season = "'" +str(i)[2:] + " - '" + str(i + 1)[2:]
        game_nums = list(year[year['MOV'] >= thres].index + 1)
        year = year[year['MOV'] >= thres].copy()
        count = len(year)
        Projected = int(((count / num_games) * 1230) - count)
        tot.append([season, count, game_nums, Projected])
    
    return pd.DataFrame(tot, columns=['Season', 'Count', 'Game Nums', 'Projected'])

def draw_court(color='black', lw=2):
    '''
    From http://savvastjortjoglou.com/nba-shot-sharts.html
    '''
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, linewidth=lw, color=color, fill=False)
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color, linestyle='dashed')
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, linewidth=lw, color=color)
    outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)

    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                        bottom_free_throw, restricted, corner_three_a,
                        corner_three_b, three_arc, center_outer_arc,
                        center_inner_arc, outer_lines]

    return court_elements

def zone_label(row):
    '''
    Creates zone for shots
    '''
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
        # 3's
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
    '''
    returns df ready for make_shot_chart from shots, avgs dfs
    '''
    # - Change zones
    shots['ZONE'] = shots.apply(lambda row: zone_label(row), axis=1)
    avgs['ZONE'] = avgs.apply(lambda row: zone_label(row), axis=1)

    shots_group = shots.groupby(by=['ZONE']).sum().reset_index()[['ZONE', 'SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG']].copy()
    shots_group['AVG_FG_PCT'] = round(shots_group['SHOT_MADE_FLAG'] / shots_group['SHOT_ATTEMPTED_FLAG'], 3)

    avgs = avgs.groupby(by=['ZONE']).sum().reset_index()
    avgs['AVG_FG_PCT'] = round(avgs['FGM'] / avgs['FGA'], 3)
    avgs = avgs.drop('FG_PCT', axis=1)

    merged = pd.merge(shots_group, avgs, on=['ZONE']).copy()
    merged = merged.rename({'AVG_FG_PCT_x': 'PLAYER_PCT', 'AVG_FG_PCT_y':'LEAGUE_PCT'}, axis=1).copy()
    merged['PCT_DIFF'] = merged['PLAYER_PCT'] - merged['LEAGUE_PCT']

    to_plot = pd.merge(shots, merged, on=['ZONE'])[['LOC_X', 'LOC_Y', 'SHOT_DISTANCE', 'SHOT_TYPE', 'ACTION_TYPE',
                                                'SHOT_MADE_FLAG_x', 'ZONE', 
                                                'PLAYER_PCT', 'LEAGUE_PCT', 'PCT_DIFF']]
    # - This SHOT_TYPE is how many points the attempted shot was for. Renamed to PTS
    to_plot['SHOT_TYPE'] = to_plot['SHOT_TYPE'].astype(str).str[0].astype(int)
    # * From here on SHOT_TYPE refers to ACTION_TYPE
    to_plot.columns = ['X', 'Y', 'SHOT_DISTANCE', 'PTS', 'SHOT_TYPE', 'SHOT_MADE', 'ZONE', 'PLAYER_PCT', 'LEAGUE_PCT', 'PCT_DIFF']
    
    to_plot['P_PPS'] = to_plot['PLAYER_PCT'] * to_plot['PTS']
    to_plot['L_PPS'] = to_plot['LEAGUE_PCT'] * to_plot['PTS']
    to_plot['D_PPS'] = to_plot['P_PPS'] - to_plot['L_PPS']

    return to_plot

def make_shot_chart(df, kind='normal', show_misses=True, 
                        title=None, title_size=22, 
                        context=None, context_size=14, show_pct=True,
                        make_marker='o', miss_marker= 'x', 
                        make_marker_size=90, miss_marker_size=86, 
                        make_marker_color='#007A33', miss_marker_color='#C80A18',
                        make_width=1, miss_width=3,
                        hex_grid=50, scale_factor=5, min_factor=0,
                        scale='P_PPS'
                        ):
    '''
    Returns a matplotlib fig of the player's shot chart given certain parameters.
    Will create the shot chart given a df created from the get_shot_chart method

    Parameters:
    
    kind (string, default: 'normal')
        'normal' or 'hex'
        Kind of shot chart
        'normal' - shows makes as dots
            Best for single game
        'hex' - shows frequency of shots in area as size of hex and color of zone compared to league average in zone
            Best for multiple games or players

    show_misses (boolean, default: True)
    
    title (string, default: None)
        The title on the top of the figure

    title_size (integer, default: 22)
        The title on the top of the figure
    
    context (string, default: None)
        Text on the bottom of the plot.
        Used to add context about a plot.
    
    context_size (integer, default: 14)
        context fontsize
    
    show_pct (boolean, default: True)
        Adds text in bottom right detailing 3pt% and 2pt%

    'normal' parameters:
        make_marker (string, default: 'o')
            Marker for the made shots

        miss_marker (string, default: 'x')
            Marker for missed shots

        make_marker_size (integer, default: 18)
            Marker size for made shots

        miss_marker_size (integer, default: 20)
            Marker size for missed shots

        make_marker_color (string, default: '#007A33' - green)
            Marker color for made shots

        miss_marker_color (string, default: '#C80A18' - red)
            Marker color for missed shots
        
        make_width (integer, default: 1)
            Width of marker for made shots

        miss_width (integer, default: 3)
            Width of marker for missed shots

    'hex' parameters:
        hex_grid (integer, default: 50)
            Number of hexes in the axis of each grid
            Larger number = smaller hexes
        
        scale_factor (integer, default: 5)
            Number of points in a hex to register as max size
            Usually between 4-6 works but it's a preference thing.

        min_factor (integer, default: 0)
            Number of points in a hex to register as min size
            Usually low, like 0-2
        
        scale (string, default: P_PPS)
            Must be one of 'PCT_DIFF', 'P_PPS', 'D_PPS'
            The value that the zones will be colored by

    Returns:

    fig
        fig of shot data
    '''
    # * add parameter to toggle scale factor
    # ? see if I can dynamically pull team logos to add to charts, maybe store them in a folder in this workspace
    fig, ax = make_shot_fig(title, title_size, context, context_size)
    
    df_t = df.copy()

    if scale == 'P_PPS':
        # - error if highest val is 1
        df_t['P_PPS'] = df_t['P_PPS']/3

    if show_pct:
        att_2 = len(df[(df['PTS'] == 2)])
        att_3 = len(df[(df['PTS'] == 3)])

        if att_2 != 0:
            made_2 = len(df[(df['PTS'] == 2) & (df['SHOT_MADE'] == 1)])
            if made_2 != 0:
                _2pt = round(round(made_2 / att_2, 4) * 100, 2)
            else:
                _2pt = 0
            _2_str = '2pt%: {0}/{1} for {2}%'.format(made_2, att_2, _2pt)
        
        if att_3 != 0:
            made_3 = len(df[(df['PTS'] == 3) & (df['SHOT_MADE'] == 1)])
            if made_3 != 0:
                _3pt = round(round(made_3 / att_3, 4) * 100, 2)
            else:
                _3pt = 0
            _3_str = '3pt%: {0}/{1} for {2}%'.format(made_3, att_3, _3pt)
        
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
            # - just 2pt%
            plt.text(txt_x, txt_b, _2_str, horizontalalignment='right', verticalalignment='bottom', fontsize=f_size)
        elif (att_3 != 0) and (att_2 == 0):
            # - just 3pt%
            plt.text(txt_x, txt_b, _3_str, horizontalalignment='right', verticalalignment='bottom', fontsize=f_size)
        else:
            # - both 2 and 3pt%
            plt.text(txt_x, txt_t, _2_str, horizontalalignment='right', verticalalignment='bottom', fontsize=f_size)
            plt.text(txt_x, txt_b, _3_str, horizontalalignment='right', verticalalignment='bottom', fontsize=f_size)
    
    if kind == 'normal':
        df_1 = df_t[df_t['SHOT_MADE'] == 1].copy()
        plt.scatter(df_1['X'], df_1['Y'], s=make_marker_size, marker=make_marker, c=make_marker_color, linewidth=make_width)
        if show_misses:
            df_2 = df[df['SHOT_MADE'] == 0].copy()
            # - linewidths increase
            plt.scatter(df_2['X'], df_2['Y'], s=miss_marker_size, marker=miss_marker, c=miss_marker_color, linewidth=miss_width)
    else:
        plt.text(196, 414, 'The larger hexagons\nrepresent a higher\ndensity of shots',
                    horizontalalignment='center', bbox=dict(facecolor='#d9d9d9', boxstyle='round'))
        
        if not show_misses:
            df_t = df_t[df_t['SHOT_MADE'] == 1].copy()
        hexbin = ax.hexbin(df_t['X'], df_t['Y'], C=df_t[scale].values
            , gridsize=hex_grid, edgecolors='black',cmap=cm.get_cmap('RdYlBu_r'), extent=[-275, 275, -50, 425]
            , reduce_C_function=np.sum)
        # - color
        hexbin2 = ax.hexbin(df_t['X'], df_t['Y'], C=df_t[scale].values, gridsize=hex_grid, edgecolors='black',
            cmap=cm.get_cmap('RdYlBu_r'), extent=[-275, 275, -50, 425], reduce_C_function=np.mean)
        
        axins1 = inset_axes(ax, width="16%", height="2%", loc='lower left')
        cbar = fig.colorbar(hexbin, cax=axins1, orientation="horizontal", ticks=[-1, 1])
        interval = hexbin.get_clim()[1] - hexbin.get_clim()[0]
        ltick = hexbin.get_clim()[0] + (interval * .2)
        rtick = hexbin.get_clim()[1] - (interval * .2)
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
        # - scale factor - usually 4 or 5 works
        values1 = np.array([scale_factor if i > scale_factor else 0 if i < min_factor else i for i in values1])
        values1 = ((values1 - 1.0)/(scale_factor-1.0))*(1.0-.4) + .4
        values2 = hexbin2.get_array()
        patches = []

        for offset,val in zip(offsets,values1):
            v1 =  verts*val + offset
            path = Path(v1, orgpath.codes)
            patch = PathPatch(path)
            patches.append(patch)

        pc = PatchCollection(patches, cmap=cm.get_cmap('RdYlBu_r'), edgecolors='black')
        if scale == 'PCT_DIFF':
            if pc.get_clim()[0] is None:
                bottom = abs(df_t[scale].min())
                top = abs(df_t[scale].max())
            else:
                top = abs(pc.get_clim()[1])
                bottom = abs(pc.get_clim()[0])
            m = min(top, bottom)
            # - Need one extreme of the comparison to be at least 1.5 percent off from average
            if m < .025:
                m = .025
            pc.set_clim([-1 * m, m])
        # - pps is .4 to 1.something 
        elif scale in ['P_PPS', 'L_PPS']:
            # - for 2: 20% to 60%
            # - for 3: 13% to 40%
            pc.set_clim([0.13333, .4])
        else:
            pc.set_clim([-.05,.05])
        
        pc.set_array(values2)

        ax.add_collection(pc)
        hexbin.remove()
        hexbin2.remove()

    return fig

def make_shot_fig(title, title_size, context, context_size):
    background_color = '#d9d9d9'
    fig, ax = plt.subplots(facecolor=background_color, figsize=(10,10))
    fig.patch.set_facecolor(background_color)
    ax.patch.set_facecolor(background_color)

    court_elements = draw_court()
    for element in court_elements:
        ax.add_patch(element)
    
    if title is not None:
        plt.title(title, pad=10, fontdict={'fontsize': title_size, 'fontweight':'semibold'})

    img = plt.imread("basketball-floor-texture.png")
    plt.imshow(img,zorder=0, extent=[-275, 275, -50, 425])

    plt.xlim(-250,250)
    plt.ylim(422.5, -47.5)
    plt.axis(False)

    if context is not None:
        # - If multiple lines then add context size to second variable for each additional line
        ax.text(0, 435 + (context_size * context.count('\n')), s=context, fontsize=context_size, ha='center')

    return fig, ax

def get_team_id(abbrev):
    '''Returns team_id when given an abbreviation'''
    return pd.DataFrame(teams.get_teams())[pd.DataFrame(teams.get_teams())['abbreviation'] == abbrev]['id'].iloc[0]
        


#%%
def make_shot_dist(df, title=None, title_size=22, 
                        context=None, context_size=14,
                        color=None):
    # todo add bar color - one val or the different shooting %ages
    # todo stacked historgram for makes and misses

    background_color = '#d9d9d9'
    fig, ax = plt.subplots(facecolor=background_color, figsize=(10,7))
    fig.patch.set_facecolor(background_color)
    ax.patch.set_facecolor(background_color)

    if title is not None:
        plt.title(title, pad=10, fontdict={'fontsize': title_size, 'fontweight':'semibold'})
    plt.ylabel('Frequency')
    plt.xlabel('Shot Distance')

    bins = df['SHOT_DISTANCE'].max()

    # - if color is none then set to a color
    if color == 'make_miss':
        # - stacked bar chart with makes and misses
        plt.hist([df[df['SHOT_MADE'] == 1]['SHOT_DISTANCE'], df[df['SHOT_MADE'] == 0]['SHOT_DISTANCE']], 
                color=["#007A33", "#C80A18"], bins=bins, stacked=True)
    elif color == 'shot_type':
        # - stacked bar chart by shot type
        print('')
    elif color == 'PCT_DIFF':
        # - change color to be scaled by this column
        print('')
    elif color == 'P_PPS':
        # - change color to be scaled by this column
        print('')
    elif color == 'D_PPS':
        # - change color to be scaled by this column
        print('')
    elif color == 'team':
        # - color by team that player played for when the shot was taken
        print('')
    else:
        # - set a color manually
        plt.hist(df['SHOT_DISTANCE'], bins=df['SHOT_DISTANCE'].max())

    return fig



# %%
