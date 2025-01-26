#%%
import datetime
from typing import Any, Dict, List, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation, cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Arc, Circle, PathPatch, Rectangle
from matplotlib.path import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from nba_api.stats.endpoints import (commonplayerinfo, playercareerstats,
                                     playergamelog, shotchartdetail,
                                     shotchartlineupdetail)
from nba_api.stats.static import players, teams
from pandas import DataFrame

def get_team_id(abbrev: str) -> str:
    all_teams = pd.DataFrame(teams.get_teams())
    return all_teams[all_teams['abbreviation'] == abbrev]['id'].iloc[0]

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

class ShotChart:
    def __init__(self, player_name:str=None, seasons:List[int]=None, chart_params:Dict[str, Any]={}, **data_filters):
        self.player_name = player_name
        player_search = players.find_players_by_full_name(self.player_name)
        if len(player_search) == 0:
            raise PlayerNotFoundError('Name not found in database. Try being more specific or look for the player here: https://stats.nba.com/players/')
        self.player = player_search[0]
        self.player_id = self.player['id']
        self.chart_params = chart_params
        self.data_filters = data_filters
        self.seasons = seasons
        self._career = self.get_career()
        
    def get_career(self):
        df = playercareerstats.PlayerCareerStats(player_id=self.player_id, per_mode36='Totals').get_data_frames()[0]
        df['Player'] = self.player_name
        df['Season'] = df['SEASON_ID'].copy()
        df['Team'] = df['TEAM_ABBREVIATION'].copy()
        df['TS_PCT'] = round(df['PTS'] / (2*(df['FGA'] + (.44 * df['FTA']))),3)
        df = df[df['Team'] != 'TOT'][['Season', 'Team', 'TEAM_ID']].copy()
        df['start'] = df['Season'].apply(lambda x: int(x[:4]))
        df['end'] = df['start'] + 1
        df = df.rename({'TEAM_ID': 'Team ID', 'start': 'season'}, axis=1)[['Team ID', 'season']]

        return df
    
    def generate_chart(self):
        return self._generate_static_chart()

    def _generate_static_chart(self) -> Tuple[DataFrame, plt.Figure]:
        df = self._fetch_data(**self.data_filters)
        fig = self._make_shot_chart(df, **self.chart_params)
        
        return df, fig

    def _make_shot_chart(self, df, kind='normal', show_misses=True, 
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
    
    def _get_title(self, **data_filters) -> str:
        if 'date_to_nullable' in data_filters.keys():
            d_from = datetime.datetime.strptime(data_filters['date_from_nullable'], '%m-%d-%Y').strftime("%B %-d, %Y")
            d_to = datetime.datetime.strptime(data_filters['date_to_nullable'], '%m-%d-%Y').strftime("%B %-d, %Y")
            title += ' from ' + d_from + ' to ' + d_to
        else:
            if self.seasons is None:
                # get most recent season
                print('error')
                return
            if len(self.seasons) == 1:
                title += ' in the ' + str(str(self.seasons[0]) + "-" + str(self.seasons[0] + 1)[2:]) + ' season'
            elif self.seasons[1] - self.seasons[0] == 1:
                title += ' in the ' + str(str(self.seasons[0]) + "-" + str(self.seasons[0] + 1)[2:]) + ' and ' +str(str(self.seasons[1]) + "-" + str(self.seasons[1] + 1)[2:]) + ' seasons'
            else:
                title += ' from the ' + str(str(self.seasons[0]) + "-" + str(self.seasons[0] + 1)[2:]) + ' to ' +str(str(self.seasons[1]) + "-" + str(self.seasons[1] + 1)[2:]) + ' seasons'
        return title
    
    def _fetch_data(self, **data_filters) -> DataFrame:
        if 'opponent_team_id' in data_filters.keys():
            data_filters['opponent_team_id'] = get_team_id(data_filters['opponent_team_id'])
        
        if 'title' not in self.chart_params.keys():
            self.chart_params['title'] = self._get_title(**data_filters)

        log = shotchartdetail.ShotChartDetail(
            team_id=0
            , player_id=self.player_id
            , context_measure_simple=['FGA', 'FG3A']
            , **data_filters
        )
        
        shots = log.get_data_frames()[0]
        avgs = log.get_data_frames()[1]
        
        shots.reset_index(inplace=True, drop=True)
        avgs.reset_index(inplace=True, drop=True)

        if shots.empty:
            raise ValueError('No data found for the given data_filters: ' + str(data_filters))
        
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

if __name__ == '__main__':
    ShotChart(player_name='luka doncic', seasons=[2019], season_type_all_star='Regular Season', chart_params=dict(kind='hex', 
                scale_factor=4, hex_grid=35, 
                title=luka.name + '\n2019-20 All-NBA First Team Guard',
                context='In just his second season, Luka has taken the league by storm. He was named an All-Star\n ' \
                        "starter finished the regular season averaging 30.9 points. He continued this production\n" \
                        'in the bubble, leading the Mavericks to the playoffs and averaging 31 points per game while\n' \
                        'pushing the Clippers to 6 games.')).generate_chart()
    # ShotChart(
    #     player_name='luka doncic'
    #     , seasons=[2024]
    #     , SeasonType='Regular Season'
    #     , DateFrom='2024-12-21'
    #     , chart_params=dict(kind='hex', 
    #             scale_factor=4, hex_grid=35, 
    #             title=luka.name + '\n2024-25 end',
    #             context='regular season')).generate_chart()