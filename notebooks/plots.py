import os
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import circvar
import numpy as np
from matplotlib.lines import Line2D

from trial import Trial
from constants import FS, COND_LABELS, DV_LABELS, DV_COLORS, ROLLING_DVS, COUNTER_LABELS
from analysis_helper import match_trials

sns.set_style('dark', {'axes.grid' : False, 'dpi': 144})


def plot_player_moves_through_match():
    df_trials.groupby('match_nick')[['p0_n_moves', 'p1_n_moves']].mean()
    fig, axs = plt.subplots(1, 2, dpi=300)
    sns.lineplot(x='trial', y='p0_n_moves', hue='match_nick', data=df_trials, ax=axs[0])
    sns.lineplot(x='trial', y='p1_n_moves', hue='match_nick', data=df_trials, ax=axs[1])
    axs[0].set_title("P0")
    axs[1].set_title("P1")    
    plt.suptitle("Moves per trial")


def condition_agg_match_timeseries(df_trials, dvs=None, 
                                   title="Cross-trial trends by counter condition",
                                   force_ylim=None):
    fig, axs = plt.subplots(1, 2, dpi=200)
    lpkwargs = {'marker': 'o', 'markersize': 2, 'alpha': 0.6}    
    
    for i, first_counter in enumerate("CG"):
        df = df_trials[df_trials.first_counter_block == first_counter]
        ax = axs[i]
        for dv in dvs:
            sns.lineplot(data=df, x='trial', y=dv['var'], 
                    label=dv['label'], 
                    color=dv['color'], ax=ax, **lpkwargs)
        ax.set_ylabel("")
        if force_ylim is not None:
            ax.set_ylim((0.0, force_ylim))
        y_min, y_max = ax.get_ylim()
        ax.plot([39, 39], ax.get_ylim(), color="gray", dashes=[2, 4])
        # Which block starts in middle
        counter_label = {
            'C': "Gray",
            'G': "Color"
        }[first_counter]
        ax.text(39, y_min, "%s block starts" % counter_label, 
                ha="center", fontsize='small', color='gray')
        ax.set_title("First counter: %s" % first_counter)
    
    plt.suptitle(title)
    plt.savefig("./out/%s.png" % title)
    plt.show()
    
def aggregate_timeseries(df_trials, rolling_window_k=5,
                        dvs = ["goal_color_split", "successful", "total_moves", "goal_pct_diff"],
                         one_group=None,
                        fix_ylim=None, title="Rolling average metrics across trials, by group"):
    """
    Timeseries with moving average across trials, of 4 dvs
    One plot for each group
    """
    nrow, ncol = (2, 2) if not one_group else (1, 2)
    if not one_group:
        plots = [("C", "C"), ("C", "G"), ("G", "G"), ("G", "C")] 
        figsize = (6, 6)
    else:
        plots = [(one_group, one_group), (one_group, "C" if one_group == "G" else "C")]
        figsize = (6, 3)
    fig, axs = plt.subplots(nrow, ncol, dpi=300, figsize=figsize, sharey=True)
    
    dv_scale = {
        "total_moves": 1/df_trials["total_moves"].max(),
        "mean_dist": 1/df_trials["mean_dist"].max(),
        "duration": 1/df_trials["duration"].max(),        
        "goal_pct_diff": 1/df_trials["goal_pct_diff"].max(),        
        "fairness": 1/df_trials["fairness"].max(),        
        "spatial_cons": 1/df_trials["spatial_cons"].max(),        
    }
    i = 0
    plots = [("C", "C"), ("C", "G"), ("G", "G"), ("G", "C")]
    for ax, (group, counter) in zip(fig.axes, plots):
        df = df_trials[(df_trials.first_counter_block == group) & (df_trials.cond_count == counter)]
        match_nicks = df.match_nick.unique()
        t_first, t_last = int(df.trial.min()), int(df.trial.max())
        avg_timeseries = {}  # dv -> list of lists, one per match
        for match_nick in match_nicks:
            df_match = df[df.match_nick == match_nick]
            incomplete = len(df_match) < 40
            # TODO: Keep match with missing trials?
            if incomplete:
                continue
            for dv in dvs:
                if dv not in avg_timeseries:
                    avg_timeseries[dv] = []
                scale = dv_scale.get(dv, 1.)
                # Handle spatial_cons as edge case, since it's already a rolling window
                if dv in ROLLING_DVS:
                    match_dv_ave = list(scale * df_match[dv].values)
                else:
                    rolling_vals = df_match.rolling(rolling_window_k, min_periods=1)[dv]
                    match_dv_ave = list(scale * rolling_vals.mean().values)
                avg_timeseries[dv].append(match_dv_ave)
        for dv, match_timeseries in avg_timeseries.items():
            stacked = np.vstack(match_timeseries)
            dv_timeseries = stacked.mean(axis=0)
            dv_timeseries_std = stacked.std(axis=0)
            xaxis = range(t_first, t_last+1)
            color = DV_COLORS.get(dv)
            ax.plot(xaxis, dv_timeseries, 
                    label=DV_LABELS.get(dv),
                    color=color)
            ax.fill_between(xaxis, dv_timeseries - dv_timeseries_std, 
                                   dv_timeseries + dv_timeseries_std, 
                            color=color,
                            alpha=0.1)
        if i == 0:
            ax.legend()
        title_color = 'purple' if counter == 'C' else 'gray'
        row = i // 4
        col = i % 2
        title_color = 'purple' if counter == 'C' else 'gray'
        ax.set_title("Block %d (%s)" % (col+1, COUNTER_LABELS[counter]), color=title_color)
        if fix_ylim is not None:
            ax.set_ylim(fix_ylim)
        ylim = ax.get_ylim()
        # ax.plot([40, 40], ylim, color='gray', dashes=[2, 2])
        i += 1
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
        


def single_timeseries_with_traces(df_trials, rolling_window_k=5,
                        dv = "goal_color_split", figsize=(7, 4),
                        fix_ylims=None, one_group=None, yticks=None,
                        do_scale=True, title_fs=10, figlabel=None, save=None):
    """
    Timeseries with moving average for individual dyads as traces
    One plot for each group
    """
    nrow, ncol = (2, 2)
    plots = [("C", "C"), ("C", "G"), ("G", "G"), ("G", "C")] 
    fig, axs = plt.subplots(nrow, ncol, dpi=300, figsize=figsize, sharey='row')
    
    dv_scale = {
        "total_moves": 1/df_trials["total_moves"].max(),
        "mean_dist": 1/df_trials["mean_dist"].max(),
        "duration": 1/60.,
        # "goal_pct_diff": 1/df_trials["goal_pct_diff"].max(),        
        "spatial_cons": 1/df_trials["spatial_cons"].max(),        
    }
    for i, ax, (group, counter) in zip(range(len(plots)), fig.axes, plots):
        row = i // 4
        col = i % 2
        df = df_trials[(df_trials.first_counter_block == group) & (df_trials.cond_count == counter)]
        match_nicks = df.match_nick.unique()
        t_first, t_last = int(df.trial.min()), int(df.trial.max())
        avg_timeseries = {}  # dv -> list of lists, one per match
        for match_nick in match_nicks:
            # Compute rolling average for each dyad
            df_match = df[df.match_nick == match_nick]
            incomplete = len(df_match) < 40
            # TODO: Keep match with missing trials?
            if incomplete:
                continue
            if dv not in avg_timeseries:
                avg_timeseries[dv] = []
            scale = dv_scale.get(dv, 1.) if do_scale else 1.
            if dv in ROLLING_DVS:
                # This DV already aggregated (std) across multiple trials
                match_dv_ave = list(scale * df_match[dv].values)
            else:
                rolling_vals = df_match.rolling(rolling_window_k, min_periods=1)[dv]
                match_dv_ave = list(scale * rolling_vals.mean().values)
            avg_timeseries[dv].append(match_dv_ave)
        for dv, match_timeseries in avg_timeseries.items():
            xaxis = range(t_first+1, t_last+2)
            color = DV_COLORS.get(dv)
            for j, trace in enumerate(match_timeseries):
                # Traces
                ax.plot(xaxis, trace, 
                        label=DV_LABELS.get(dv) if j == 0 else None,
                        color=color, lw=0.3, alpha=0.5)

            # Plot mean thicker
            stacked = np.vstack(match_timeseries)
            mean_timeseries = stacked.mean(axis=0)
            ax.plot(xaxis, mean_timeseries, color=color, lw=1.0)

        title_color = 'purple' if counter == 'C' else 'gray'
        ax.set_title("Block %d (%s)" % (col+1, COUNTER_LABELS[counter]), fontsize=title_fs, color=title_color)
        
        if fix_ylims is not None:
            ax.set_ylim(fix_ylims)
        if yticks is not None:
            ax.set_yticks(yticks)
            ax.set_yticklabels("%.2f" % x for x in yticks)
            
        ylim = ax.get_ylim()

    group_fs = 11
    plt.gcf().text(0.0, 0.11, "HMC-first group", ha='center', rotation="vertical", fontsize=group_fs)
    plt.gcf().text(0.0, 0.55, "LMC-first group", ha='center', rotation="vertical", fontsize=group_fs)
    suptitle = DV_LABELS[dv]
    if figlabel:
        suptitle = "%s) %s" % (figlabel, suptitle)
    plt.suptitle(suptitle, fontsize=14)
    plt.subplots_adjust(left=0.3)
    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches='tight')
    plt.show()

def timeseries_with_traces(df_trials, rolling_window_k=5,
                        dvs = ["goal_color_split", "successful", "total_moves", "goal_pct_diff"],
                        fix_ylims=None, one_group=None, 
                        do_scale=True,
                        title="Dyad strategy traces across trials, by group"):
    """
    Timeseries with moving average for individual dyads as traces
    One plot for each group
    """
    nrow, ncol = (len(dvs), 4) if not one_group else (len(dvs), 2)
    if not one_group:
        plots = [("C", "C"), ("C", "G"), ("G", "G"), ("G", "C")] 
    else:
        plots = [(one_group, one_group), (one_group, "C" if one_group == "G" else "C")]
    plots *= len(dvs)
    figheight = nrow * 1.75
    fig, axs = plt.subplots(nrow, ncol, dpi=300, figsize=(8, figheight), sharey='row')
    
    dv_scale = {
        "total_moves": 1/df_trials["total_moves"].max(),
        "mean_dist": 1/df_trials["mean_dist"].max(),
        "duration": 1/60.,
        "goal_pct_diff": 1/df_trials["goal_pct_diff"].max(),        
        "spatial_cons": 1/df_trials["spatial_cons"].max(),        
    }
    for i, ax, (group, counter) in zip(range(len(plots)), fig.axes, plots):
        row = i // 4
        dv = dvs[row]
        df = df_trials[(df_trials.first_counter_block == group) & (df_trials.cond_count == counter)]
        match_nicks = df.match_nick.unique()
        t_first, t_last = int(df.trial.min()), int(df.trial.max())
        avg_timeseries = {}  # dv -> list of lists, one per match
        for match_nick in match_nicks:
            # Compute rolling average for each dyad
            df_match = df[df.match_nick == match_nick]
            incomplete = len(df_match) < 40
            # TODO: Keep match with missing trials?
            if incomplete:
                continue
            if dv not in avg_timeseries:
                avg_timeseries[dv] = []
            scale = dv_scale.get(dv, 1.) if do_scale else 1.
            if dv in ROLLING_DVS:
                # This DV already aggregated (std) across multiple trials
                match_dv_ave = list(scale * df_match[dv].values)
            else:
                rolling_vals = df_match.rolling(rolling_window_k, min_periods=1)[dv]
                match_dv_ave = list(scale * rolling_vals.mean().values)
            avg_timeseries[dv].append(match_dv_ave)
        for dv, match_timeseries in avg_timeseries.items():
            xaxis = range(t_first, t_last+1)
            color = DV_COLORS.get(dv)
            for j, trace in enumerate(match_timeseries):
                # Traces
                ax.plot(xaxis, trace, 
                        label=DV_LABELS.get(dv) if j == 0 else None,
                        color=color, lw=0.3, alpha=0.5)

            # Plot mean thicker
            stacked = np.vstack(match_timeseries)
            mean_timeseries = stacked.mean(axis=0)
            ax.plot(xaxis, mean_timeseries, color=color, lw=1.0)

        if i % 4 == 0:
            ax.legend()
        title_color = 'purple' if counter == 'C' else 'gray'
        ax.set_title("%s first, %s counter" % (COUNTER_LABELS[group], COUNTER_LABELS[counter]), fontsize=8, color=title_color)
        
        if fix_ylims is not None:
            ax.set_ylim(fix_ylims[row] if row < len(fix_ylims) else (0, 1.))
        ylim = ax.get_ylim()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
def match_timeseries(df_match, split_row_attr=None, split_row_values=None):
    match_id = df_match.iloc[0]['match_id']
    match_nick = df_match.iloc[0]['match_nick']
    if split_row_attr:
        ncols = len(split_row_values)
        views = [df_match[df_match[split_row_attr] == val] for val in split_row_values]
        row_labels = ["%s:%s" % (split_row_attr, v) for v in split_row_values]
    else:
        ncols = 1
        views = [df_match]
        row_labels = [""]
    
    fig, axs = plt.subplots(2, ncols, dpi=200, figsize=(7, 2.5 * ncols))
    
    for i, (label, df_view) in enumerate(zip(row_labels, views)):
        lpkwargs = {'marker': 'o', 'linestyle': 'None', 'markersize': 2, 'alpha': 0.6}
        unsuccessful_trials = df_view[df_view.successful == 0].trial.values

        ax = axs[0][i]
        ax.plot(df_view['trial'], df_view['goal_color_split'], label="Color Polarization", color='purple', **lpkwargs)
        ax.plot(df_view['trial'], df_view['goal_pct_diff'], label="Work Imbalance (difference of % of goals taken)", color='orange', **lpkwargs)
#         sns.lineplot(x='trial', y='goal_color_split', label="Color Polarization", color='purple', ax=ax, **lpkwargs)
#         sns.lineplot(x='trial', y='goal_pct_diff', label="Goal Split (difference of % of goals taken)", ax=ax, **lpkwargs)
        ax.set_ylabel("Percent")
        title = "Goal & Color Polarization Across Trials"
        if label:
            title += " (%s)" % label
        ax.set_title(title, fontsize=FS)
        ax.set_ylim((-0.05, 1.05))
        ax.legend(fontsize=6)
        for ut in unsuccessful_trials:
            ax.plot([ut, ut], [0., 1.], dashes=[2, 2], lw=0.5, color='red')

        ax = axs[1][i]
        ax.plot(df_view['trial'], df_view['p0_c0_goal_pct'], label="Color Assignment (P0 % of all reds taken)", color='red', **lpkwargs)
        ax.plot(df_view['trial'], df_view['p0_c1_goal_pct'], label="Color Assignment (P0 % of all blues taken)", color='blue', **lpkwargs)
        title = "Color Assignment Across Trials"
        if label:
            title += " (%s)" % label
        ax.set_title(title, fontsize=FS)
        ax.set_ylim((-0.05, 1.05))
        ax.legend(fontsize=6)
        for ut in unsuccessful_trials:
            ax.plot([ut, ut], [0., 1.], dashes=[2, 2], lw=0.5, color='red')
    plt.suptitle(match_nick, fontsize=10)
    plt.tight_layout()
    plt.savefig("./out/%s/color_goal_split_across_trials.png" % match_id)
    
def move_time_correlations(df_match, match_id):
    df_match = df_match[df_match.p0_first_move_time + df_match.p1_first_move_time < 30]
    fig, ax = plt.subplots(dpi=144)
    p0_first = df_match.query("p0_first_move_time < p1_first_move_time")
    p1_first = df_match.query("p1_first_move_time < p0_first_move_time")
    sns.scatterplot(data=p0_first, x='p0_first_move_time', y='p1_first_move_time', color='green', ax=ax)
    sns.scatterplot(data=p1_first, x='p0_first_move_time', y='p1_first_move_time', color='magenta', ax=ax)
    slope, intercept, r_value, p_value, std_err = stats.linregress(df_match['p0_first_move_time'], df_match['p1_first_move_time'])
    xmax = df_match['p0_first_move_time'].max()
    fit_axis = np.array([0, xmax])
    text_x = xmax * 0.8
    text_y = intercept + slope * text_x
    ax.text(text_x * 1.2, text_y, "p=%.3f, r=%.3f" % (p_value, r_value), c='blue')
    ax.plot(fit_axis, intercept + slope * fit_axis, color='blue')
    ax.plot([0, 5], [0, 5], dashes=[2, 2], color='gray')
    ax.set_xlim((0, xmax))
    ax.set_title("Correlation of player first move times (seconds)")
    plt.savefig("./out/%s/corr_first_move_times.png" % match_id)    
    
    
def second_mover_delay(df_trials):
    df_trials['second_mover_delay'] = df_trials['second_mover_time'] - df_trials['first_mover_time']
    df_trials['p0_p1_first_move_time_delta'] = df_trials['p1_first_move_time'] - df_trials['p0_first_move_time']

    for match_id in df_trials.match_id.unique():
        df_match = match_trials(df_trials, match_id=match_id)
        df_match = df_match[df_match.p0_first_move_time + df_match.p1_first_move_time < 100]

        fig, ax = plt.subplots(dpi=144)
        sns.lineplot(data=df_match, x='trial', y='p0_p1_first_move_time_delta')
        ax.plot([0, 80], [0, 0], dashes=[2, 2], color='gray')
        ax.set_title("Delay between players' first moves (positive when P0 moves first)")
        ax.set_ylabel("Delay (p1 after p0, seconds)")
        plt.savefig("./out/%s/player_delay_timeseries.png" % match_id)
        plt.show()    
        
def scatter_angle_over_trials(df_trials, max_dyads=5, match_nick_order=None):
    if match_nick_order:
        mids = match_nick_order
    else:
        mids = df_trials.match_nick.unique()[:max_dyads]
    fig, axs = plt.subplots(len(mids), 1, figsize=(5, 5))
    for ax, match_nick in zip(axs, mids):
        match_data = df_trials[df_trials.match_nick == match_nick]
        ax.scatter(match_data.trial, match_data['theta_mean_p0'], color='green', s=1)
        ax.scatter(match_data.trial, match_data['theta_mean_p1'], color='magenta', s=1)
        ax.set_title(match_nick, fontsize=FS)
        ax.set_ylabel("Theta (rads)", fontsize=FS)
        ax.set_ylim((0, 2*np.pi))
        ax.grid(False)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Mean angle from middle, both players, across trials", fontsize=11)
    plt.show()    
    
def coords_over_trials_plot(df_trials, coord='x'):
    fig, axs = plt.subplots(4, 1, figsize=(5, 5))
    for ax, match_id in zip(axs, df_trials.match_id.unique()):
        match_data = df_trials[df_trials.match_id == match_id]
        ax.scatter(match_data.trial, match_data['%s_mean_p0' % coord], color='green', s=1)
        ax.scatter(match_data.trial, match_data['%s_mean_p1' % coord], color='magenta', s=1)
        ax.set_title(match_id, fontsize=FS)
        ax.set_ylabel("Coordinate", fontsize=FS)
        ax.set_ylim((-5, 5))
        ax.grid(False)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Mean %s-Coordinate (rel middle), both players, across trials" % coord, fontsize=11)
    plt.show()
    
    
def coords_over_trials_boxplots(df_trials, coord='x'):
    fig, axs = plt.subplots(4, 1, figsize=(6, 7))
    trial_axis = list(range(80))
    for ax, match_id in zip(axs, df_trials.match_nick.unique()):
        match_data = df_trials[df_trials.match_nick == match_id].copy()
        match_data['trial_bin'] = (match_data.trial / 10).astype(int)
        match_data['%s_mean' % coord] = match_data['%s_mean_p0' % coord]        
        match_data['player'] = 0
        match_data_p1 = match_data.copy()
        match_data_p1['player'] = 1
        match_data_p1['%s_mean' % coord] = match_data_p1['%s_mean_p1' % coord]
        match_data = match_data.append(match_data_p1, ignore_index=True)
        g = sns.boxplot(data=match_data, x='trial_bin', y='%s_mean' % coord, 
                        hue='player', palette=['green', 'magenta'], linewidth=0.5, ax=ax)
        ax.legend([],[], frameon=False)
        ax.set_title(match_id, fontsize=FS)
        ax.set_ylabel("%s-Coordinate" % coord, fontsize=FS)
        ax.set_ylim((-5, 5))
        ax.set_xlabel("Binned Trials", fontsize=FS)
        ax.grid(False)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.suptitle("Mean %s-Coordinate (rel middle), both players, across 10-trial bins" % coord, fontsize=11)
    plt.show()
    
def p2p_dist_plot(df_distance, kind='x', goals=8): 
    # Kind in ['mag', 'x', 'y']
    wide = kind in 'xy'
    fig, axs = plt.subplots(1, 4, dpi=300, figsize=(8, 3))
    color = {
        'mag': "#34a1eb",
        'x': "#26e058",
        'y': "#e0268f"
    }[kind]
    ymax = {
        'mag': 8, 
        'x': 5,
        'y': 5
    }[kind]
    dist_key = 'p2p_dist_%s' % kind if kind != 'mag' else 'p2p_dist'
    i = 0
    LEVELS = [
        "g%dbEcC" % goals,
        "g%dbEcG" % goals,
        "g%dbFBcC" % goals,
        "g%dbFBcG" % goals,
    ]
    mean_key = 'mean_g%d' % goals
    for level in LEVELS:
        ax = fig.axes[i]
        subplot_data = df_distance[(df_distance.level.isin([level, mean_key])) & (df_distance.cond_goals == goals)]
        sns.lineplot(data=subplot_data, x='time_pct_dig', y=dist_key, hue='level', 
                     hue_order=[level, mean_key], palette=[color, "#777"], alpha=0.9, legend=False, ax=ax)
        ax.set_title(level, fontsize=FS)
        ax.set_xlabel("")
        ax.set_ylabel("P2P Dist (%s)" % kind, fontsize=FS)
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)    
        ax.set_ylim((0, ymax))
        i += 1
    plt.suptitle("P2P distance (%s) by condition level, through trial (G=%d)" % (kind, goals))
    plt.tight_layout()
    plt.savefig('./out/p2p_dist_%s_all_levels.png' % kind)

def plot_main_effect_f20(df_trials, dv):
    df_trials_f20 = df_trials[df_trials.trial <= 19]
    N = len(df_trials.match_id.unique())
    fig, ax = plt.subplots(2, 3, dpi=144, figsize=(8, 4))
    for row in range(2):
        palette = ['black', '#777', '#AAA'] if row else ['#024a1a', '#088532', '#64a178']
        for i, iv in enumerate(['cond_goals', 'cond_bal', 'cond_count']):
            ax = fig.axes[row*3 + i]
            first_20 = row == 0
            df = df_trials_f20 if first_20 else df_trials
            order = None
            if iv == "cond_bal":
                order = ["E", "FB"]
            sns.barplot(data=df, x=iv, y=dv, order=order, palette=palette, ax=ax)
            ax.set_xlabel(COND_LABELS[iv])
            title = "DV: %s" % dv
            if first_20:
                title += " (first 20 trials)"
            ax.set_title(title, fontsize=10)
    ymax = max([ax.get_ylim()[1] for ax in fig.axes])
    ymin = max([ax.get_ylim()[0] for ax in fig.axes])    
    for ax in fig.axes:
        ax.set_ylim((ymin, ymax))
    ax.set_ylim((ymin, ymax))
    plt.tight_layout()
    plt.savefig("./out/%s_by_condition.png" % (dv))
    plt.show()    

def plot_main_effect(df_trials, dv):
    N = len(df_trials.match_id.unique())
    fig, ax = plt.subplots(1, 4, dpi=144, figsize=(8, 4))
    palette = ['black', '#777', '#AAA']
    for i, iv in enumerate(['cond_goals', 'cond_bal', 'cond_count', 'first_counter_block']):
        ax = fig.axes[i]
        order = None
        if iv == "cond_bal":
            order = ["E", "FB"]
        elif iv == "first_counter_block":
            order = ["C", "G"]
        elif iv == "cond_count":
            order = ["C", "G"]
        sns.barplot(data=df_trials, x=iv, y=dv, order=order, palette=palette, ax=ax)
        ax.set_xlabel(COND_LABELS[iv])
        ax.set_ylabel(DV_LABELS.get(dv, dv))
        title = "DV: %s" % DV_LABELS.get(dv, dv)
        ax.set_title(title, fontsize=10)
    ymax = max([ax.get_ylim()[1] for ax in fig.axes])
    ymin = max([ax.get_ylim()[0] for ax in fig.axes])    
    for ax in fig.axes:
        ax.set_ylim((ymin, ymax))
    ax.set_ylim((ymin, ymax))
    plt.tight_layout()
    plt.savefig("./out/%s_by_condition.png" % (dv))
    plt.show()    
    
def plot_first_move_arrows():
    dirs = ['up', 'right', 'down', 'left']
    units = [[0, 1], [1, 0], [0, -1], [-1, 0]]

    for match_id in df_trials.match_id.unique():
        arrows = [[0, 0, 0, 0] for d in dirs]
        df_match = match_trials(match_id)
        for key, trial_row in df_match.iterrows():
            fm_player = trial_row.first_move_player
            for pid in [0, 1]:
                is_first_mover = fm_player == pid
                player_dir = trial_row['p%d_first_move_dir' % pid]
                idx = 2*pid
                if player_dir:
                    dir_idx = dirs.index(player_dir)
                    arrows[dir_idx][idx] += 1
                    if is_first_mover:
                        arrows[dir_idx][idx + 1] += 1
        print(arrows)

        # TODO: Show total direction & first move (maybe inset), not independentlly

        gap = 2.5
        center_r = 7.
        fig, ax = plt.subplots(dpi=144, figsize=(5, 5))
        colors = ["#29ba3744", "#29ba37ff", "#fc38ff44", "#fc38ffff"]
        for i, arrow in enumerate(arrows):
            unit = np.array(units[i])
            for j, (count, color) in enumerate(zip(arrow, colors)):
                vec = unit * count
                origin = unit * center_r
                offset = -gap if j < 2 else gap
                if unit[0] == 0:
                    origin[0] += offset
                elif unit[1] == 0:
                    origin[1] += offset
                fa = FancyArrow(origin[0], origin[1], vec[0], vec[1], width=1., color=color)
                ax.add_patch(fa)
        ax.add_patch(Ellipse((0, 0), center_r * 1.4, center_r * 1.4, lw=2, fill=False, color='k'))
        ax.add_patch(Ellipse((0, 0), center_r, center_r, fill=True, color='k'))

        # Legend
        custom_lines = [Line2D([0], [0], color=c, lw=4) for c in colors]
        ax.legend(custom_lines, ['Player 0', 'Player 0 (as first mover)', 'Player 1', 'Player 1 (as first mover)'], fontsize=7)

        S = np.array(arrows).max() * 1.5
        ax.set_xlim((-S, S))
        ax.set_ylim((-S, S))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.suptitle("Count of First Move Direction")
        plt.savefig("./out/%s/first_move_direction.png" % match_id)
        plt.show()    
        
def trial_duration(df_trials):
    g = sns.displot(df_trials, x='duration', hue='match_nick', kind='kde', rug=True, multiple='layer')
    plt.title("Trial Duration Distribution (seconds)")
    plt.savefig("./out/trial_dur_dist.png")
    plt.show()

def trial_success(df_trials):
    g = sns.histplot(df_trials, x='end_type', hue='match_nick', shrink=0.5, multiple='dodge')
#     g.fig.set_size_inches(3,3)
#     g.legend.set_bbox_to_anchor((.61, .6))
    plt.savefig("./out/trial_success.png")
    plt.show()

    
def render_color_distr_across_maps(match_id, match_nick=None, dyad_id=None, trial_bins=4, n_trials=80, group=None):
    fig, axs = plt.subplots(1, trial_bins, dpi=200, figsize=(5, 2))
    bin_trials = n_trials // trial_bins
    player_counts = []
    for x in range(trial_bins):
        player_counts.append([np.zeros((11, 11)), np.zeros((11, 11))])
    for i, trial in enumerate(Trial.All(match_id)):
        if i < 41:
            i -= 1
        elif i > 41:
            i -= 2
        bin_num = i // bin_trials
        if trial.practice():
            continue
        p0_counts, p1_counts = trial.occupancy_distributions()
        player_counts[bin_num][0] += p0_counts
        player_counts[bin_num][1] += p1_counts
    for ax, bin_num in zip(fig.axes, range(trial_bins)):
        player_total = player_counts[bin_num][0] + player_counts[bin_num][1]
        player_nonzero = player_total > 0
        pct_p0 = np.zeros(p0_counts.shape)
        pct_p0[player_nonzero] = player_counts[bin_num][0][player_nonzero] / player_total[player_nonzero]
        pct_p0[~player_nonzero] = 0.5  # white?
        ax.imshow(pct_p0, cmap="PiYG_r", vmin=0, vmax=1, origin='lower')
        ax.set_title("Trials %d-%d" % (bin_num * bin_trials + 1, (bin_num + 1) * bin_trials))
        ax.grid(False)
        ax.set_axis_off()
    fn = "./out/occup_dyad_%d.png" % dyad_id
    plt.suptitle("Dyad %d (%s)" % (dyad_id, group), fontsize=12)
    plt.savefig(fn, facecolor='white')
    plt.show()
    
def anova_figure(df=None, ax=None, dv='fairness', dv_label="Work Balance", 
                 x=None, show_legend=True, title=None, ylim=None):
    if df is None:
        df = df_trials
    order = ['G', 'C'] if x == "cond_count" else ["HMC-Bal", "HMC-Imb", "LMC-Bal", "LMC-Imb"]
    g = sns.barplot(data=df, x=x, y=dv, order=order, ec=(1,1,1,0.5), 
                    hue='first_counter_block', hue_order=['C', 'G'], 
                    palette=['purple', 'gray'], ax=ax)
    ax.set_ylabel(dv_label, fontsize=10)
    if x == "cond_count":
        ax.set_xticklabels(["HMC", "LMC"])
        ax.set_xlabel("Counter", fontsize=10)
    else:
        ax.set_xlabel("Counter-Color Balance", fontsize=10)

    # Note that bars are ordered by color first
    b1_blocks = [2, 3, 4, 5] if x == "count-bal" else [1, 2]
    for i, bar in enumerate(ax.patches):
        if i in b1_blocks:
            bar.set_hatch('//')
    ax.tick_params(axis='x', labelsize=8)
    if not title:
        title = "%s by Group & Condition" % dv_label
    ax.set_title(title, fontsize=11)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_legend:
        leg = g.legend_
        leg.set_title("Group")
        new_labels = ['LMC-first', 'HMC-first']
        for t, l in zip(g.legend_.texts, new_labels):
            t.set_text(l)
    else:
        ax.legend([],[], frameon=False)
