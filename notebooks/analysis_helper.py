import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import numpy as np

from util import dist
from trial import Trial
from constants import FS

def match_trials(df_trials, match_id=None, match_nick=None):
    if match_id is not None:
        df_match = df_trials[df_trials.match_id == match_id]
    elif match_nick is not None:
        df_match = df_trials[df_trials.match_nick == match_nick]        
    return df_match


def build_distance_df(df_trials, add_mean_rows=True):
    df_distance = pd.DataFrame(columns=['p2p_dist', 'p2p_dist_x', 'p2p_dist_y', 'time', 'trial', 'cond_goals', 'cond_bal', 'cond_count'])
    df_distance.cond_goals = df_distance.cond_goals.astype(int)

    for match_id in df_trials.match_id.unique():
        df_match = match_trials(df_trials, match_id)
        for key, trial_row in df_match.iterrows():
            trial_id = int(trial_row.trial)
            try:
                trial = Trial(trial_row.match_id, trial_idx=trial_id)
            except:
                continue
            states = trial.state_timeseries()
            trial_secs = (trial_row.ts_finished - trial_row.ts_started)/1000
            for s in states:
                loc0, loc1 = s.get('p0'), s.get('p1')
                p2p_dist = dist(loc0, loc1)
                p2p_dist_x = abs(loc1[0] - loc0[0])
                p2p_dist_y = abs(loc1[1] - loc0[1])            
                secs = s.get('time')
                df_distance = df_distance.append(pd.Series({
                    'match_id': match_id,
                    'trial': trial_id,
                    'time': secs,
                    'time_pct': secs / trial_secs,
                    'p2p_dist': p2p_dist,
                    'p2p_dist_x': p2p_dist_x,
                    'p2p_dist_y': p2p_dist_y,
                    'cond_goals': trial_row.cond_goals,
                    'cond_bal': trial_row.cond_bal,
                    'cond_count': trial_row.cond_count
                }, name="%s_%.2f" % (trial_row.trial, secs)))

    df_distance['trial'] = df_distance['trial'].astype(int)        
    df_distance['level'] = "g" + df_distance['cond_goals'].astype(str) + "b" + df_distance['cond_bal'] + "c" + df_distance['cond_count']
    df_distance['p2p_dist_x'] = df_distance['p2p_dist_x'].astype(float)
    df_distance['p2p_dist_y'] = df_distance['p2p_dist_y'].astype(float)
    df_distance['time_pct_dig'] = np.digitize(df_distance['time_pct'], bins = np.arange(0, 1.05, 0.025))
    df_distance['time_pct_dig'] /= df_distance['time_pct_dig'].max()
    
    if add_mean_rows:
        for goals in [8, 12]:
            for each_time_pct_dig in df_distance['time_pct_dig'].unique():
                dig_view = df_distance[(df_distance.time_pct_dig == each_time_pct_dig) & (df_distance.cond_goals == goals)]
                df_distance = df_distance.append(pd.Series({
                    'level': 'mean_g%d' % goals,
                    'cond_goals': goals,
                    'time_pct_dig': each_time_pct_dig,
                    'p2p_dist': df_distance[df_distance.time_pct_dig == each_time_pct_dig]['p2p_dist'].mean(),
                    'p2p_dist_x': dig_view['p2p_dist_x'].mean(),
                    'p2p_dist_y': dig_view['p2p_dist_y'].mean()
                }, name='mean_g%d_%s' % (goals, each_time_pct_dig)))
    
    return df_distance