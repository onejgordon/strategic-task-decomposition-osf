import numpy as np
import matplotlib.pyplot as plt
import json
import os

from util import dist, min_bounding_ellipse
from problems_util import ellipse_metrics
from scipy.stats import circmean

from constants import PCOLORS

VERSION = "v4"
MAP_VERSION = "v3"

class Trial():
    N_TRIALS = 82
    
    def __init__(self, match_id, trial_idx=0, version=VERSION):
        self.version = version
        self.match_id = match_id
        self.trial_idx = trial_idx
        self.problem = None
        self.trial_data = None
        self.metadata = None
        self.p0_data = None
        self.p1_data = None
        self.load_metadata(match_id)
        self.load_trial(trial_idx)
    
    @staticmethod
    def All(match_id):
        all_trials = []
        for i in range(Trial.N_TRIALS):
            try:
                t = Trial(match_id, trial_idx=i)
            except Exception as e:
                print("Error loading trial data: %s, trial %s: %s" % (match_id, i, e))
            else:
                all_trials.append(t)
        return all_trials

    def load_metadata(self, match_id):
        with open("../data/trial_data/%s/metadata.json" % (match_id), 'r') as f:
            self.metadata = json.load(f)
    
    def practice(self):
        map_order = self.metadata.get("matchData").get("map_order")
        map_name = map_order[self.trial_idx]
        return map_name.startswith('prac_')  # Trials 0 and 41
        
    def first_counter(self):
        code_even = self.metadata.get("matchData", {}).get('code') % 2 == 0
        return "C" if code_even else "G"

    def counter_order(self):
        code_even = self.metadata.get("matchData", {}).get('code') % 2 == 0
        return "C->G" if code_even else "G->C"
    
    def load_trial(self, trial_idx):
        # Get trial metadata
        self.trial_data = self.metadata.get('trials').get(str(int(trial_idx)))
        # Get problem / map
        map_name = self.metadata.get("matchData").get("map_order")[trial_idx]
        with open("../stimuli/final_maps/%s.json" % (map_name), 'r') as f:
            self.problem = json.load(f)
        # Get each player's stored data
        for p in [0, 1]:
            with open("../data/trial_data/%s/trial%d_p%d.json" % (self.match_id, trial_idx, p), 'r') as f:
                data = json.load(f)
                if p == 0:
                    self.p0_data = data
                else:
                    self.p1_data = data
        
    def node_loc(self, node_id):
        n = self.problem.get("nodes").get(str(node_id))
        return (n.get('X'), n.get('Y'))
    
    def node_color(self, node_id):
        n = self.problem.get("nodes").get(str(node_id))
        return n.get('color')
    
    def successful(self):
        return self.trial_data.get("successful")
    
    def get_duration(self):
        """
        Use trial start and last host move to overcome out-of-sync issues
        with ts_finished
        """
#         ts_started = self.trial_data.get('ts_started')
#         return (ts_finished - ts_started) / 1000.
        last_p0_event = self.events(pid=0, moves_only=False)[-1]
        return last_p0_event.get("time")

        
    def players(self):
        pl = self.metadata.get("matchData").get("players")
        return [pl[str(i)] for i in range(2)]

    def state_timeseries(self):
        """
        Using event data, reconstruct state timeseries for both players
        """
        middle_loc = self.node_loc(self.problem.get("middle_node"))
        # Both start at middle
        state = {
            'p0': middle_loc,
            'p1': middle_loc,
            'time': 0
        }  
        states = [state]
        p0_events = self.events(pid=0, moves_only=True)
        p1_events = self.events(pid=1, moves_only=True)        
        cursor_0 = 0
        cursor_1 = 0
        done = False
        while not done:
            next_p0 = p0_events[cursor_0].get("time") if cursor_0 < len(p0_events) else float('inf')
            next_p1 = p1_events[cursor_1].get("time") if cursor_1 < len(p1_events) else float('inf')          
            next_move_player = 0 if next_p0 < next_p1 else 1
            next_move = None
            if next_move_player == 0 and cursor_0 < len(p0_events):
                next_move = p0_events[cursor_0]
                cursor_0 += 1
            elif cursor_1 < len(p1_events):
                next_move = p1_events[cursor_1]
                cursor_1 += 1
            if next_move is not None:
                next_state = dict(states[-1])    
                next_state['p%d' % next_move_player] = self.node_loc(next_move.get("nodeId"))
                next_state['time'] = next_move.get("time")
                states.append(next_state)
            done = cursor_0 >= len(p0_events) and cursor_1 >= len(p1_events)
        return states
        
    def gem_stack_timeseries(self):
        stack = {0: [], 1: []}
        for player in [0, 1]:
            p_goal_ids = self.trial_data.get('goals_collected').get(str(player), [])
            for node_id in p_goal_ids:
                color = self.node_color(node_id)
                stack[player].append(color)
        return stack
    
    def first_gem_collector(self):
        p0_gems = self.events(pid=0, collect_only=True)
        p1_gems = self.events(pid=1, collect_only=True)        
        first_p0 = p0_gems[0].get('time') if p0_gems else float('inf')
        first_p1 = p1_gems[0].get('time') if p1_gems else float('inf')
        return 0 if first_p0 < first_p1 else 1
                    
    def mean_thetas(self):
        middle_loc = self.node_loc(self.problem.get("middle_node"))
        states = self.state_timeseries()
        p0_thetas = []
        p1_thetas = []
        dthetas = []
        for s in states:
            x, y = s.get('p0')
            dy = y - middle_loc[1]
            dx = x - middle_loc[0]
            p0_theta = np.arctan2(dy, dx)
            p0_thetas.append(p0_theta)
            x, y = s.get('p1')
            dy = y - middle_loc[1]
            dx = x - middle_loc[0]
            p1_theta = np.arctan2(dy, dx)
            p1_thetas.append(p1_theta)

            theta_diff = p1_theta - p0_theta
            theta_diff = abs((theta_diff + np.pi) % (2*np.pi) - np.pi)
            dthetas.append(theta_diff)
        return circmean(p0_thetas), circmean(p1_thetas), np.mean(dthetas)
    
    def mean_coords(self):
        middle_loc = self.node_loc(self.problem.get("middle_node"))
        states = self.state_timeseries()
        p0_xs = []
        p0_ys = []
        p1_xs = []
        p1_ys = []
        dists = []
        for s in states:
            p0 = s.get('p0')
            dx = p0[0] - middle_loc[0]
            dy = p0[1] - middle_loc[1]
            p0_xs.append(dx)
            p0_ys.append(dy)            
            p1 = s.get('p1')
            dx = p1[0] - middle_loc[0]
            dy = p1[1] - middle_loc[1]
            p1_xs.append(dx)
            p1_ys.append(dy)          
            dists.append(dist(p0, p1))
        return np.mean(p0_xs), np.mean(p0_ys), np.mean(p1_xs), np.mean(p1_ys), np.mean(dists)
        
    def start_node(self, pid=0):
        return self.problem.get("middle_node")
    
    def move_dir(self, from_node_id, to_node_id):
        loc0 = np.array(self.node_loc(from_node_id))
        loc1 = np.array(self.node_loc(to_node_id))
        vector = loc1 - loc0
        assert abs(vector.sum()) == 1
        if np.array_equal(vector, [1, 0]):
            return 'right'
        elif np.array_equal(vector, [-1, 0]):
            return 'left'
        elif np.array_equal(vector, [0, 1]):
            return 'up'
        elif np.array_equal(vector, [0, -1]):
            return 'down'

    def spatial_index_gem_centroids(self):
        p0_goals = self.trial_data.get("goals_collected")[0]
        p1_goals = self.trial_data.get("goals_collected")[1]

    def spatial_index_ellipses(self):
        p0_goals = self.trial_data.get("goals_collected")[0]
        p1_goals = self.trial_data.get("goals_collected")[1]
        
        ellipse_data = {}
        for p in ['p0', 'p1']:
            center, a, b, angle = min_bounding_ellipse(np.stack(points), jitter=0.2)
            size = ellipse_size(a, b)
            ellipse_data[p] = {
                'center': center.tolist(),
                'a': a,
                'b': b,
                'angle': angle
            }

        es_p0, es_p1, overlap = ellipse_metrics(ellipse_data, keys=['p0', 'p1'])
        max_el_size = 1. # TODO
        mean_size = (es_p0 + es_p1)/2./max_el_size
        index = (1-overlap) * (1-mean_size)
        return index
    
    def events(self, pid=0, moves_only=False, collect_only=False):
        if pid == 0:
            data = self.p0_data
        else:
            data = self.p1_data
        events = data.get("trial_data").get("TrialEventData", [])
        if events is None:
            events = []
        if not events:
            print("M %s T %d PID %d has no events" % (self.match_id, self.trial_idx, pid))
        if moves_only:
            events = [e for e in events if e.get("eventType") == "move"]
        elif collect_only:
            events = [e for e in events if e.get("eventType") == "collect"]
        return events
        
    def render_problem(self, ax=None, title=None, no_rewards=False):
        colors = ['red', 'blue']
        if ax is None:
            fig, ax = plt.subplots(dpi=144, figsize=(5, 5))
        X = []
        Y = []
        C = []
        for node_id, n in self.problem.get('nodes').items():
            X.append(n.get('X'))
            Y.append(n.get('Y'))   
            color = n.get('color') if not no_rewards else None
            if color is None:
                c = 'gray'
            else:
                c = colors[color]
            C.append(c)
        for link_key in self.problem.get('links').keys():
            src_id, tgt_id = link_key.split('_')
            src = self.problem.get('nodes')[src_id]
            tgt = self.problem.get('nodes')[tgt_id]        
            sx, sy = src.get('X'), src.get('Y')
            tx, ty = tgt.get('X'), tgt.get('Y')
            ax.plot([sx, tx], [sy, ty], c='gray', zorder=1)
        ax.scatter(X, Y, c=C, s=50, zorder=10)
        if title:
            ax.set_title(title, fontsize=8)
        ax.set_axis_off()
        return ax
        
    
    def render(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(dpi=144, figsize=(5, 5))
        self.render_problem(ax=ax)
        for i, pdata in enumerate([self.p0_data, self.p1_data]):
            X, Y, C = [], [], []
            pcolor = PCOLORS[i]
            for event in self.events(pid=i, moves_only=True):
                node_id = event.get("nodeId")
                eventType = event.get("eventType")
                n = self.problem.get('nodes')[str(node_id)]
                X.append(n.get("X"))
                Y.append(n.get("Y"))                
                C.append(pcolor)
            ax.plot(X, Y, lw=5, c=pcolor, alpha=0.5)
#             ax.scatter(X, Y, c=C, s=10, zorder=10)

    def occupancy_distributions(self):
        """
        Returns list(2) of 11x11 arrays holding counts of each player on each cell
        """
        counts = [np.zeros((11, 11)), np.zeros((11, 11))]
        for i, pdata in enumerate([self.p0_data, self.p1_data]):
            X, Y, C = [], [], []
            pcolor = PCOLORS[i]
            for event in pdata.get("trial_data").get("TrialEventData", []):
                node_id = event.get("nodeId")
                if node_id:
                    eventType = event.get("eventType")
                    n = self.problem.get('nodes')[str(node_id)]
                    x, y = n.get("X"), n.get("Y")
                    counts[i][y, x] += 1
        return counts
        