PCOLORS = ["green", "magenta"]

FS = 8

COND_LABELS = {
    'cond_goals': "# of Goals",
    'cond_bal': 'Color Balance',
    'cond_count': 'Counter Type',
    'first_counter_block': 'First Counter'
}

COUNTER_LABELS = {
    "C": "LMC",
    "G": "HMC"
}

DV_LABELS = {
    "goal_color_split": "Colour Polarisation",
    "recent_ct_goal_color_split": "Colour Split (cross-trial)",
    "successful": "Success",
    "spatial_cons": "Spatial Consistency",
    "log_spatial_cons": "Log Spatial Consistency",
    "goal_pct_diff": "Inverse Fairness",
    "recent_ct_fairness": "Work balance (cross-trial)",
    "fairness": "Work balance",
    "first_mover_time": "First Mover Time",
    "second_mover_time": "Second Mover Time",    
    "total_moves": "# of moves (scaled)",
    "mean_dist": "Player separation (scaled)",
    "theta_diff": "Player angular separation",
    "duration": "Trial duration (scaled)",
    "ave_gem_diag": "Foraging Size",
    "spatial_heat_index": "Spatial Polarisation (mean)",
    "spatial_heat_index_prod": "Spatial Polarisation (prod)"    
}

DV_COLORS = {
    "goal_color_split": "purple",
    "recent_ct_goal_color_split": "#0CF",
    "spatial_cons": "red",
    "log_spatial_cons": "red",
    "spatial_heat_index": "red",
    "spatial_heat_index_prod": "red",    
    "successful": "green",
    "total_moves": "black",
    "goal_pct_diff": "orange",
    "fairness": "orange",    
    "recent_ct_fairness": "#FA7",
    "mean_dist": "red",
    "theta_diff": "blue",   
    "duration": "blue",
    "ave_gem_diag": "red"    
}

ROLLING_DVS = ["log_spatial_cons", "recent_ct_fairness", "recent_ct_goal_color_split"]  # "spatial_cons"