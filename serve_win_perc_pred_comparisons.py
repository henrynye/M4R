import json
import pandas as pd
import os
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression


ta_root_dir = 'tennis_abstract'
ta_df_dir = 'tennis_abstract_dfs'
matchhead_verbose = ["date","tourn","surf","level","win/loss","rank","seed","entry","round",
                 "score","max_num_of_sets","opp","orank","oseed","oentry","ohand","obday",
                 "oheight","ocountry","oactive","time_minutes","aces","dfs","service_pts","first_serves_in","first_serves_won",
                 "second_serves_won",'service_games',"break_points_saved","break_points_faced","oaces","odfs","oservice_pts","ofirst_serves_in","ofirst_serves_won",
                 "osecond_serves_won",'oservice_games',"obreak_points_saved","obreak_points_faced", "obackhand", "chartlink",
                 "pslink","whserver","matchid","wh","roundnum","matchnum", "oforeign_key"]

round_vals = {
    'F': 1,
    'SF': 2,
    'QF': 3,
    'R16': 4,
    'R32': 5,
    'R64': 6,
    'R128': 7,
    'Q3': 8,
    'Q2': 9,
    'Q1': 10
}
round_vals = defaultdict(lambda: 11, round_vals)
prev_match_stats = ['prev_10_mean_serve_win_rate','prev_10_median_serve_win_rate','prev_10_mean_return_win_rate','prev_10_median_return_win_rate','prev_10_mean_first_serve_in_rate','prev_10_median_first_serve_in_rate','prev_10_mean_first_return_in_rate','prev_10_median_first_return_in_rate','prev_10_mean_first_return_win_rate','prev_10_median_first_return_win_rate','prev_10_mean_overall_win_rate','prev_10_median_overall_win_rate']
o_prev_match_stats = [f'o{n}' for n in prev_match_stats]


def calc_serve_win_rate(row):
    elems = ['first_serves_won', 'second_serves_won', 'service_pts']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = float(row['service_pts'])
    if denom == 0:
        return 0
    return (float(row['first_serves_won']) + float(row['second_serves_won'])) / denom

def calc_overall_win_rate(row):
    elems = ['first_serves_won', 'second_serves_won', 'ofirst_serves_won', 'osecond_serves_won', 'service_pts', 'oservice_pts']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = float(row['service_pts'] + row['oservice_pts'])
    if denom == 0:
        return 0
    return (float(row['first_serves_won']) + float(row['second_serves_won']) + (float(row['oservice_pts']) - float(row['first_serves_won']) - float(row['second_serves_won'])) ) / denom

def calc_return_win_rate(row):
    elems = ['ofirst_serves_won', 'osecond_serves_won', 'oservice_pts']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = float(row['oservice_pts'])
    if denom == 0:
        return 0
    return 1 - ((float(row['ofirst_serves_won']) + float(row['osecond_serves_won'])) / denom)

def calc_first_serve_win_rate(row):
    elems = ['first_serves_won', 'first_serves_in']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = float(row['first_serves_in'])
    if denom == 0:
        return 0
    return float(row['first_serves_won']) / denom

def calc_first_serve_in_rate(row):
    elems = ['service_pts', 'first_serves_in']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = float(row['service_pts'])
    if denom == 0:
        return 0
    return float(row['first_serves_in']) / denom

def calc_first_return_in_rate(row):
    elems = ['oservice_pts', 'ofirst_serves_in']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = float(row['oservice_pts'])
    if denom == 0:
        return 0
    return float(row['ofirst_serves_in']) / denom

def calc_second_serve_win_rate(row):
    elems = ['second_serves_won', 'service_pts', 'first_serves_in']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = (float(row['service_pts']) - float(row['first_serves_in']))
    if denom == 0:
        return 0
    return float(row['second_serves_won']) / denom
def calc_first_return_win_rate(row):
    elems = ['ofirst_serves_won', 'ofirst_serves_in']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = float(row['ofirst_serves_in'])
    if denom == 0:
        return 0
    return 1 - (float(row['ofirst_serves_won']) / denom)

def calc_second_return_win_rate(row):
    elems = ['osecond_serves_won', 'oservice_pts', 'ofirst_serves_in']
    if any([row[elem] == '' for elem in elems]):
        return None
    denom = (float(row['oservice_pts']) - float(row['ofirst_serves_in']))
    if denom == 0:
        return 0
    return 1-(float(row['osecond_serves_won']) / denom)

def edit_df(name):
    if not os.path.exists(f'{ta_df_dir}/{name}.pkl'):
        return
    match_df = pd.read_pickle(f'{ta_df_dir}/{name}.pkl')
    match_df['formatted_opp'] = match_df['opp'].apply(lambda name: name.replace(' ', ''))
    match_df['round_val'] = match_df['round'].apply(lambda rd: round_vals[rd])
    match_df['serve_win_rate'] = match_df.apply(calc_serve_win_rate, axis=1)
    match_df['overall_win_rate'] = match_df.apply(calc_overall_win_rate, axis=1)
    match_df['return_win_rate'] = match_df.apply(calc_return_win_rate, axis=1)
    match_df['first_serve_in_rate'] = match_df.apply(calc_first_serve_in_rate, axis=1)
    match_df['first_return_in_rate'] = match_df.apply(calc_first_return_in_rate, axis=1)
    match_df['first_serve_win_rate'] = match_df.apply(calc_first_serve_win_rate, axis=1)
    match_df['second_serve_win_rate'] = match_df.apply(calc_second_serve_win_rate, axis=1)
    match_df['first_return_win_rate'] = match_df.apply(calc_first_return_win_rate, axis=1)
    match_df['second_return_win_rate'] = match_df.apply(calc_second_return_win_rate, axis=1)

    rows_to_go = match_df[match_df['serve_win_rate'].isna() | match_df['score'].str.contains('RET') | ~match_df['win/loss'].isin(['W', 'L']) ]
    # assert len(rows_to_go) < 0.15 * len(match_df)
    match_df.drop(rows_to_go.index, inplace=True)
    match_df.sort_index(ascending=False, inplace=True)
    window_size = 10
    match_df['prev_10_mean_serve_win_rate'] = match_df['serve_win_rate'].rolling(window=window_size, closed='left').mean()
    match_df['prev_10_median_serve_win_rate'] = match_df['serve_win_rate'].rolling(window=window_size, closed='left').median()
    match_df['prev_10_mean_return_win_rate'] = match_df['return_win_rate'].rolling(window=window_size, closed='left').mean()
    match_df['prev_10_median_return_win_rate'] = match_df['return_win_rate'].rolling(window=window_size, closed='left').median()
    match_df['prev_10_mean_first_serve_win_rate'] = match_df['first_serve_win_rate'].rolling(window=window_size, closed='left').mean()
    match_df['prev_10_median_first_serve_win_rate'] = match_df['first_serve_win_rate'].rolling(window=window_size, closed='left').median()
    match_df['prev_10_mean_first_serve_in_rate'] = match_df['first_serve_in_rate'].rolling(window=window_size, closed='left').mean()
    match_df['prev_10_median_first_serve_in_rate'] = match_df['first_serve_in_rate'].rolling(window=window_size, closed='left').median()
    match_df['prev_10_mean_first_return_in_rate'] = match_df['first_return_in_rate'].rolling(window=window_size, closed='left').mean()
    match_df['prev_10_median_first_return_in_rate'] = match_df['first_return_in_rate'].rolling(window=window_size, closed='left').median()
    match_df['prev_10_mean_first_return_win_rate'] = match_df['first_return_win_rate'].rolling(window=window_size, closed='left').mean()
    match_df['prev_10_median_first_return_win_rate'] = match_df['first_return_win_rate'].rolling(window=window_size, closed='left').median()
    match_df['prev_10_mean_overall_win_rate'] = match_df['overall_win_rate'].rolling(window=window_size, closed='left').mean()
    match_df['prev_10_median_overall_win_rate'] = match_df['overall_win_rate'].rolling(window=window_size, closed='left').median()
    match_df = pd.concat([match_df, rows_to_go])
    match_df.sort_index(ascending=False, inplace=True)
    pd.to_pickle(match_df, f'{ta_df_dir}/{name}.pkl')


def get_player_nums(base_p_name):
    alcaraz_html_path = f'{ta_root_dir}/{base_p_name}.html'
    with open(alcaraz_html_path, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    script_tags = soup.find_all("script")
    ochoices = None
    for script in script_tags:
        if script.string:  # Ensure the script tag contains JavaScript code
            match = re.search(r'var\s+ochoices\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
            if match:
                # Extract the JavaScript array or object
                ochoices_str = match.group(1)
                # Convert the JavaScript array/object to a Python object
                ochoices = eval(ochoices_str)
    ta_names = [v.replace(' ', '') for v in ochoices]
    edit_df(base_p_name)
    for name in ta_names:
        edit_df(name)
    def set_opposition_prev_match_stats(match):
        if not os.path.exists(f'{ta_df_dir}/{match["formatted_opp"]}.pkl'):
            return pd.Series([np.nan] * len(prev_match_stats), index=prev_match_stats)
        opp_df = pd.read_pickle(f'{ta_df_dir}/{match["formatted_opp"]}.pkl')
        opp_match_stats = opp_df.loc[(opp_df['date'] == match['date']) & (opp_df['formatted_opp'] == base_p_name) & (opp_df['round'] == match['round']), prev_match_stats]
        assert opp_match_stats.shape[0] <= 1
        # if opp_match_stats.shape[0] > 1:
        #     breakpoint()
        if opp_match_stats.shape[0] == 1:
            opp_match_stats = opp_match_stats.iloc[0]
            if len(opp_match_stats) == len(prev_match_stats):
                return opp_match_stats
        return pd.Series([np.nan] * len(prev_match_stats), index=prev_match_stats)
    match_df = pd.read_pickle(f'{ta_df_dir}/{base_p_name}.pkl')
    match_df[o_prev_match_stats] = np.nan
    match_df[o_prev_match_stats] = match_df.apply(set_opposition_prev_match_stats,axis=1)
    # match_df.loc[match_df['serve_win_rate'] < 0.3, 'score']
    X_cols1 = ['prev_10_mean_serve_win_rate', 'oprev_10_mean_return_win_rate']
    X_cols2 = ['prev_10_mean_first_serve_win_rate','prev_10_mean_first_serve_in_rate','oprev_10_mean_first_return_win_rate','oprev_10_mean_first_return_in_rate']
    X_cols3 = ['prev_10_mean_overall_win_rate', 'oprev_10_mean_overall_win_rate', 'prev_10_mean_serve_win_rate', 'oprev_10_mean_return_win_rate']
    matches_with_oprev_values = match_df.dropna(subset=o_prev_match_stats+prev_match_stats)
    def get_LR_vals(X_cols, idx):
        model = LinearRegression()
        X = matches_with_oprev_values[X_cols]
        y = matches_with_oprev_values['serve_win_rate']
        model.fit(X, y)
        match_df[f'LR_serve_win_rate_pred_{idx}'] = match_df.loc[matches_with_oprev_values.index].apply(lambda row: np.dot(row[X_cols],model.coef_)+model.intercept_, axis=1)
        return [round(c,2) for c in model.coef_ + [model.intercept_]]
    LR_vals1 = get_LR_vals(X_cols1, 1)
    LR_vals2 = get_LR_vals(X_cols2, 2)
    LR_vals3 = get_LR_vals(X_cols3, 3)
    matches_with_LR_pred = match_df.dropna(subset=['LR_serve_win_rate_pred_1'])
    real_vals = list(matches_with_LR_pred['serve_win_rate'])
    num_of_vals = len(real_vals)
    correct_threshold = 0.01
    predicted_vals_mean = matches_with_LR_pred['prev_10_mean_serve_win_rate']
    predicted_vals_median = matches_with_LR_pred['prev_10_median_serve_win_rate']
    predicted_vals_LR = matches_with_LR_pred['LR_serve_win_rate_pred_1']
    predicted_vals_LR_2 = matches_with_LR_pred['LR_serve_win_rate_pred_2']
    predicted_vals_LR_3 = matches_with_LR_pred['LR_serve_win_rate_pred_3']
    def get_SD(preds):
        return statistics.stdev(np.abs(np.subtract(real_vals, preds)))

    def get_MAE(preds):
        return np.abs(np.subtract(real_vals, preds)).mean()
    
    def get_correct_rate(preds):
        return float(len([p for p,r in zip(real_vals, preds) if abs(p-r)/r < correct_threshold])) / num_of_vals
    

    mean_SD = get_SD(predicted_vals_mean)
    median_SD = get_SD(predicted_vals_median)
    LR_SD = get_SD(predicted_vals_LR)
    LR_2_SD = get_SD(predicted_vals_LR_2)
    LR_3_SD = get_SD(predicted_vals_LR_3)
    mean_MAE = get_MAE(predicted_vals_mean)
    median_MAE = get_MAE(predicted_vals_median)
    LR_MAE = get_MAE(predicted_vals_LR)
    LR_2_MAE = get_MAE(predicted_vals_LR_2)
    LR_3_MAE = get_MAE(predicted_vals_LR_3)
    mean_correct_rate = get_correct_rate(predicted_vals_mean)
    median_correct_rate = get_correct_rate(predicted_vals_median)
    LR_correct_rate = get_correct_rate(predicted_vals_LR)
    LR_2_correct_rate = get_correct_rate(predicted_vals_LR_2)
    LR_3_correct_rate = get_correct_rate(predicted_vals_LR_3)

    return mean_MAE, median_MAE, LR_MAE, LR_2_MAE, LR_3_MAE 

with open('api_p_keys_to_ta_names.json', 'r') as file:
    players = json.load(file)
ta_names = [v for k,v in players.items()]
vals = []
ta_names = ['CarlosAlcaraz', 'JannikSinner', 'AlexanderZverev','TaylorFritz' ,'DaniilMedvedev', 'AndreyRublev', 'StefanosTsitsipas', 'CasperRuud', 'TommyPaul', 'AlexDeMinaur','NovakDjokovic']
for n in tqdm(ta_names):
    vals.append(get_player_nums(n))
