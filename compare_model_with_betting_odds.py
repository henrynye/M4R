import json
from utils.utilities import *
from utils.Match import Match
from utils.MatchSkeleton import MatchSkeleton
import csv
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')


michelsen_minaur = ('michelsen_minaur', '12008400', 1, 0.634, 0.641)
sinner_rune = ('sinner_rune', '12008434', 0, 0.712, 0.66)
djokovic_alcaraz = ('djokovic_alcaraz', '12008501', 0, 0.684, 0.679)

match_name, match_key, player_to_plot, p0_serve_odds, p1_serve_odds = djokovic_alcaraz
odds_file_path = f'{match_name}/scores_and_odds_mapping.csv'
odds_throughout_match_path = f'{match_name}/odds_throughout_match.csv'

json_data={
    'match_key': match_key
}

def generate_match_scores_from_servers_and_ptwinners(formatted_pbp):
    starting_serve = formatted_pbp[0][0]
    ptwinners = [ptwinner for s,ptwinners in formatted_pbp for ptwinner in ptwinners]
    matchSkeleton = MatchSkeleton(starting_serve=starting_serve, sets=3)
    for winner in ptwinners:
        matchSkeleton.point_win(winner-1)
    return matchSkeleton.scores

def distance_from_bounds(val, bound_1, bound_2):
    midpoint = (bound_1+bound_2)/2
    return abs(val-midpoint)

def get_betting_odds(odds):
    win_probs = []
    for odd in odds:
        if odd:
            win_probs.append(odd)
            break
    for odd in odds:
        if odd:
            win_probs.append(odds_string_to_probability(odd))
        else:
            win_probs.append(win_probs[-1])
    win_probs = win_probs[1:]
    return win_probs

def write_scores_and_odds_mapping_from_raw_files():
    # Pick odds files
    files = os.listdir(match_name)
    files = [file for file in files if 'json' in file]
    relevant_file_idxs_path = f'{match_name}/relevant_file_idxs.csv'
    if os.path.exists(relevant_file_idxs_path):
        with open(relevant_file_idxs_path, 'r') as file:
            relevant_file_idxs = list(csv.reader(file))[0]
            files = [file for file in files if file[12:-5] in relevant_file_idxs]

    # Make scores_and_odds_mapping.csv
    scores = []
    first_idxs = []
    odds = []
    for file in files:
        match_info = pd.read_json(f'{match_name}/{file}', encoding='latin-1')
        important_info = match_info.iloc[0]
        match_odds_id = int(match_info.loc[match_info['NA']=='Match', 'ID'].iloc[0])
        g_score = important_info['XP']
        s_score = important_info['SS']
        score = f'{s_score},{g_score}'
        p0,p1 = important_info['NA'].split(' v ')
        if score not in scores:
            scores.append(score)
            idx = file[12:-5]
            first_idxs.append(idx)
            match_odds_idxs = match_info[match_info['MA']==match_odds_id].index
            assert match_odds_idxs.shape[0] == 2
            match_info.loc[match_odds_idxs, 'NA'] = match_info['NA'].shift().loc[match_odds_idxs]
            odds.append((match_info.loc[(match_info['MA'] == match_odds_id) & (match_info['NA'] == p0), 'OD'].iloc[0], match_info.loc[(match_info['MA'] == match_odds_id) & (match_info['NA'] == p1), 'OD'].iloc[0]))

    scores_and_odds = [(s,o1,o2) for s,(o1,o2) in zip(scores, odds)]
    sorted_so = sorted(scores_and_odds, key=lambda row: row[0])
    with open(odds_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Score', 'Odds1', 'Odds2'])
        writer.writerows(sorted_so)
    with open(relevant_file_idxs_path,'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(first_idxs)



def write_odds_throughout_match_from_scores_and_odds_mapping(formatted_pbp):
    all_scores = generate_match_scores_from_servers_and_ptwinners(formatted_pbp)
    with open(odds_file_path, 'r') as file:
        scores_and_odds = list(csv.reader(file))
    score_odds_dict = {
        score: (odd1, odd2) for score, odd1, odd2 in scores_and_odds
    }
    all_scores_and_odds = [(point, score, *score_odds_dict[score]) if score in score_odds_dict.keys() else (point, score, None, None) for point, score in zip(just_points, all_scores)]
    with open(odds_throughout_match_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PtWinner', 'Score', 'Odds1', 'Odds2'])
        writer.writerows(all_scores_and_odds)

def pick_best_fitting_serve_ps(ptwinners_and_odds):
    pts = [p for p,o1,o2 in ptwinners_and_odds]
    p1_win_odds = get_betting_odds([o1 for p, o1, o2 in ptwinners_and_odds])
    p2_win_odds = get_betting_odds([o2 for p, o1, o2 in ptwinners_and_odds])

    win_odds = [elem[player_to_plot] for elem in zip(p1_win_odds, p2_win_odds)]
    loss_odds = [1-elem[1-player_to_plot] for elem in zip(p1_win_odds, p2_win_odds)]
    serve_p1_range = [i/100 for i in range(50, 100)]
    serve_p2_range = [i/100 for i in range(50, 100)]
    closest_to_bounds = (1,0,0)

    for p1 in serve_p1_range:
        for p2 in serve_p2_range:        
            match = Match(3, 0, p1, p2)
            match.simulate_match_by_point(pts)
            model_probs = [1 - prob if player_to_plot == 1 else prob for prob in match.pre_points_match_probs][:-1]
            sum_sqrs_distances = sum([distance_from_bounds(val, win_odds[i], loss_odds[i])**2 for i, val in enumerate(model_probs)])
            if sum_sqrs_distances < closest_to_bounds[0]:
                closest_to_bounds = (sum_sqrs_distances, p1, p2)

    p1_val = int(closest_to_bounds[1]*1000)
    p2_val = int(closest_to_bounds[2]*1000)
    serve_p1_range = [i/1000 for i in range(p1_val-10, p1_val+10)]
    serve_p2_range = [i/1000 for i in range(p2_val-10, p2_val+10)]
    closest_to_bounds = (1,0,0)
    for p1 in serve_p1_range:
        for p2 in serve_p2_range:        
            match = Match(3, 0, p1, p2)
            match.simulate_match_by_point(pts)
            model_probs = [1 - prob if player_to_plot == 1 else prob for prob in match.pre_points_match_probs][:-1]
            sum_sqrs_distances = sum([distance_from_bounds(val, win_odds[i], loss_odds[i])**2 for i, val in enumerate(model_probs)])
            if sum_sqrs_distances < closest_to_bounds[0]:
                closest_to_bounds = (sum_sqrs_distances, p1, p2)

    _, best_p0_serve_odds, best_p1_serve_odds = closest_to_bounds
    print(closest_to_bounds)
    return best_p0_serve_odds, best_p1_serve_odds

def odds_to_implied_probability(odds):
    """
    Converts decimal betting odds to implied probability.
    Example: odds=2.5 -> 1/2.5 = 0.4 (40%)
    """
    try:
        odds = float(odds)
        if odds <= 0:
            return None
        return 1.0 / odds
    except (ValueError, TypeError):
        return None
# #---------------------------------------------------------------------------------

bets_match_key = get_bets_match_key_from_api_match_key(match_key)
match = get_match_info_locally(bets_match_key, 'matches_api_info', 'json')
formatted_pbp = get_servers_and_ptwinners_from_api_scores(match)
just_points = [point for s, points in formatted_pbp for point in points]
first_server, _ = formatted_pbp[0]

if not os.path.exists(odds_file_path):
    write_scores_and_odds_mapping_from_raw_files()

if not os.path.exists(odds_throughout_match_path):
    write_odds_throughout_match_from_scores_and_odds_mapping(formatted_pbp)

with open(odds_throughout_match_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)
    ptwinners_and_odds = [(int(ptwinner)-1,odd1,odd2) for ptwinner,_,odd1, odd2 in reader]
pts = [p for p,o1,o2 in ptwinners_and_odds]

p0_serve_odds, p1_serve_odds = pick_best_fitting_serve_ps(ptwinners_and_odds)
match = Match(3, first_server-1, p0_serve_odds, p1_serve_odds)
match.simulate_match_by_point(pts)
all_states_api = pd.DataFrame(columns=['states'])
all_states_api['states'] = match.historical_states
# all_states_api['sc_f_st'] = all_states_api['states'].apply(score_from_state)

# -------------

# bets_match_key = get_bets_match_key_from_api_match_key(match_key)
# match_df = get_match_info_locally(bets_match_key, 'matches_betting_odds', 'csv')

# match_api_info = get_match_info_locally(bets_match_key, 'matches_api_info', 'json')
# formatted_pbp = get_servers_and_ptwinners_from_api_scores(match_api_info)
# first_server, _ = formatted_pbp[0]

# match_obj = Match(3, first_server-1, p0_serve_odds, p1_serve_odds)
# reverse_players = False
# match_df[['state', 'in_tiebreak', 'currently_serving', 'set_starting_server']] = match_df['ss'].apply(lambda elem: match_obj.set_state_from_score(elem, reverse_players)).apply(pd.Series)
# match_df['progression_score'] = match_df['state'].apply(match_progression_score_from_states)
# match_df['model_odds'] = match_df.apply(lambda row: match_obj.get_match_win_prob_from_state(row['state'], row['in_tiebreak'], row['currently_serving'], row['set_starting_server']), axis=1)
# match_df.sort_values(by=['progression_score', 'add_time'], inplace=True)
# match_df.loc[match_df['model_odds'] == 1.0, ['home_od', 'away_od']] = [0, 1] if reverse_players else [1, 0] 
# match_df.loc[match_df['model_odds'] == 0.0, ['home_od', 'away_od']] = [1, 0] if reverse_players else [0, 1] 
# match_df = match_df.loc[match_df['home_od'] != '-'].reset_index()
# for i in range(0, len(match_df)):
#     if i == 0:
#         match_df.at[i, 'swap'] = False
#     else:
#         match_df.at[i, 'swap'] = are_odds_swapped(match_df.iloc[i-1], match_df.iloc[i])
# swap_mask = match_df['swap']
# match_df.loc[swap_mask, ['home_od', 'away_od']] = match_df.loc[swap_mask, ['away_od', 'home_od']].values

# def get_odds(score):
#     row = match_df.loc[match_df['sc_f_st'] == score, ['home_od', 'away_od', 'model_odds']]
#     return row.iloc[0] if not row.empty else pd.Series([None, None, None], index=['home_od', 'away_od', 'model_odds'])

# # all_states_api[['home_od', 'away_od', 'model_odds']] = all_states_api['sc_f_st'].apply(get_odds)

# hopeful_model_odds = list(match_df['model_odds'])
# final_prob = hopeful_model_odds[-1]
# # all_states_api = all_states_api[~all_states_api['home_od'].isna()]
# # for row in all_states_api.iterrows(): 
# #     if all_states_api['home_od'].iloc[i].isna():
# #         breakpoint()
# #         all_states_api['home_od'].iloc[i] = all_states_api['home_od'].iloc[i-1]
# #     if all_states_api['away_od'].iloc[i].isna():
# #         all_states_api['away_od'].iloc[i] = all_states_api['away_od'].iloc[i-1]
# home_od = list(match_df['home_od'])
# away_od = list(match_df['away_od'])
# breakpoint()
# if not home_od[0]:
#     home_od[0] = [h for h in home_od if h][0]
#     away_od[0] = [h for h in away_od if h][0]
#     hopeful_model_odds[0] = [h for h in hopeful_model_odds if h][0]
# for i in range(1, len(home_od)):
#     h = home_od[i]
#     if not h:
#         home_od[i] = home_od[i-1]
#         away_od[i] = away_od[i-1]
#         hopeful_model_odds[i] = hopeful_model_odds[i-1]
# match_df['home_ods'] = get_betting_odds([f'{float(od) - 1}/1' for od in home_od])
# match_df['away_ods'] = get_betting_odds([f'{float(od) - 1}/1' for od in away_od])
# hopeful_p1_win_odds = match_df['home_ods']
# hopeful_p2_win_odds = match_df['away_ods']
# win_odds = [elem[player_to_plot] for elem in zip(hopeful_p1_win_odds, hopeful_p2_win_odds)] + [final_prob]
# loss_odds = [1-elem[1-player_to_plot] for elem in zip(hopeful_p1_win_odds, hopeful_p2_win_odds)] + [final_prob]

# fig = plt.figure(figsize=(6, 6))
# gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])  # Zero vertical spacing
# ax1 = fig.add_subplot(gs[0])  # Top subplot
# ax2 = fig.add_subplot(gs[1], sharex=ax1) 

# ax1.plot(hopeful_model_odds, marker=None, linestyle='-', color='r', label='model probs')
# ax1.fill_between(range(len(win_odds)), win_odds, loss_odds, color='b', alpha=0.5)
# plt.xlim(0, len(win_odds))

# for idx, set_point in enumerate(match_obj.set_points):
#     ax1.axvline(x=set_point, color='gray', linestyle='--', linewidth=1, label=f'End of set {idx}')
#     ax2.axvline(x=set_point, color='gray', linestyle='--', linewidth=1, label=f'End of set {idx}')

# ax1.set_ylabel(f'probability of player {player_to_plot} winning')
# ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
# ax2.set_ylabel(f'importance of point')
# ax1.set_title(f"Probability of player {player_to_plot} winning match at each point")

# ax1.set_ylim(0,1)
# ax1.spines['bottom'].set_visible(False)
# ax2.yaxis.tick_right()  # Move ticks to the right
# ax2.yaxis.set_label_position('right')  # Move label to the right

# # plt.legend()
# plt.tight_layout()
# plt.xlabel('point')
# plt.show()

# -------------

model_probs = [1 - prob if player_to_plot == 1 else prob for prob in match.pre_points_match_probs]
breakpoint()
final_prob = model_probs[-1]
p1_win_odds = get_betting_odds([o1 for p, o1, o2 in ptwinners_and_odds]) 
p2_win_odds = get_betting_odds([o2 for p, o1, o2 in ptwinners_and_odds]) 
point_importances = match.pre_points_importances

win_odds = [elem[player_to_plot] for elem in zip(p1_win_odds, p2_win_odds)] + [final_prob]
loss_odds = [1-elem[1-player_to_plot] for elem in zip(p1_win_odds, p2_win_odds)] + [final_prob]
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 1, hspace=0, height_ratios=[3, 1])  # Zero vertical spacing
ax1 = fig.add_subplot(gs[0])  # Top subplot
ax2 = fig.add_subplot(gs[1], sharex=ax1) 

# Plot model probability line
model_line, = ax1.plot(model_probs, marker=None, linestyle='-', color='r', label='model probability')
# Plot implied betting probability fill
betting_fill = ax1.fill_between(range(len(win_odds)), win_odds, loss_odds, color='b', alpha=0.5, label='implied betting probability')
ax2.plot(point_importances, marker='.', markersize=3, linestyle='', color='black', label='point importance')
plt.xlim(0, len(win_odds))

for idx, set_point in enumerate(match.set_points):
    ax1.axvline(x=set_point, color='gray', linestyle='--', linewidth=1, label=f'End of set {idx}')
    ax2.axvline(x=set_point, color='gray', linestyle='--', linewidth=1, label=f'End of set {idx}')

djokovic_alcaraz = ('djokovic_alcaraz', '12008501', 0, 0.684, 0.679)

ax1.set_ylabel(f'probability of Novak Djokovic winning match')
ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
ax2.set_ylabel(f'importance of point')
ax1.set_title(f"Probability of Novak Djokovic winning match at each point")

ax1.set_ylim(0,1)
ax1.spines['bottom'].set_visible(False)
ax2.yaxis.tick_right()  # Move ticks to the right
ax2.yaxis.set_label_position('right')  # Move label to the right

# Add legend for ax1
handles = [model_line, betting_fill]
labels = ['model probability', 'implied betting probability']
ax1.legend(handles, labels)

plt.tight_layout()
plt.xlabel('point')
plt.show()