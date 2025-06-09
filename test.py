import requests
import json
import re
from tqdm import tqdm
import pandas as pd
import time
from utils.utilities import *
from datetime import datetime, timedelta
from utils.MatchSkeleton import MatchSkeleton
from utils.Match import Match
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

tennisapi_key = "bd1630361980df248336db9408629ee33b483bd4a9995eca4a8e3b0553895bd8"
betsapi_key = "211894-MoC1lB32q2bw1g"
betsapi_key_events = '212216-wWuDctNJQEE9w6'
bets_ended_request_url = f"https://api.b365api.com/v3/events/ended?sport_id=13&token={betsapi_key_events}"
bets_odds_summary_request_url = f"https://api.b365api.com/v2/event/odds/summary?token={betsapi_key_events}"
bets_odds_request_url = f"https://api.b365api.com/v2/event/odds?token={betsapi_key_events}"
bets_upcoming_events_url = f"https://api.b365api.com/v3/events/upcoming?sport_id=13&token={betsapi_key_events}"
bets_ended_events_url = f"https://api.b365api.com/v3/events/ended?sport_id=13&token={betsapi_key_events}"
bets_players_request_url = f"https://api.b365api.com/v1/team?token={betsapi_key_events}&sport_id=13"
bets_tournament_request_url = f"https://api.b365api.com/v1/league?token={betsapi_key_events}&sport_id=13"
bets_event_view_request_url = f"https://api.b365api.com/v1/event/view?token={betsapi_key_events}"

b365_inplay_filter_request_url = f"https://api.b365api.com/v1/bet365/inplay_filter?sport_id=13&token={betsapi_key}"
b365_inplay_event_request_url = f"https://api.b365api.com/v1/bet365/event?token={betsapi_key}"
b365_upcoming_events_url = f"https://api.b365api.com/v1/bet365/upcoming?sport_id=13&token={betsapi_key}"

tournaments_request_url = f"https://api.api-tennis.com/tennis/?method=get_tournaments&APIkey={tennisapi_key}"
fixtures_request_url = f"https://api.api-tennis.com/tennis/?method=get_fixtures&APIkey={tennisapi_key}"
players_request_url = f"https://api.api-tennis.com/tennis/?method=get_players&APIkey={tennisapi_key}"
liveodds_request_url = f'https://api.api-tennis.com/tennis/?method=get_live_odds&APIkey={tennisapi_key}'

player_keys_names_path = 'api_player_ids_names.json'
api_matches_path = 'all_2024_api_matches.json'


grandslam_name_keys = {'ATP Australian Open': '1236',
                       'ATP French Open': '2155',
                       'ATP Wimbledon': '2053',
                       'ATP US Open': '1217'
                       }

with open(player_keys_names_path, 'r') as tourn:
    grandslam_players = json.load(tourn) #list of tuples (key, name)

# csv line template
# 20200103-M-ATP_Cup-RR-Alex_De_Minaur-Alexander_Zverev,207,1,1,5,2,40-30,31,True,1,6f28f1f1f1f2b1f2b+3b1v1f3w#,,,1
# csv headers
# match_id,Pt,Set1,Set2,Gm1,Gm2,Pts,Gm#,TbSet,Svr,1st,2nd,Notes,PtWinner
# csv relevant headers
# match_id, _, _, _, _, _, _, _, TbSet, Svr, _, _, _, PtWinner
# "player_served": "Second Player", "serve_winner": "Second Player", "serve_lost": null,

def generate_unified_matchid_api(match):
    # 20200103-M-ATP_Cup-RR-Alex_De_Minaur-Alexander_Zverev
    date = match['event_date'].replace("-","")
    match['event_first_player'] = match['event_first_player'].replace('&apos;', '')
    match['event_second_player'] = match['event_second_player'].replace('&apos;', '')
    p1_fname = match['event_first_player'][0]
    p1_sname = re.split(r'[ -]', match['event_first_player'])[-1]
    p2_fname = match['event_second_player'][0]
    p2_sname = re.split(r'[ -]', match['event_second_player'])[-1]
    incomplete_matchid = f'{date}-{p1_fname}_{p1_sname}-{p2_fname}_{p2_sname}'.lower()
    return incomplete_matchid

def generate_alternative_unified_matchid_api(match):
    # 20200103-M-ATP_Cup-RR-Alex_De_Minaur-Alexander_Zverev
    date = match['event_date'].replace("-","")
    match['event_first_player'] = match['event_first_player'].replace('&apos;', '')
    match['event_second_player'] = match['event_second_player'].replace('&apos;', '')
    p2_fname = match['event_first_player'][0]
    p2_sname = re.split(r'[ -]', match['event_first_player'])[-1]
    p1_fname = match['event_second_player'][0]
    p1_sname = re.split(r'[ -]', match['event_second_player'])[-1]
    incomplete_matchid = f'{date}-{p1_fname}_{p1_sname}-{p2_fname}_{p2_sname}'.lower()
    return incomplete_matchid

def generate_unified_matchid_charting(match_id):
    # 20200103-M-ATP_Cup-RR-Alex_De_Minaur-Alexander_Zverev
    (date, gender, tournament, round, p1, p2) = match_id.split('-')
    p1_fname = p1.split('_')[0][0]
    p1_sname = p1.split('_')[-1]
    p2_fname = p2.split('_')[0][0]
    p2_sname = p2.split('_')[-1]
    return f'{date}-{p1_fname}_{p1_sname}-{p2_fname}_{p2_sname}'.lower()

def trim_matches_with_no_points(df):
    points = df['pointbypoint']
    bad_idxs = points[points.apply(len) == 0].index
    df.drop(bad_idxs, inplace=True)

def trim_matches_with_badly_formatted_points(df):
    match_points = df['pbp_formatted'].apply(lambda match: [game_points for (svr, game_points) in match])
    error_ids = df.loc[match_points.apply(lambda match: ['ERROR'] in match), 'id_formatted']
    print(f"ID errors: {list(error_ids)}")
    df.drop(error_ids.index, inplace=True)

def get_charting_first_player(charting_df, player_names):
    ((fname_1, sname_1), (fname_2, sname_2)) = player_names
    player1_pattern = rf"^.*{fname_1}.*{sname_1}.*{fname_2}.*{sname_2}$"
    player2_pattern = rf"^.*{fname_2}.*{sname_2}.*{fname_1}.*{sname_1}$"
    match_id = charting_df['match_id'].iloc[0]
    if re.match(player1_pattern,match_id, re.IGNORECASE):
        return (fname_1, sname_1)
    assert re.match(player2_pattern,match_id, re.IGNORECASE), "Neither player is first"
    return (fname_2, sname_2)

def switch_df_columns(df, cols_to_switch):
    rename_dict = {c1:c2 for pair in cols_to_switch for c1,c2 in [pair, pair[::-1]]}
    df.rename(rename_dict, inplace=True)

def switch_score(score):
    score_list = score.split(' - ')[::-1]
    new_score = f"{score_list[0]} - {score_list[1]}"
    return new_score

def switch_pbp_vals(point):
    rename_dict = {'First Player': 'Second Player', 'Second Player': 'First Player'}
    point['player_served'] = rename_dict[point['player_served']]
    point['serve_winner'] = rename_dict[point['serve_winner']]
    point['serve_lost'] = rename_dict[point['serve_lost']] if point['serve_lost'] else None
    if 'TieBreak' in point['set_number']:
        return point
    og_points = point['points']
    for og_point in og_points:
        og_point['score'] = switch_score(og_point['score'])
    return point

def match_up_players(api_match_row, charting_df, player_names):
    true_player1_name = get_charting_first_player(charting_df, player_names)
    api_player1_name = api_match_row['event_first_player']
    true_fname, true_sname = true_player1_name
    true_name_pattern = rf"^{true_fname}.*{true_sname}.*$"
    if not re.match(true_name_pattern, api_player1_name):
        cols_to_switch = [('event_first_player', 'event_second_player'), ('first_player_key', 'second_player_key')]
        # Also inside pointbypoint: player_served (First Player) serve_winner (First Player), serve_lost (First Player), points 
        # Also inside pointbypoint: 'score'
        switch_df_columns(api_match_row, cols_to_switch)
        og_pbp = api_match_row['pointbypoint']
        switched_pbp = [switch_pbp_vals(entry) for entry in og_pbp]
        api_match_row['pointbypoint'] = switched_pbp
    
        api_match_row['id_formatted'] = generate_unified_matchid_api(api_match_row)
        api_match_row['pbp_formatted'] = get_servers_and_ptwinners_from_api_scores(api_match_row)

def ensure_api_charting_agree(api_match_points, charting_df):
    api_unspooled = [[(svr, point) for point in points] for svr,points in api_match_points]
    api_unspooled_flat = [tup for sublist in api_unspooled for tup in sublist]
    charting_unspooled_flat = [(row[0],row[1]) for row in zip(charting_df['Svr'],charting_df['PtWinner'])]
    assert charting_unspooled_flat == api_unspooled_flat

def unified_matchid_change_day(id, delta):
    id_split = id.split('-')
    date_string = id_split[0]
    date_obj = datetime.strptime(date_string, '%Y%m%d').date()
    new_date_string = datetime.strftime(date_obj + timedelta(days=delta), '%Y%m%d')
    id_split[0] = new_date_string
    new_id = '-'.join(id_split)
    return new_id

def generate_match_scores_from_servers_and_ptwinners(formatted_pbp):
    starting_serve = formatted_pbp[0][0]
    ptwinners = [ptwinner for s,ptwinners in formatted_pbp for ptwinner in ptwinners]
    matchSkeleton = MatchSkeleton(starting_serve=starting_serve, sets=3)
    for winner in ptwinners:
        matchSkeleton.point_win(winner-1)
    return matchSkeleton.scores

def time_function(function, function_name='Function', *args):
    s_time = time.time()
    result = function(*args)
    e_time = time.time()
    runtime = e_time - s_time
    print(f'{function_name} Runtime: {runtime:.2f} seconds')
    return result

#Half done
# def convert_betapi_scores_into_ptwinners(score_list):
#     # game score: 6-3,0-1,A-40
#     # tiebreak score: 6-3,7-6,6-7,1-6,6-6,6-6

#     for idx in range(len(score_list)-1):
#         score = score_list[idx]
#         next_score = score_list[idx+1]
#         def get_score_breakdowns(scr):
#             score_split = scr.split(",")
#             return score_split[-1], score_split[-2]
#         game_score, set_score = get_score_breakdowns(score)
#         nxt_game_score, nxt_set_score = get_score_breakdowns(next_score)
#         if set_score == '6-6':
#             # In a tiebreak
#             if nxt_set_score == '6-6':
#                 # 
#         else:
#             # Not in a tiebreak

year = '2024'
gender = 'm'
CONFIG = load_config()
CONFIG = update_git_repos(CONFIG)
# #---------------------------------------------------------------------------------

# with open(api_matches_path, 'r') as file:
#     api_whole_df = pd.read_json(file).T

# api_whole_df['id_formatted'] = api_whole_df.apply(generate_unified_matchid_api, axis=1)
# api_whole_df['id_formatted_alternative'] = api_whole_df.apply(generate_alternative_unified_matchid_api, axis=1)
# trim_matches_with_no_points(api_whole_df)
# assert api_whole_df['id_formatted'].is_unique
# api_whole_df['pbp_formatted'] = api_whole_df.apply(get_servers_and_ptwinners_from_api_scores, axis=1)
# trim_matches_with_badly_formatted_points(api_whole_df)

# #---------------------------------------------------------------------------------

# # 20240127-j_kumstat-r_sakamoto
# # 20240414-k_sell-e_zhu
# # 20241228-s_tsitsipas-p_busta
# # 20241228-s_tsitsipas-p_busta
# player_names = (('s','tsitsipas'), ('p', 'busta'))
# # competition = 'us_open'
# charting_df = get_match_df(CONFIG, gender, year, player_names)
# charting_match_id_unified = generate_unified_matchid_charting(charting_df['match_id'].iloc[0])
# exact_date_charting = charting_df['match_id'].iloc[0].split('-')[0]

# api_formatted_both_options = pd.concat([api_whole_df['id_formatted'], api_whole_df['id_formatted_alternative']]).drop_duplicates()
# api_match_id = find_match_id(api_formatted_both_options, year, player_names)
# exact_date_api = api_match_id.split('-')[0]
# breakpoint()

# #---------------------------------------------------------------------------------

# era = get_era_from_year(year)
# csv_name = f'charting-{gender}-points-{era}.csv'

# path = os.path.join(CONFIG.MATCH_CHARTING_LOCAL, csv_name)
# cols = ['match_id', 'Pt', 'Svr', 'PtWinner']
# charting_whole_df = pd.read_csv(path, encoding='latin1', usecols=cols)
# charting_whole_df.sort_values(by=['match_id', 'Pt'], inplace=True)
# charting_whole_df.drop('Pt', axis=1, inplace=True)

# charting_whole_df = charting_whole_df[charting_whole_df['match_id'].apply(lambda id: id[:4] == '2024')]
# charting_whole_df['id_formatted'] = charting_whole_df['match_id'].apply(generate_unified_matchid_charting)

# # Ground truth for player1 = charting
# api_formatted_both_options = pd.concat([api_whole_df['id_formatted'], api_whole_df['id_formatted_alternative']]).drop_duplicates()
# charting_ids = charting_whole_df['id_formatted'].drop_duplicates()
# # charting ids that aren't in the api
# absent_from_api = charting_ids[~charting_ids.isin(api_formatted_both_options)]
# # charting ids that aren't in the api plus minus 1 day
# charting_ids_m1 = absent_from_api.apply(lambda id: unified_matchid_change_day(id, -1))
# # charting ids that aren't in the api plus 1 day
# charting_ids_p1 = absent_from_api.apply(lambda id: unified_matchid_change_day(id, 1))
# # charting ids with one day added or subtracted that are in the api
# date_altered_charting_ids_in_api_ids = pd.concat([charting_ids_m1[charting_ids_m1.isin(api_formatted_both_options)], charting_ids_p1[charting_ids_p1.isin(api_formatted_both_options)]])

# # charting ids that aren't in api or one day away from an api id
# really_absent_from_api = absent_from_api.drop(date_altered_charting_ids_in_api_ids.index)
# indices_to_drop = api_formatted_both_options.index[api_formatted_both_options.isin(charting_ids)]
# api_ids_reduced = api_formatted_both_options[~api_formatted_both_options.index.isin(indices_to_drop)]
# # api_formatted_both_options_reduced = api_formatted_both_options[]

# for i in range(50):
#     # charting ids that aren't in the api plus minus i days
#     charting_ids_m = really_absent_from_api.apply(lambda id: unified_matchid_change_day(id, -i))
#     # charting ids that aren't in the api plus i days
#     charting_ids_p = really_absent_from_api.apply(lambda id: unified_matchid_change_day(id, i))
#     # charting ids with one day added or subtracted that are in the api
#     date_altered_charting_ids_in_api_ids = pd.concat([charting_ids_m[charting_ids_m.isin(api_ids_reduced)], charting_ids_p[charting_ids_p.isin(api_ids_reduced)]])
#     really_absent_from_api.drop(date_altered_charting_ids_in_api_ids.index, inplace=True)

# breakpoint()
# #---------------------------------------------------------------------------------


# api_match_id = find_match_id(all_matches['id_formatted'], year, player_names, exact_date=exact_date)
# api_match_row = all_matches[all_matches['id_formatted']==api_match_id].iloc[0]
# match_up_players(api_match_row, charting_df, player_names)

# api_match_points = api_match_row['pbp_formatted']
# ensure_api_charting_agree(api_match_points, charting_df)

# e_time = time.time()
# runtime = e_time - s_time
# print(f"Runtime: {runtime:.3f} seconds")
    # for val in all_matches.values():
    #     pbp = val['pointbypoint']
    #     Svr = 

# #---------------------------------------------------------------------------------

# # match_keys = []
# # all_matches = {}
# # for key, name in tqdm(grandslam_players.items()):
# json_data={
#     'date_start': '2025-10-01',
#     'date_stop': '2025-21-01',
#     # 'player_key': key,
#     # 'tournament_key': val
# }

# raw_response = requests.post(fixtures_request_url, data=json_data)
# breakpoint()
# matches = json.loads(raw_response.text)['result']
# # for match in matches:
# #     if match['event_key'] not in match_keys:
# #         match_keys.append(match['event_key'])
# #         all_matches[match['event_key']] = match

# #---------------------------------------------------------------------------------

# key = '1093'
# event_key = '12007629'
# json_data={
#     'date_start': '2025-01-16',
#     'date_stop': '2025-01-16',
#     'player_key': key,
#     'match_key': event_key
#     # 'tournament_key': val
# }

# with open('medvedev_odds.jsonl', 'w') as file:
#     while True:
#         time.sleep(3)
#         raw_response = requests.post(liveodds_request_url, data=json_data)
#         file.writelines(str(raw_response.text))
#         print(raw_response)


# #---------------------------------------------------------------------------------

# regex = re.compile(r'Alcaraz', re.IGNORECASE)
# alcaraz_matches = [game for game in games if regex.search(game['event_first_player']) or regex.search(game['event_second_player'])]

# alcaraz_key = '2382'
# json_data={
#     # 'date_start': '2024-01-01',
#     # 'date_stop': '2024-12-01',
#     'player_key': alcaraz_key,
#     # 'tournament_key': val
# }
# raw_response = requests.post(players_request_url, data=json_data)
# breakpoint()

# #---------------------------------------------------------------------------------

# 20241228-s_tsitsipas-p_busta
# key = '2382'
# json_data={
#     'date_start': '2024-12-25',
#     'date_stop': '2025-12-31',
#     'player_key': key,
#     # 'tournament_key': val
# }

# raw_response = requests.post(fixtures_request_url, data=json_data)
# matches = json.loads(raw_response.text)['result']
# breakpoint()


# #---------------------------------------------------------------------------------

# # 20241228-s_tsitsipas-p_busta
# key = '1906'
# ao_women_id = '10045806'
# draper_vukic_id = '168284582'

# getting_event_stats_params = {
#     'FI': draper_vukic_id,
#     # 'stats': True
# }

# events_info = requests.get(b365_inplay_filter_request_url, params=getting_event_id_params)
# matches = json.loads(events_info.text)['results']
# events_stats = requests.get(b365_inplay_event_request_url, params=getting_event_stats_params)
# match_stats = json.loads(events_stats.text)['results']
# breakpoint()


# #---------------------------------------------------------------------------------

# b365_ao_men_id = '10045480'
# date = '20250116'
# getting_event_id_params={
#     # 'league_id': b365_ao_men_id,
#     'day': date
# }

# events_info = requests.get(bets_ended_request_url, params=getting_event_id_params)
# events_info = json.loads(events_info.text)
# num_of_pages = int(events_info['pager']['total']) // int(events_info['pager']['per_page']) + 1
# matches = []
# for i in range(num_of_pages):
#     getting_event_id_params['page'] = i+1
#     events_info = requests.get(bets_ended_request_url, params=getting_event_id_params)
#     events_info = json.loads(events_info.text)
#     matches += events_info['results']

# tournament_names = [match['league']['name'] for match in matches]
# tournament_names = list(set(tournament_names))
# breakpoint()

# #---------------------------------------------------------------------------------

# bets_ao_men_id = '13657'
# date = '20250116'
# getting_event_id_params={
#     'league_id': bets_ao_men_id,
#     'day': date
# }

# events_info = requests.get(bets_ended_request_url, params=getting_event_id_params)
# events_info = json.loads(events_info.text)
# matches = events_info['results']
# ids = [match['id'] for match in matches]
# breakpoint()

# #---------------------------------------------------------------------------------

# # medvedev_tien_id = '9373908'
# bets_ao_men_id = '13657'
# date = '20250116'
# getting_event_id_params={
#     'league_id': bets_ao_men_id,
#     'day': date
# }

# events_info = requests.get(bets_ended_request_url, params=getting_event_id_params)
# events_info = json.loads(events_info.text)
# matches = events_info['results']
# ids = [match['id'] for match in matches]
# for match_id in ids:
#     getting_event_id_params = {
#         'event_id': match_id
#     }

#     events_info = requests.get(bets_odds_summary_request_url, params=getting_event_id_params)
#     events_info = json.loads(events_info.text)
#     odds_summary = events_info['results']
#     num_of_intervals = max([len(os['odds'].keys()) for os in odds_summary.values()])
#     print(f"event_id: {match_id}; num_of_intervals: {num_of_intervals}")
# # platforms = list(odds_summary.keys())
# # assert 'Bet365' in platforms
# # b365_os = odds_summary['Bet365']
# breakpoint()

# #---------------------------------------------------------------------------------

# 20241228-s_tsitsipas-p_busta
# player_key = '2072'
# match_key = '12007629'
# tournament_key_2 = '13657'
# json_data={
#     'date_start': '2025-01-20',
#     'date_stop': '2025-01-20',
#     'player_key': player_key,
#     # 'match_key': match_key,
#     # 'tournament_key': tournament_key_2
# }

# raw_response = requests.post(fixtures_request_url, data=json_data)
# matches = json.loads(raw_response.text)['result']
# match = matches[0]
# formatted_pbp = get_servers_and_ptwinners_from_api_scores(match)

# breakpoint()

# #---------------------------------------------------------------------------------

# b365_ao_men_id = '10045480'
# bets_ao_men_id = '13657'
# id_finder_params = {
#     'league_id': b365_ao_men_id
# }

# raw_response = requests.get(b365_inplay_filter_request_url, params=id_finder_params)
# match_info = json.loads(raw_response.text)
# if match_info['results']:
#     match_info = match_info['results']
# breakpoint()

# #---------------------------------------------------------------------------------

# b365_ao_men_id = '10045480'
# bets_ao_men_id = '13657'
# b365_l_id = '10045987'
# minaur_key = '1106'
# json_data={
#     # 'date_start': '2025-01-21',
#     # 'date_stop': '2025-01-21',
#     # 'player_key': minaur_key,
#     # 'match_key': michelsen_minaur_match_key
#     'league_id': b365_ao_men_id
# }

# raw_response = requests.get(b365_upcoming_events_url, params=json_data)
# matches = json.loads(raw_response.text)['results']
# breakpoint()

# #---------------------------------------------------------------------------------


# moutet_tien_id = '168340563'
# rune_kecmanovic_id = '168324554'
# draper_alcaraz_id = '168388298'
# kawagashi_simpson_id = '168456460'
# boscardin_lavallen_id = '168459268'
# sinner_rune_id = '168442289'
# michelsen_minaur_id = '168427642'
# paul_zverev_id = '168481850'
# djokovic_alcaraz_id = '168485145'
# temp_id = '168350444'
# inplay_params = {
#     'FI': djokovic_alcaraz_id
# }

# match_name = 'djokovic_alcaraz'
# try:
#     os.makedirs(match_name)
#     print(f'Directory {match_name} created, it will now be populated')
# except:
#     # if len(os.listdir(match_name)) > 0:
#     #     raise FileExistsError(f'Directory {match_name} already exists and is populated')
#     # else:
#     print(f'Directory {match_name} already exists, it will now be populated')
# score = 'not started'
# idx = 0
# start_time = datetime.strptime('2025-01-21-09-25', '%Y-%m-%d-%H-%M')
# if start_time > datetime.now():
#     sleeptime = start_time - datetime.now()
#     print(f"sleeping for {sleeptime.seconds} seconds")
#     time.sleep(sleeptime.seconds)
# while True:
#     time.sleep(3)
#     if datetime.now() < start_time:
#         print('event not started yet')
#         continue
#     print('requesting odds...')
#     raw_response = requests.get(b365_inplay_event_request_url, params=inplay_params)
#     match_info = json.loads(raw_response.text)
#     if match_info and 'results' in match_info.keys() and match_info['results']:
#         match_info = match_info['results'][0]
#         score_element = [match for match in match_info if 'type' in match.keys() and match['type']=='EV'][0]
#         cur_score = score_element['XP']
#         if cur_score == score:
#             continue
#         score = cur_score
#         # element = [match for match in match_info if 'IT' in match.keys() and match['IT']=='6VP168361880-2075831417_1_1'][0]
#         # odds = element['OD']
#         print(f"{score}")
#         file = open(f'{match_name}/inplay_info_{idx}.json', 'w')
#         file_contents = str(match_info).replace("'", '"')
#         file.write(file_contents)
#         file.close()
#         idx += 1


# #---------------------------------------------------------------------------------

# def setup():
#     match_name = 'michelsen_minaur'
#     # rune_odds_id = '6VP168491741-2094249044_1_1'
#     # sinner_odds_id = '6VP168491741-2094249052_1_1'
#     files = os.listdir(match_name)
#     files = [file for file in files if 'json' in file]
#     relevant_file_idxs_path = f'{match_name}/relevant_file_idxs.csv'
#     if os.path.exists(relevant_file_idxs_path):
#         with open(relevant_file_idxs_path, 'r') as file:
#             relevant_file_idxs = list(csv.reader(file))[0]
#             files = [file for file in files if file[12:-5] in relevant_file_idxs]
#     return files, match_name, relevant_file_idxs_path

# def forloop(files, match_name):
#     scores = []
#     first_idxs = []
#     odds = []
#     for file in files:
#         match_info = pd.read_json(f'{match_name}/{file}', encoding='latin-1')
#         important_info = match_info.iloc[0]
#         match_odds_id = int(match_info.loc[match_info['NA']=='Match', 'ID'].iloc[0])
#         g_score = important_info['XP']
#         s_score = important_info['SS']
#         score = f'{s_score},{g_score}'
#         p0,p1 = important_info['NA'].split(' v ')
#         if score not in scores:
#             # print(f'{s_score},{g_score}')
#             scores.append(score)
#             idx = file[12:-5]
#             first_idxs.append(idx)
#             match_odds_idxs = match_info[match_info['MA']==match_odds_id].index
#             assert match_odds_idxs.shape[0] == 2
#             match_info.loc[match_odds_idxs, 'NA'] = match_info['NA'].shift().loc[match_odds_idxs]
#             odds.append((match_info.loc[(match_info['MA'] == match_odds_id) & (match_info['NA'] == p0), 'OD'].iloc[0], match_info.loc[(match_info['MA'] == match_odds_id) & (match_info['NA'] == p1), 'OD'].iloc[0]))
#     return scores, odds, match_name, first_idxs

# def finish(relevant_file_idxs_path, scores, odds, match_name, first_idxs):
#     scores_and_odds = [(s,o1,o2) for s,(o1,o2) in zip(scores, odds)]
#     sorted_so = sorted(scores_and_odds, key=lambda row: row[0])
#     with open(f'{match_name}/scores_and_odds_mapping.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Score', 'Odds1', 'Odds2'])
#         writer.writerows(sorted_so)
#     with open(relevant_file_idxs_path,'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(first_idxs)

# files, match_name, relevant_file_idxs_path = time_function(setup, 'setup')
# scores, odds, match_name, first_idxs = time_function(forloop, 'forloop', files, match_name)
# time_function(finish, 'finish', relevant_file_idxs_path, scores, odds, match_name, first_idxs)

# #---------------------------------------------------------------------------------

# # 20241228-s_tsitsipas-p_busta
# player_key = '1093'
# boscardin_lavallen_match_key = '12008460'
# sinner_rune_match_key = '12008434'
# tournament_key_1 = '8895'
# tournament_key_2 = '2809'
# match_name = 'michelsen_minaur'
# michelsen_minaur_match_key = '12008400'
# json_data={
#     # 'date_start': '2025-01-19',
#     # 'date_stop': '2025-01-19',
#     # 'player_key': key,
#     'match_key': michelsen_minaur_match_key
#     # 'tournament_key': tournament_key_2
# }

# raw_response = requests.post(fixtures_request_url, data=json_data)
# match = json.loads(raw_response.text)['result'][0]
# formatted_pbp = get_servers_and_ptwinners_from_api_scores(match)
# just_points = [point for s, points in formatted_pbp for point in points]
# first_server, _ = formatted_pbp[0]
# all_scores = generate_match_scores_from_servers_and_ptwinners(formatted_pbp)
# odds_file_path = f'{match_name}/scores_and_odds_mapping.csv'
# with open(odds_file_path, 'r') as file:
#     scores_and_odds = list(csv.reader(file))
# score_odds_dict = {
#     score: (odd1, odd2) for score, odd1, odd2 in scores_and_odds
# }
# all_scores_and_odds = [(point, score, *score_odds_dict[score]) if score in score_odds_dict.keys() else (point, score, None, None) for point, score in zip(just_points, all_scores)]
# odds_throughout_match_path = f'{match_name}/odds_throughout_match.csv'
# with open(odds_throughout_match_path, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['PtWinner', 'Score', 'Odds1', 'Odds2'])
#     writer.writerows(all_scores_and_odds)


# #---------------------------------------------------------------------------------


# sinner_rune_match_key = '12008434'
# match_name = 'sinner_rune'
# michelsen_minaur_match_key = '12008400'
# json_data={
#     # 'date_start': '2025-01-19',
#     # 'date_stop': '2025-01-19',
#     # 'player_key': key,
#     'match_key': sinner_rune_match_key
#     # 'tournament_key': tournament_key_2
# }

# raw_response = requests.post(fixtures_request_url, data=json_data)
# match = json.loads(raw_response.text)['result'][0]
# formatted_pbp = get_servers_and_ptwinners_from_api_scores(match)
# first_server, _ = formatted_pbp[0]
# odds_throughout_match_path = f'{match_name}/odds_throughout_match.csv'
# with open(odds_throughout_match_path, 'r') as file:
#     reader = csv.reader(file)
#     next(reader)
#     ptwinners_and_odds = [(int(ptwinner)-1,odd1,odd2) for ptwinner,_,odd1, odd2 in reader]
# match = Match(3, first_server-1, 0.712, 0.66)
# pts = [p for p,o1,o2 in ptwinners_and_odds]
# match.simulate_match_by_point(pts)

# player_to_plot = 0
# model_probs = [1 - prob if player_to_plot == 1 else prob for prob in match.pre_points_match_probs]
# def get_betting_odds(odds):
#     win_probs = []
#     for odd in odds:
#         if odd:
#             win_probs.append(odd)
#             break
#     for odd in odds:
#         if odd:
#             win_probs.append(odds_string_to_probability(odd))
#         else:
#             win_probs.append(win_probs[-1])
#     win_probs = win_probs[1:]
#     return win_probs
# p1_win_odds = get_betting_odds([o1 for p, o1, o2 in ptwinners_and_odds])
# p2_win_odds = get_betting_odds([o2 for p, o1, o2 in ptwinners_and_odds])

# win_odds = [elem[player_to_plot] for elem in zip(p1_win_odds, p2_win_odds)]
# loss_odds = [1-elem[1-player_to_plot] for elem in zip(p1_win_odds, p2_win_odds)]
# plt.plot(model_probs, marker=None, linestyle='-', color='r', label='model probs')
# plt.fill_between(range(len(p1_win_odds)), win_odds, loss_odds, color='b', alpha=0.5)

# for idx, set_point in enumerate(match.set_points):
#     plt.axvline(x=set_point, color='gray', linestyle='--', linewidth=1, label=f'End of set {idx}')

# plt.ylim(0,1)
# plt.legend()
# plt.xlabel('point')
# plt.ylabel(f'probability of player {player_to_plot} winning')
# plt.title(f"Probability of player {player_to_plot} winning match at each point")
# plt.show()


# #---------------------------------------------------------------------------------

# # 370 points in total
# medvedev_tien_id = '9373908'
# getting_event_id_params = {
#     'event_id': medvedev_tien_id
# }
# events_info = requests.get(bets_odds_request_url, params=getting_event_id_params)
# events_info = json.loads(events_info.text)
# info = events_info['results']
# odds = info['odds'] # keys = '13_1', '13_4' 13 probably represents tennis, 1 and 4 ??
# stats = info['stats']
# pd_game_odds = pd.DataFrame([odd for odds_section in odds.values() for odd in odds_section if odd['ss']])
# pd_game_odds.drop_duplicates(subset=['ss'],inplace=True) #this gets 350 out of 370 points (i think)
# pd_game_odds.sort_values(by='add_time', inplace=True) 
# pd_game_odds.to_csv('medvedev_tien_odds.csv', index=True)

# #---------------------------------------------------------------------------------

# 370 points in total

# with open('bets_tournament_info_filtered.json', 'r') as file:
#     filtered_tournaments = json.load(file)
# num_of_tournaments = len(filtered_tournaments)
# starting_page = 1
# starting_match = 0
# starting_tournament = 0
# tournament_ids = [t['id'] for t in filtered_tournaments]
# for t_idx in range(starting_tournament, len((tournament_ids))):
#     t_id = tournament_ids[t_idx]
#     odds_dir_name = f'{t_id}_matches_betting_odds'
#     bets_tournament_id = t_id
#     try:
#         os.makedirs(odds_dir_name)
#         print(f'Directory {odds_dir_name} created, it will now be populated')
#     except:
#         print(f'Directory {odds_dir_name} already exists, it will now be populated') 
#     getting_event_ids = {
#         'league_id': bets_tournament_id
#     }
#     events_info = requests.get(bets_ended_events_url, params=getting_event_ids)
#     events_info = json.loads(events_info.text)
#     num_of_pages = (events_info['pager']['total'] // events_info['pager']['per_page']) + 1
#     match_infos = []

#     for i in range(starting_page,num_of_pages+1):
#         getting_event_ids['page'] = i
#         events_info = requests.get(bets_ended_events_url, params=getting_event_ids)
#         info = json.loads(events_info.text)['results']
#         match_infos = [(match['id'], match['home']['id'], match['away']['id']) for match in info]
#         print(f'Got {len(match_infos)} matches')
#         match_id_names = []
#         for match in match_infos:
#             name = f"{match[1]}_{match[2]}"
#             match_id_names.append((match[0],name))

#         for idx in range(starting_match, len(match_id_names)):
#             time.sleep(2)
#             print(f'page: {i}/{num_of_pages}; match: {idx+1}/{len(match_id_names)}; tournament: {t_idx+1}/ {num_of_tournaments}')
#             id, name = match_id_names[idx]
#             getting_event_id_params = {
#                 'event_id': id
#             }
#             events_info = requests.get(bets_odds_request_url, params=getting_event_id_params)
#             events_info = json.loads(events_info.text)
#             info = events_info['results']
#             odds = info['odds'] # keys = '13_1', '13_4' 13 probably represents tennis, 1 and 4 ??
#             stats = info['stats']
#             pd_game_odds = pd.DataFrame([odd for odds_section in odds.values() for odd in odds_section if odd['ss']])
#             if pd_game_odds.empty:
#                 continue
#             pd_game_odds.drop_duplicates(subset=['ss'],inplace=True)
#             pd_game_odds.sort_values(by='add_time', inplace=True) 
#             pd_game_odds.to_csv(f'{odds_dir_name}/{id}_{name}.csv', index=True)
#         starting_match = 0
#     starting_page = 0


# #---------------------------------------------------------------------------------

# odds_root_dir = 'matches_betting_odds'
# stats_root_dir = 'matches_stats'
# tournaments = sorted(os.listdir(odds_root_dir))
# starting_tourn = 0
# starting_match = 0
# for i in range(starting_tourn, len(tournaments)):
#     tourn = tournaments[i]
#     tourn_id = tourn.split('_')[0]
#     match_files = sorted(os.listdir(f'{odds_root_dir}/{tourn}'))
#     try:
#         os.makedirs(f'{stats_root_dir}/{tourn_id}_matches_stats')
#     except:
#         pass
#     for j in range(starting_match, len(match_files)):
#         match_file = match_files[j]
#         match_id = match_file.split('_')[0]
#         players = '_'.join(match_file.split('_')[1:])[:-4]
#         raw_response = requests.get(bets_event_view_request_url, params={'event_id':match_id})
#         event = json.loads(raw_response.text)['results'][0]
#         time.sleep(2)
#         if 'stats' in event.keys() and len(event['stats']) >= 4:
#             with open(f'{stats_root_dir}/{tourn_id}_matches_stats/{match_id}_{players}.json', 'w') as file:
#                 json.dump(event['stats'], file, indent=4)
#         print(f'match: {j}/{len(match_files)}; tournament: {i}/{len(tournaments)}')
#     starting_match = 0

# #---------------------------------------------------------------------------------

# from utils.Match import Match
# from utils.utilities import *


michelsen_minaur = ('michelsen_minaur', '12008400', 1, 0.634, 0.641)
sinner_rune = ('sinner_rune', '12008434', 0, 0.712, 0.66)
djokovic_alcaraz = ('djokovic_alcaraz', '12008501', 0, 0.684, 0.679)
match_name, match_key, player_to_plot, p0_serve_odds, p1_serve_odds = djokovic_alcaraz

bets_match_key = get_bets_match_key_from_api_match_key(match_key)
match = get_match_info_locally(bets_match_key, 'matches_betting_odds', 'csv')
match_obj = Match(3, 0, p0_serve_odds, p1_serve_odds)
# state = match.get_state_from_score('6-4,6-7,6-6,13-12', False)
# bets_match_key
match['state'] = match['ss'].apply(lambda elem: match_obj.get_state_from_score(elem, False))
match['progression_score'] = match['state'].apply(match_progression_score_from_states)
match.sort_values(by=['progression_score', 'add_time'], inplace=True)
match