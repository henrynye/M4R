import shutil
import os
import yaml
from git import Repo
from datetime import datetime
from utils.config import Config
from datetime import datetime
import re
import pandas as pd
import json
import csv

tennisapi_key = "bd1630361980df248336db9408629ee33b483bd4a9995eca4a8e3b0553895bd8"
betsapi_key = "211894-MoC1lB32q2bw1g"
bets_ended_request_url = f"https://api.b365api.com/v3/events/ended?sport_id=13&token={betsapi_key}"
bets_odds_summary_request_url = f"https://api.b365api.com/v2/event/odds/summary?token={betsapi_key}"
bets_odds_request_url = f"https://api.b365api.com/v2/event/odds?token={betsapi_key}"
bets_upcoming_events_url = f"https://api.b365api.com/v3/events/upcoming?sport_id=13&token={betsapi_key}"

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

def load_config():
    # Load the YAML configuration file
    with open('config.yml', 'r') as file:
        config_dict = yaml.safe_load(file)

    CONFIG = Config(config_dict)
    return CONFIG

def backup_file(backup_dir, file_path):
    shutil.copy(file_path, os.path.join(backup_dir,file_path))

def update_config(CONFIG, key, new_val):
    CONFIG[key] = new_val
    with open('config.yml', 'w') as file:
        yaml.safe_dump(CONFIG, file, default_flow_style=False)
    return load_config()

def reset_git_dir(dir_path):
    cloned_repo = Repo(dir_path)
    print("Fetching updates...")
    cloned_repo.remotes.origin.fetch()
    cloned_repo.git.reset('origin', hard=True)

def clone_git_dir(dir_url, dir_local_path):
    Repo.clone_from(dir_url, dir_local_path)

def update_git_repos(CONFIG):
    match_charting_last_cloned = CONFIG.MATCH_CHARTING_LAST_CLONED
    current_date = datetime.now().date()
    if current_date > match_charting_last_cloned:
        reset_git_dir(CONFIG.MATCH_CHARTING_LOCAL)
        backup_file(CONFIG.BACKUP_PATH, 'config.yml')
        CONFIG = update_config(CONFIG.get_dict(), 'MATCH_CHARTING_LAST_CLONED', current_date)
    return CONFIG

def get_date_from_match_id(match_id):
    date_part = re.match(r"([^-]+)-", match_id).group(1)
    print(date_part)
    date_obj = datetime.strptime(date_part, '%Y%m%d')
    return date_obj

def find_match_id(match_ids, year, player_names, competition=None, exact_date=None):
    # match_id = f'{year}0130-M-{competition}-F-{player_surnames[0]}-{player_surnames[1]}'
    # match_id = f'{year}0130-M-{competition}-F-{player_surnames[1]}-{player_surnames[0]}'
    ((fname_1, sname_1), (fname_2, sname_2)) = player_names
    if not (exact_date or competition):
        print("As you have not entered either competition or exact date, the match_id is ambiguous. The first one found is returned.")
    date = exact_date if exact_date else year
    competition = '' if not competition else competition
    pattern = rf"^{date}.*{competition}.*{fname_1}.*{sname_1}.*{fname_2}.*{sname_2}$|^{date}.*{competition}.*{fname_2}.*{sname_2}.*{fname_1}.*{sname_1}$"
    ids_matched = []
    for match_id in match_ids:
        if re.match(pattern,match_id, re.IGNORECASE):
            ids_matched.append(match_id)
    print(f'{len(list(set(ids_matched)))} matches found')
    if ids_matched:
        return ids_matched[0]
    return None

def get_match_df(CONFIG, gender, year, player_names, competition=None, exact_date=None):
    era = get_era_from_year(year)
    csv_name = f'charting-{gender}-points-{era}.csv'

    path = os.path.join(CONFIG.MATCH_CHARTING_LOCAL, csv_name)
    cols = ['match_id', 'Pt', 'Set1', 'Set2', 'Gm1', 'Gm2', 'TbSet',
        'Svr', 'PtWinner']
    dtype_map = {'TbSet': 'object'}
    df = pd.read_csv(path, encoding='latin1', usecols=cols, dtype=dtype_map)

    bad_matches = df.loc[df['TbSet'].isna(), 'match_id'].drop_duplicates()
    for id in bad_matches:
        modal_value = df.loc[df['match_id']==id, 'TbSet'].mode().iloc[0]
        df.loc[df['match_id']==id, 'TbSet'] = modal_value

    match_id = find_match_id(df['match_id'], year, player_names, competition, exact_date)
    assert match_id is not None
    match_df = df[df['match_id'] == match_id].sort_values(by='Pt')
    return match_df

def get_era_from_year(year):
    if int(year) >= 2020:
        return '2020s'
    if int(year) >= 2010:
        return '2010s'
    return 'to-2009'

def convert_num_to_tennis_points(num):
    return ['0', '15', '30', '40', 'A'][num]

def get_servers_and_ptwinners_from_api_scores(match):
    # match_id, _, _, _, _, _, _, _, TbSet, Svr, _, _, _, PtWinner
    pbp = match['pointbypoint']
    scores = [(get_server_number_from_api_player_served(game['player_served']),convert_game_scores_into_ptwinners(game)) for game in pbp]
    scores = [(server, pts) for (server, pts) in scores if pts]
    return scores 

def get_server_number_from_api_player_served(player_served):
    Svr = 2 if player_served == 'Second Player' else 1
    return Svr

def odds_string_to_probability(odds):
    num, den = map(float, odds.split('/'))
    if num + den == 0:
        return 0
    prob = den/(num+den)
    return round(prob, 3)

equivalents = {}

def add_equivalents(pairs):
    for pair in pairs:
        equivalents[pair[0]] = pair[1]
        equivalents[pair[1]] = pair[0]

add_equivalents([('event_first_player', 'First Player'), ('event_second_player', 'Second Player')])
add_equivalents([
    (0, '0'),
    (1, '15'),
    (2, '30'),
    (3, '40'),
    (4, 'A')
])

def get_equivalent(word):
    return equivalents[word]

def convert_game_scores_into_ptwinners(game):
    if 'TieBreak' in game['set_number']:
       return [get_server_number_from_api_player_served(game['serve_winner'])]
    game_scores = [point['score'] for point in game['points']]
    pts = [score.split(' - ') for score in game_scores]
    if game_scores == [' - ']:
        return []
    for pt1, pt2 in pts:
        try:
            get_equivalent(pt1)
        except KeyError:
            return ['ERROR']
    number_pts = [(get_equivalent(pt1), get_equivalent(pt2)) for pt1, pt2 in pts]
    ptwinners = []
    ptwinners.append(2-number_pts[0][0])
    for idx in range(0,len(number_pts)-1):
        cur_pts = number_pts[idx]
        nxt_pts = number_pts[idx+1]
        ptwinners.append(1 if nxt_pts[0]>cur_pts[0] or nxt_pts[1]<cur_pts[1] else 2)
    ptwinners.append(1 if number_pts[-1][0] > number_pts[-1][1] else 2)
    return ptwinners

def get_match_info_locally(match_key, root_dir, file_extension):
    pattern = rf'{match_key}.*'
    for file in os.listdir(root_dir):
        path = f'{root_dir}/{file}'
        for match_file in os.listdir(path):
            if re.match(pattern, match_file):
                path = f'{path}/{match_file}'
                if file_extension == 'json':
                    with open(path, 'r') as file:
                        match_info = json.load(file)
                elif file_extension == 'csv':
                    match_info = pd.read_csv(path)
    return match_info

def get_bets_match_key_from_api_match_key(api_match_key):
    with open('api_m_keys_to_bets_api_m_keys.json', 'r') as file:
        m_key_map = json.load(file)
    return m_key_map[api_match_key]

def match_progression_score_from_states(states, tiebreak_length=7):
    sets_played = sum(int(s['sets']) for s in states)
    games_played_in_set = sum(int(s['games']) for s in states)
    points_played_in_game = sum(int(s['points']) for s in states)
    if games_played_in_set == 12:
        points_played_in_game = min((tiebreak_length-1)*2, points_played_in_game)
    else:
        points_played_in_game = min(4, points_played_in_game)
    multiplier = max(tiebreak_length*2, 13)
    return points_played_in_game + multiplier*games_played_in_set + (multiplier**2)*sets_played

def player_progression_score_from_states(states, player, tiebreak_length=7):
    sets_won = int(states[player]['sets'])
    games_won_in_set = int(states[player]['games'])
    points_won_in_game = int(states[player]['points'])
    games_played_in_set = sum(int(s['games']) for s in states)
    if games_played_in_set == 12:
        points_won_in_game = min((tiebreak_length-1)*2, points_won_in_game)
    else:
        points_won_in_game = min(4, points_won_in_game)
    multiplier = max(tiebreak_length*2, 13)
    return points_won_in_game + multiplier*games_won_in_set + (multiplier**2)*sets_won

def are_odds_swapped(ss1, ss2):
    if not ss1['swap']:
        home_ods1, away_ods1 = float(ss1['home_od']), float(ss1['away_od'])
    else:
        home_ods1, away_ods1 = float(ss1['away_od']), float(ss1['home_od'])
    home_ods2, away_ods2 = float(ss2['home_od']), float(ss2['away_od'])

    return (abs(home_ods2 - 1/ss2['model_odds']) > abs(away_ods2 - 1/ss2['model_odds'])) and (abs(home_ods1 - away_ods2) < abs(home_ods1 - home_ods2)) and (abs(away_ods1 - home_ods2) < abs(away_ods1 - away_ods2)) 

## THIS IS SO WRONG
# def score_from_state(state):
#     sets = '-'.join([str(s['sets']) for s in state])
#     games = '-'.join([str(s['games']) for s in state])
#     points = '-'.join([str(s['points']) for s in state])
#     return ','.join([sets, games, points])