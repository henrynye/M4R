import json
import pandas as pd
import os
from bs4 import BeautifulSoup
import re
from tqdm import tqdm

ta_root_dir = 'tennis_abstract'
ta_df_dir = 'tennis_abstract_dfs'
matchhead_verbose = ["date","tourn","surf","level","win/loss","rank","seed","entry","round",
                 "score","max_num_of_sets","opp","orank","oseed","oentry","ohand","obday",
                 "oheight","ocountry","oactive","time_minutes","aces","dfs","service_pts","first_serves_in","first_serves_won",
                 "second_serves_won",'service_games',"break_points_saved","break_points_faced","oaces","odfs","oservice_pts","ofirst_serves_in","ofirst_serves_won",
                 "osecond_serves_won",'oservice_games',"obreak_points_saved","obreak_points_faced", "obackhand", "chartlink",
                 "pslink","whserver","matchid","wh","roundnum","matchnum", "oforeign_key"]

def gen_df(name):
    html_path = f'{ta_root_dir}/{name}.html'
    assert os.path.exists(html_path)
    with open(html_path, 'r') as file:
        html_content = file.read()
    soup = BeautifulSoup(html_content, "html.parser")
    script_tags = soup.find_all("script")
    matchmx = None
    for script in script_tags:
        if script.string:  # Ensure the script tag contains JavaScript code
            match = re.search(r'var\s+matchmx\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
            if match:
                # Extract the JavaScript array or object
                ochoices_str = match.group(1)
                # Convert the JavaScript array/object to a Python object
                matchmx = eval(ochoices_str)
    assert matchmx is not None
    if len(matchmx[0])== 47:
        matchmx = [m+[''] for m in matchmx]
    try:
        match_df = pd.DataFrame(matchmx, columns=matchhead_verbose)
    except Exception as e:
        breakpoint()
    return match_df

# name = 'CarlosAlcaraz'
# alcaraz_html_path = f'{ta_root_dir}/{name}.html'
# with open(alcaraz_html_path, 'r') as file:
#     html_content = file.read()
# soup = BeautifulSoup(html_content, "html.parser")
# script_tags = soup.find_all("script")
# ochoices = None
# for script in script_tags:
#     if script.string:  # Ensure the script tag contains JavaScript code
#         match = re.search(r'var\s+ochoices\s*=\s*(\[.*?\]);', script.string, re.DOTALL)
#         if match:
#             # Extract the JavaScript array or object
#             ochoices_str = match.group(1)
#             # Convert the JavaScript array/object to a Python object
#             ochoices = eval(ochoices_str)
top_150_names = []
for file in tqdm(os.listdir(ta_df_dir)):
    name = file[:-4]
    file_name = f'{ta_df_dir}/{file}'
    p_df = pd.read_pickle(f'{ta_df_dir}/{file}')
    latest_rank = p_df['rank'].iloc[-1]
    if latest_rank:
        latest_rank = int(latest_rank)
        if latest_rank < 150:
            top_150_names.append(name)
# ta_names = [v.replace(' ', '') for v in ochoices]
for name in tqdm(top_150_names):
    df = gen_df(name)
    df.to_pickle(f'{ta_df_dir}/{name}.pkl')


