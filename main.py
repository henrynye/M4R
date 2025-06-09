from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.environ['WORKSPACE_PATH'])
import pandas as pd
from utils.utilities import *
from utils.Match import Match
import numpy as np
import matplotlib.pyplot as plt
import timeit

CONFIG = load_config()
CONFIG = update_git_repos(CONFIG)

gender = 'm'

match = get_match_df(CONFIG, gender, '2024', ('alcaraz', 'zverev'), 'french_open')
pts = list(match['PtWinner'])
first_server = match['Svr'].iloc[0] - 1
pts = [pt - 1 for pt in pts]
points0 = match[match['Svr']==1]
points1 = match[match['Svr']==2]
p0 = points0[points0['PtWinner']==1].shape[0] / points0.shape[0]
p1 = points1[points1['PtWinner']==2].shape[0] / points1.shape[0]
print(f'p0: {p0}')
print(f'p1: {p1}')
# p0 = 0.7
# p1 = 0.7
match = Match(prob_of_0_winning_service_point=p0, prob_of_1_winning_service_point=p1, starting_serve=first_server, num_of_sets=3)

# reps = 1000
# exec_time = timeit.timeit(first_match.generate_game_probability_matrix_analytically_negbin, number=reps)
# av_time = exec_time / reps
# print(f'time taken: {av_time:.8f} seconds')
# first = first_match.generate_game_probability_matrix()
# second = first_match.generate_game_probability_matrix_analytically()
# third = first_match.generate_game_probability_matrix_analytically_negbin()

# assert np.array_equal(first, second)
# assert np.array_equal(first, third)
match.simulate_match_by_point(pts)
points = match.pre_points_match_probs
print(points[-10:])
plt.plot(points, marker=None, linestyle='-', color='b', label='points')

for idx, set_point in enumerate(match.set_points):
    plt.axvline(x=set_point, color='gray', linestyle='--', linewidth=1, label=f'End of set {idx}')

plt.ylim(0,1)
plt.legend()
plt.xlabel('point')
plt.ylabel('probability of player 0 winning')
plt.title("Probability of player 0 winning match at each point")
plt.show()
