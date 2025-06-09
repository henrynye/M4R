import numpy as np
import math
import scipy.integrate as integrate
from utils.utilities import convert_num_to_tennis_points

class Match:
    def __init__(self, num_of_sets = 2, starting_serve = 0, prob_of_0_winning_service_point=0.5, prob_of_1_winning_service_point=0.5, tiebreak_length = 7):
        self.states = [{
            'sets': 0,
            'games': 0, 
            'points': 0, 
        },
        {
            'sets': 0,
            'games': 0, 
            'points': 0, 
        }
        ]
        self.currently_serving = starting_serve
        self.starting_serve = starting_serve
        self.num_of_sets = num_of_sets
        self.match_winner = None
        self.in_tiebreak = False
        self.tiebreak_length = tiebreak_length
        self.probs_of_winning_point = [prob_of_0_winning_service_point, 1-prob_of_1_winning_service_point]
        self.inner_tiebreak_probabilities = self.generate_tiebreak_probability_matrix()
        self.probs_of_0_winning_tiebreak = self.inner_tiebreak_probabilities[0,0,0] #TODO: Maybe solve this analytically
        self.inner_game_probabilities = self.generate_game_probability_matrix()
        self.inner_set_probabilities = self.generate_set_probability_matrix()
        self.probs_of_0_winning_set = self.inner_set_probabilities[0,0,0] #TODO: Maybe solve this analytically
        self.inner_match_probabilities = self.generate_match_probability_matrix()
        self.set_starting_server = starting_serve
        self.pre_points_match_probs = []
        self.pre_points_importances = []
        self.historical_states = []
        self.set_points = []
        self.update_probs_of_winning_set()
        self.update_probs_of_winning_match()

    def update_probs_of_winning_set(self):
        self.prob_of_winning_set_given_cur_game_win, self.prob_of_winning_set_given_cur_game_loss = self.get_probs_of_winning_set(self.states, self.set_starting_server)

    def update_probs_of_winning_match(self):
        self.prob_of_winning_match_given_cur_set_win, self.prob_of_winning_match_given_cur_set_loss = self.get_probs_of_winning_match(self.states)

    def get_probs_of_winning_set(self, states, set_starting_server):
        return self.inner_set_probabilities[set_starting_server, states[0]['games']+1, states[1]['games']], self.inner_set_probabilities[set_starting_server, states[0]['games'], states[1]['games']+1]

    def get_probs_of_winning_match(self, states):
        if states[0]['sets'] == self.num_of_sets:
            return 1, 1
        if states[1]['sets'] == self.num_of_sets:
            return 0, 0
        return self.inner_match_probabilities[states[0]['sets']+1, states[1]['sets']], self.inner_match_probabilities[states[0]['sets'], states[1]['sets']+1]

    def populate_prob_matrix(self, starting_p, max_units_played, max_units, mat, mode, alternative_p=0.5, deuce=True):
        assert mat.shape[0] == mat.shape[1]
        deuce_number = max_units if deuce else mat.shape[0]+1
        mat[max_units+1,:deuce_number] = 1
        mat[:deuce_number, max_units+1] = 0
        for units_played in range(max_units_played, -1, -1):
            upper_bound = min(units_played, max_units)
            lower_bound = max(0, units_played - max_units)
            for a_score in range(lower_bound, upper_bound+1):
                b_score = units_played - a_score
                p_wins = (a_score+1, b_score)
                p_loses = (a_score, b_score+1)
                current_p = starting_p if (
                    (mode in ['sets', 'points']) or 
                    (mode == 'games' and (a_score+b_score) % 2 == 0) or 
                    (mode == 'tiebreak_points' and int((a_score+b_score) % 4) in [0,3])
                    ) else alternative_p
                mat[a_score, b_score] = current_p * mat[*p_wins] + (1-current_p) * mat[*p_loses]
    
    def generate_match_probability_matrix(self):
        p = self.probs_of_0_winning_set
        sets = self.num_of_sets
        probs = np.zeros((sets+1,sets+1))
        self.populate_prob_matrix(
            starting_p=p, 
            max_units_played=2*sets-2, 
            max_units=sets - 1, 
            mat=probs,
            mode='sets',
            deuce=False)
        return probs
    
    def generate_set_probability_matrix(self):
        ps = [self.calculate_init_game_win_prob(s) for s in range(2)]
        all_probs = np.zeros((2,8,8))
        for server in range(2):
            probs = all_probs[server]
            serving_p = ps[server]
            alternative_p = ps[1-server]
            probs[7,5] = probs[7,6] = 1
            probs[5,7] = probs[6,7] = 0
            probs[6,5] = alternative_p+(1-alternative_p)*self.probs_of_0_winning_tiebreak
            probs[5,6] = alternative_p*self.probs_of_0_winning_tiebreak
            probs[6,6] = self.probs_of_0_winning_tiebreak
            self.populate_prob_matrix(
                starting_p=serving_p,
                max_units_played=10, 
                max_units=5, 
                mat=probs, 
                mode='games', 
                alternative_p=alternative_p)
        return all_probs

# currently serving = 0 -> p
# currently serving = 1 -> 1-p

    def generate_game_probability_matrix_analytically(self):
        all_probs = np.zeros((2,5,5))
        for server in range(2):
            probs = all_probs[server]
            p = self.probs_of_winning_point[server]
            deuce_p = self.calculate_deuce_game_win_prob(server)
            probs[4,:3] = 1
            probs[:3,4] = 0
            probs[:4, :3] = [[p**(4-s)  for _ in range(3)] for s in range(4)]
            probs[:4, :2] += [[(4-s)*(p**(4-s)*(1-p)) for _ in range(2)] for s in range(4)]
            probs[:4, 0] += [(math.comb(6-s, 2) - (5-s))*(p**(4-s))*((1-p)**2) for s in range(4)]
            probs[:4,:4] += [[math.comb(6-s-o, 3-s) * (p**(3-s)) * ((1-p)**(3-o)) * deuce_p for o in range(4)] for s in range(4)]
            all_probs[server] = np.round(probs, decimals=10)
        assert all_probs[0,0,0] == round(self.calculate_init_game_win_prob(0), 10)
        return all_probs
    
    def reg_inc_beta_func(self, x, a, b):
        beta_func = integrate.quad(lambda t: (t**(a-1))*((1-t)**(b-1)), 0, 1)
        beta_inc = integrate.quad(lambda t: (t**(a-1))*((1-t)**(b-1)), 0, x)
        return beta_inc[0] / beta_func[0]
    
    def generate_game_probability_matrix_analytically_negbin(self):
        all_probs = np.zeros((2,5,5))
        for server in range(2):
            probs = all_probs[server]
            p = self.probs_of_winning_point[server]
            deuce_p = self.calculate_deuce_game_win_prob(server)
            probs[4,:3] = 1
            probs[:3,4] = 0
            probs[:4, :3] = np.fromfunction(np.vectorize(lambda i,j: self.reg_inc_beta_func(p, 4-i, 3-j)), (4,3))
            probs[:4,:4] += [[math.comb(6-s-o, 3-s) * (p**(3-s)) * ((1-p)**(3-o)) * deuce_p for o in range(4)] for s in range(4)]
            all_probs[server] = np.round(probs, decimals=10)
        assert all_probs[0,0,0] == round(self.calculate_init_game_win_prob(0), 10)
        return all_probs

    def generate_game_probability_matrix(self):
        all_probs = np.zeros((2,5,5))
        for server in range(2):
            probs = all_probs[server]
            p = self.probs_of_winning_point[server] 
            probs[3,3] = self.calculate_deuce_game_win_prob(server)
            self.populate_prob_matrix(
                starting_p=p, 
                max_units_played=5, 
                max_units=3, 
                mode='points',
                mat=probs)
            all_probs[server] = np.round(probs, decimals=10)
        assert all_probs[0,0,0] == round(self.calculate_init_game_win_prob(0), 10)
        return all_probs

    def generate_tiebreak_probability_matrix(self):
        tiebreak_length = self.tiebreak_length
        all_probs = np.zeros((2,tiebreak_length+1,tiebreak_length+1))
        for server in range(2):
            probs = all_probs[server]
            p = self.probs_of_winning_point[server] 
            alternative_p = self.probs_of_winning_point[1-server]
            probs[self.tiebreak_length-1,self.tiebreak_length-1] = self.calculate_deuce_tiebreak_win_prob(server)
            self.populate_prob_matrix(
                starting_p=p, 
                max_units_played=2*tiebreak_length - 3, 
                max_units=tiebreak_length-1, 
                mat=probs,
                mode='tiebreak_points', 
                alternative_p=alternative_p)
        return all_probs
    
    def calculate_current_game_win_probability(self):
        states = self.states
        if self.in_tiebreak:
            return self.inner_tiebreak_probabilities[self.currently_serving, states[0]['points'], states[1]['points']]
        else:
            return self.inner_game_probabilities[self.currently_serving, states[0]['points'], states[1]['points']]

    def calculate_deuce_game_win_prob(self, server):
        p = self.probs_of_winning_point[server]
        return p**2 / (p**2 + (1-p)**2)
    
    def calculate_deuce_tiebreak_win_prob(self, server):
        p_a = self.probs_of_winning_point[server]
        p_b = 1 - self.probs_of_winning_point[1-server]
        return (p_a - p_a*p_b)/(1 - p_a*p_b - (1-p_a)*(1-p_b))

    def calculate_init_game_win_prob(self, server):
        p = self.probs_of_winning_point[server]
        return (p**4/(p**2 + (1-p)**2)) * (-8*p**3 + 28*p**2-34*p + 15)

    def simulate_match_by_point(self, points):
        self.pre_points_match_probs = []
        for pt_winner in points:
            self.point_win(pt_winner)

    def point_win(self, player):
        states = self.states
        match_win_prob = self.get_match_win_prob_from_state(states, self.in_tiebreak, self.currently_serving, self.set_starting_server)
        self.pre_points_match_probs.append(match_win_prob)
        self.historical_states.append([dict(s) for s in states])
        states0, in_tiebreak0, currently_serving0, set_starting_server0 = self.get_state_after_point_win(0)
        states1, in_tiebreak1, currently_serving1, set_starting_server1 = self.get_state_after_point_win(1)
        match_win_prob0 = self.get_match_win_prob_from_state(states0, in_tiebreak0, currently_serving0, set_starting_server0)
        match_win_prob1 = self.get_match_win_prob_from_state(states1, in_tiebreak1, currently_serving1, set_starting_server1)
        self.pre_points_importances.append(match_win_prob0-match_win_prob1)
        states[player]['points'] += 1
        win_num = self.tiebreak_length if self.in_tiebreak else 4
        if all(state['points'] == win_num-1 for state in states):
            states[0]['points'] = win_num-2
            states[1]['points'] = win_num-2
        player_state = states[player]
        if player_state['points'] == win_num:
            self.game_win(player)

    def game_win(self, player):
        states = self.states
        self.currently_serving = 1-self.currently_serving
        games = 'games'
        states[player]['points'] = 0
        states[player]['games'] += 1
        states[1-player]['points'] = 0
        # print(f'({states[0][games]} - {states[1][games]})')
        if all(state['games'] == 6 for state in states):
            self.in_tiebreak = True
            return
        player_games = states[player][games]
        if (player_games == 6 and states[1-player][games] < player_games - 1)  or (player_games == 7):
            self.set_win(player)
        self.update_probs_of_winning_set()
        # print(f'set_win_prob_if_win: {self.prob_of_winning_set_given_cur_game_win}')
        # print(f'set_win_prob_if_lose: {self.prob_of_winning_set_given_cur_game_loss}')

    def set_win(self, player):
        sets = 'sets'
        states = self.states
        self.in_tiebreak = False
        states[player]['games'] = 0
        states[1-player]['games'] = 0
        states[player]['sets'] += 1
        self.set_starting_server = self.currently_serving
        # print(f'SETS: ({states[0][sets]} - {states[1][sets]})')
        if self.num_of_sets == states[player]['sets']:
            self.match_winner = player
            self.pre_points_match_probs.append(1-player)
            self.historical_states.append([dict(s) for s in states])
            return
        self.update_probs_of_winning_match()
        self.set_points.append(len(self.pre_points_match_probs))
        # print(f'MATCH_win_prob_if_win: {self.prob_of_winning_match_given_cur_set_win}')
        # print(f'MATCH_win_prob_if_lose: {self.prob_of_winning_match_given_cur_set_loss}')

    def get_state_after_point_win(self, player):
        cur_states = [dict(s) for s in self.states]
        cur_in_tiebreak, cur_currently_serving, cur_set_starting_server = self.in_tiebreak, self.currently_serving, self.set_starting_server
        cur_states[player]['points'] += 1
        win_num = self.tiebreak_length if cur_in_tiebreak else 4
        if all(state['points'] == win_num-1 for state in cur_states):
            cur_states[0]['points'] = win_num-2
            cur_states[1]['points'] = win_num-2
        player_state = cur_states[player]
        if player_state['points'] == win_num:
            cur_currently_serving = 1-cur_currently_serving
            cur_states[player]['points'] = 0
            cur_states[player]['games'] += 1
            cur_states[1-player]['points'] = 0
            if all(state['games'] == 6 for state in cur_states):
                cur_in_tiebreak = True
            player_games = cur_states[player]['games']
            if (player_games == 6 and cur_states[1-player]['games'] < player_games - 1)  or (player_games == 7):
                cur_in_tiebreak = False
                cur_set_starting_server = cur_currently_serving
                cur_states[player]['games'] = 0
                cur_states[1-player]['games'] = 0
                cur_states[player]['sets'] += 1
        return cur_states, cur_in_tiebreak, cur_currently_serving, cur_set_starting_server

    def get_match_win_prob_from_state(self, states, in_tiebreak, currently_serving, set_starting_server):
        game_probabilities = self.inner_tiebreak_probabilities if in_tiebreak else self.inner_game_probabilities
        game_win_prob = game_probabilities[currently_serving, states[0]['points'], states[1]['points']]
        prob_of_winning_set_given_cur_game_win, prob_of_winning_set_given_cur_game_loss = self.get_probs_of_winning_set(states, set_starting_server)
        prob_of_winning_match_given_cur_set_win, prob_of_winning_match_given_cur_set_loss = self.get_probs_of_winning_match(states)
        match_win_prob = (
            game_win_prob * prob_of_winning_set_given_cur_game_win * prob_of_winning_match_given_cur_set_win +
            game_win_prob * (1-prob_of_winning_set_given_cur_game_win) * prob_of_winning_match_given_cur_set_loss +
            (1-game_win_prob) * prob_of_winning_set_given_cur_game_loss * prob_of_winning_match_given_cur_set_win +
            (1-game_win_prob) * (1-prob_of_winning_set_given_cur_game_loss) * prob_of_winning_match_given_cur_set_loss
        )
        return match_win_prob
    
    def get_match_win_prob_external(self):
        game_probabilities = self.inner_tiebreak_probabilities if self.in_tiebreak else self.inner_game_probabilities
        game_win_prob = game_probabilities[self.currently_serving, self.states[0]['points'], self.states[1]['points']]
        prob_of_winning_set_given_cur_game_win, prob_of_winning_set_given_cur_game_loss = self.get_probs_of_winning_set(self.states, self.set_starting_server)
        prob_of_winning_match_given_cur_set_win, prob_of_winning_match_given_cur_set_loss = self.get_probs_of_winning_match(self.states)
        match_win_prob = (
            game_win_prob * prob_of_winning_set_given_cur_game_win * prob_of_winning_match_given_cur_set_win +
            game_win_prob * (1-prob_of_winning_set_given_cur_game_win) * prob_of_winning_match_given_cur_set_loss +
            (1-game_win_prob) * prob_of_winning_set_given_cur_game_loss * prob_of_winning_match_given_cur_set_win +
            (1-game_win_prob) * (1-prob_of_winning_set_given_cur_game_loss) * prob_of_winning_match_given_cur_set_loss
        )
        return match_win_prob
    
    def set_state_from_score(self, score, reverse_players=False):
        # score = '6-4,6-7,2-3,40-30'
        game_points = {'0': 0, '15': 1, '30':2, '40':3, 'A':4}
        set_points = [str(i) for i in range(8)]
        states = [
            {'sets': 0,
             'games': 0,
             'points': 0},
            {'sets': 0,
             'games': 0,
             'points': 0}
        ]
        self.states = states

        scores = [s.split('-') for s in score.split(',')]

        player0 = int(reverse_players)
        self.in_tiebreak = False
        in_game = False
        games_played = 0
        games_played_before_cur_set = 0
        for score in scores:
            if all([s in set_points for s in score]) and not self.in_tiebreak and not in_game:
                score = [int(s) for s in score]
                games_played += sum(score)
                games_played_before_cur_set += sum(score)
                for i in range(2):
                    if (score[i] == 6 and score[i] - 1 > score[1-i]) or score[i] == 7:
                            states[int(player0 != i)]['sets'] += 1
                            in_game = False
                            break
                else:
                    states[player0]['games'] = score[0]
                    states[1-player0]['games'] = score[1]
                    in_game = True
                    games_played_before_cur_set -= sum(score)
                if all(s==6 for s in score):
                    self.in_tiebreak = True
            elif self.in_tiebreak:
                states[player0]['points'] = int(score[0])
                states[1-player0]['points'] = int(score[1])
            else:
                score = [game_points[s] for s in score]
                states[player0]['points'] = score[0]
                states[1-player0]['points'] = score[1]
            win_num = self.tiebreak_length if self.in_tiebreak else 4
            while all(state['points'] >= win_num-1 for state in states):
                states[player0]['points'] -= 1
                states[1-player0]['points'] -= 1
        self.currently_serving = self.starting_serve if games_played % 2 == 0 else 1-self.starting_serve
        self.set_starting_server = self.starting_serve if games_played_before_cur_set % 2 == 0 else 1-self.starting_serve
        return states, self.in_tiebreak, self.currently_serving, self.set_starting_server
    
    def get_match_score(self):
        return (self.states, self.match_winner)