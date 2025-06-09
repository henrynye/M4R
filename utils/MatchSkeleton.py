from utils.utilities import convert_num_to_tennis_points

class MatchSkeleton:
    def __init__(self, sets = 2, tiebreak_length = 7, starting_serve = 0):
        self.in_tiebreak=False
        self.tiebreak_length = tiebreak_length
        self.currently_serving = starting_serve
        self.set_starting_server = starting_serve
        self.num_of_sets = sets
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
        self.score = []
        self.scores =[]

    def point_win(self, player):
        states = self.states
        if self.in_tiebreak:
            win_num = self.tiebreak_length
            whole_score = ''.join(self.score) + f"{states[0]['games']}-{states[1]['games']},{states[0]['points']}-{states[1]['points']}"
        else:
            win_num = 4
            whole_score = ''.join(self.score) + f"{states[0]['games']}-{states[1]['games']},{convert_num_to_tennis_points(states[0]['points'])}-{convert_num_to_tennis_points(states[1]['points'])}"
        self.scores.append(whole_score)
        states[player]['points'] += 1
        if states[1-player]['points'] == win_num:
            states[player]['points'] = win_num-1
            states[1-player]['points'] = win_num-1
        if states[player]['points'] >= win_num and states[player]['points'] > states[1-player]['points'] + 1:
            self.game_win(player)

    def game_win(self, player):
        states = self.states
        self.currently_serving = 1-self.currently_serving
        states[player]['points'] = 0
        states[player]['games'] += 1
        states[1-player]['points'] = 0
        if all(state['games'] == 6 for state in states):
            self.in_tiebreak = True
            return
        player_games = states[player]['games']
        if (player_games == 6 and states[1-player]['games'] < player_games - 1)  or (player_games == 7):
            self.set_win(player)

    def set_win(self, player):
        states = self.states
        self.in_tiebreak = False
        self.score.append(f"{states[0]['games']}-{states[1]['games']},")
        states[player]['games'] = 0
        states[1-player]['games'] = 0
        states[player]['sets'] += 1
        self.set_starting_server = self.currently_serving
        if self.num_of_sets == states[player]['sets']:
            self.match_winner = player
            return
