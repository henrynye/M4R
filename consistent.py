from tqdm import tqdm
import random
def calculate_E(R_1, R_2):
    return 1/(1+10**((R_2 - R_1) / 400))

def update_R(R, S, E):
    return R + 32*(S - E)

def play_match(R_A, R_B, epsilon):
    E_A = calculate_E(R_A, R_B)
    E_B = calculate_E(R_B, R_A)
    result_A = int(random.random() < epsilon)
    result_B = 1 - result_A
    R_A = update_R(R_A, result_A, E_A)
    R_B = update_R(R_B, result_B, E_B)
    return R_A, R_B

R_A = 1000
R_B = 1500
epsilon = 0.6
E_A = calculate_E(R_A, R_B)


for i in tqdm(range(10_000_000)):
    R_A, R_B = play_match(R_A, R_B, epsilon)

R_A, R_B = play_match(R_A, R_B, epsilon)

print(calculate_E(R_A, R_B))

