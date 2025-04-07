import matplotlib.pyplot as plt

from afterstate_env import Game2048AfterStateEnv 
from approximator import NTupleApproximator
from TD_zero import td_learning

patterns = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 0), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 2)]
]

approximator = NTupleApproximator(board_size=4, patterns=patterns, v_init = 80000.0)
env = Game2048AfterStateEnv()
final_scores = td_learning(env, approximator, num_episodes=400000, alpha=0.1, gamma=0.99, save_dir="8x6_TD0_OI_80k_checkpoints")

plt.plot(final_scores)
plt.xlabel("Episodes")
plt.ylabel("Scores")
plt.title("Training Progress")
plt.show()