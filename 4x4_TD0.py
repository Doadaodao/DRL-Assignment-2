import matplotlib.pyplot as plt

from afterstate_env import Game2048AfterStateEnv 
from approximator import NTupleApproximator
from TD_zero import td_learning

patterns = [
    # Pattern 1: top row and left part of second row
    [(0, 0), (0, 1), (1, 0), (1, 1)],
    # Pattern 2: second row and left part of third row
    [(0, 1), (0, 2), (1, 1), (1, 2)],
    # Pattern 3: third row and left part of fourth row
    [(0, 0), (0, 1), (0, 2), (0, 3)],
    # Pattern 4
    [(1, 0), (1, 1), (1, 2), (1, 3)]
]

approximator = NTupleApproximator(board_size=4, patterns=patterns)
env = Game2048AfterStateEnv()
final_scores = td_learning(env, approximator, num_episodes=400000, alpha=0.1, gamma=0.99, save_dir="4x4_TD0_afterstate_checkpoints")

plt.plot(final_scores)
plt.xlabel("Episodes")
plt.ylabel("Scores")
plt.title("Training Progress")
plt.show()