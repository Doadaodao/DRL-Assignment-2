import copy
import random
import math
import numpy as np
import os
import pickle
from collections import defaultdict

# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

# Example 5x6-tuple network patterns for a 4x4 board.
patterns = [
    # Pattern 1: top row and left part of second row
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    # Pattern 2: second row and left part of third row
    [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)],
    # Pattern 3: third row and left part of fourth row
    [(2, 0), (2, 1), (2, 2), (2, 3), (3, 0), (3, 1)],
    # Pattern 4
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    # Pattern 5
    [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
]

# -------------------------------
# Transformation functions for symmetric sampling on a 4x4 board.

def identity(pattern):
    return pattern

def rot90(pattern):
    return [(c, 3 - r) for (r, c) in pattern]

def rot180(pattern):
    return [(3 - r, 3 - c) for (r, c) in pattern]

def rot270(pattern):
    return [(3 - c, r) for (r, c) in pattern]

def reflect_horizontal(pattern):
    return [(r, 3 - c) for (r, c) in pattern]

def reflect_vertical(pattern):
    return [(3 - r, c) for (r, c) in pattern]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        Initializes the N-Tuple approximator.
        'patterns' is a list of base tuple patterns (each a list of (row, col) tuples).
        """
        self.board_size = board_size
        self.patterns = patterns
        # Create one weight dictionary per base pattern (shared across its symmetric variants)
        self.weights = [defaultdict(float) for _ in patterns]
        # Group symmetric variants per base pattern.
        self.symmetry_groups = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_groups.append(syms)

    def generate_symmetries(self, pattern):
        """
        Generate the eight symmetric transformations of the input pattern.
        These include the identity, rotations (90, 180, 270 degrees) and
        the horizontal reflections of each.
        """
        sym = []
        for transform in [identity, rot90, rot180, rot270]:
            p = transform(pattern)
            sym.append(p)
            sym.append(reflect_horizontal(p))
        # Remove duplicates if any symmetry maps the pattern onto itself.
        unique = []
        for s in sym:
            if s not in unique:
                unique.append(s)
        return unique

    def tile_to_index(self, tile):
        """
        Converts tile values to an index for the lookup table.
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        """
        Extract tile values from the board based on the given coordinates
        and convert them into a feature tuple.
        """
        return tuple(self.tile_to_index(board[r, c]) for (r, c) in coords)

    def value(self, board):
        """
        Estimate the board value: sum the evaluations from all patterns.
        """
        total_value = 0.0
        for group_idx, group in enumerate(self.symmetry_groups):
            for sym in group:
                feature = self.get_feature(board, sym)
                total_value += self.weights[group_idx][feature]
        return total_value

    def update(self, board, delta, alpha):
        """
        Update weights based on the TD error using TD(0).
        For each tuple group, update each symmetric variant's shared weight.
        The update is scaled by 1/(number of symmetric variants).
        """
        for group_idx, group in enumerate(self.symmetry_groups):
            num_sym = len(group)
            for sym in group:
                feature = self.get_feature(board, sym)
                self.weights[group_idx][feature] += (alpha / num_sym) * delta

def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99,
                epsilon=0.1, save_interval=10000, save_dir="checkpoints"):
    """
    Trains the 2048 agent using TD(0) Learning with trajectory-based updates
    and saves approximator checkpoints locally at specified intervals.

    Args:
        env: The 2048 game environment.
        approximator: NTupleApproximator instance.
        num_episodes: Number of training episodes.
        alpha: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon-greedy exploration rate.
        save_interval: Number of episodes between saving checkpoints.
        save_dir: Directory to save checkpoints.
    """
    final_scores = []
    success_flags = []

    # Create the checkpoint directory if it doesn't exist.
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []  # Records (state, incremental_reward, next_state, done)
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # --- Action Selection (ε-greedy) ---
            if random.random() < epsilon:
                action = random.choice(legal_moves)
            else:
                best_value = -float('inf')
                best_action = None
                for a in legal_moves:
                    env_copy = copy.deepcopy(env)
                    sim_state, sim_score, sim_done, _ = env_copy.step(a)
                    reward = sim_score - previous_score
                    value_est = reward + gamma * approximator.value(sim_state)
                    if value_est > best_value:
                        best_value = value_est
                        best_action = a
                action = best_action

            # --- Take Action ---
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            trajectory.append((state, incremental_reward, next_state, done))
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))
            state = next_state

        # --- End of Episode: Batch TD(0) Updates ---
        for (s, r, s_next, terminal_flag) in trajectory:
            v_current = approximator.value(s)
            v_next = 0 if terminal_flag else approximator.value(s_next)
            delta = r + gamma * v_next - v_current
            approximator.update(s, delta, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 100 == 0:
            avg_score = np.mean(final_scores[-100:])
            success_rate = np.sum(success_flags[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")

        # --- Save Checkpoint ---
        if (episode + 1) % save_interval == 0:
            checkpoint_filename = os.path.join(save_dir, f"approximator_checkpoint_episode_{episode+1}.pkl")
            with open(checkpoint_filename, "wb") as f:
                pickle.dump(approximator, f)
            print(f"Checkpoint saved at episode {episode+1} to {checkpoint_filename}")

    return final_scores

approximator = NTupleApproximator(board_size=4, patterns=patterns)

env = Game2048Env()

# Run TD-Learning training
# Note: To achieve significantly better performance, you will likely need to train for over 100,000 episodes.
# However, to quickly verify that your implementation is working correctly, you can start by running it for 1,000 episodes before scaling up.
final_scores = td_learning(env, approximator, num_episodes=150000, alpha=0.1, gamma=0.99, lambda_param=0.9)

plt.plot(final_scores)
plt.xlabel("Episodes")
plt.ylabel("Scores")
plt.title("Training Progress")
plt.show()