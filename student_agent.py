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
from collections import defaultdict

def identity(pattern):
    return pattern

def rot90(pattern):
    # Rotate 90 degrees clockwise:
    # (row, col) -> (col, 3 - row)
    return [(c, 3 - r) for (r, c) in pattern]

def rot180(pattern):
    # Rotate 180 degrees:
    # (row, col) -> (3 - row, 3 - col)
    return [(3 - r, 3 - c) for (r, c) in pattern]

def rot270(pattern):
    # Rotate 270 degrees clockwise:
    # (row, col) -> (3 - col, r)
    return [(3 - c, r) for (r, c) in pattern]

def reflect_horizontal(pattern):
    # Reflect over vertical axis:
    # (row, col) -> (row, 3 - col)
    return [(r, 3 - c) for (r, c) in pattern]

def reflect_vertical(pattern):
    # Reflect over horizontal axis:
    # (row, col) -> (3 - row, c)
    return [(3 - r, c) for (r, c) in pattern]

class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """ Initializes the N-Tuple approximator. 'patterns' is a list of base tuple patterns (each a list of (row, col) tuples). """
        self.board_size = board_size
        self.patterns = patterns
        # Create one weight dictionary per base pattern (shared across its symmetric variants)
        self.weights = [defaultdict(float) for _ in patterns]
        
        # Instead of a flat list of all symmetric variants, we group them per base pattern.
        self.symmetry_groups = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            self.symmetry_groups.append(syms)

    def generate_symmetries(self, pattern):
        # Generate 8 symmetrical transformations of the given pattern.
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
        # Remove any duplicates (if any symmetry maps the pattern onto itself)
        unique = []
        for s in sym:
            if s not in unique:
                unique.append(s)
        # print(f"Unique symmetries for pattern {pattern}: {unique}")
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
        # Extract tile values from the board based on the given coordinates and convert them into a feature tuple.
        return tuple(self.tile_to_index(board[r, c]) for (r, c) in coords)

    def value(self, board):
        # Estimate the board value: sum the evaluations from all patterns.
        total_value = 0.0
        # Iterate over each base pattern's group.
        for group_idx, group in enumerate(self.symmetry_groups):
            num_sym = len(group)
            group_value = 0.0
            for sym in group:
                feature = self.get_feature(board, sym)
                group_value += self.weights[group_idx][feature]
            # Average the value across the symmetry group.
            total_value += (group_value / num_sym)
        return total_value

    def update(self, board, delta, alpha):
        # Update weights based on the TD error.
        for group_idx, group in enumerate(self.symmetry_groups):
            num_sym = len(group)
            for sym in group:
                feature = self.get_feature(board, sym)
                self.weights[group_idx][feature] += (alpha / num_sym) * delta

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

import sys
sys.modules['__main__'].NTupleApproximator = NTupleApproximator
with open('approximator_checkpoint_episode_50000.pkl', 'rb') as f:
    approximator = pickle.load(f)

def get_action(state, score):
    env = Game2048Env()
    env.board = state
    env.score = score
    
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    # print("Value estimation of state is:", approximator.value(state))
    # Use your N-Tuple approximator to play 2048
    best_value = -float('inf')
    best_action = None
    for action in legal_moves:
        env_copy = copy.deepcopy(env)

        if action == 0:
            moved = env_copy.move_up()
        elif action == 1:
            moved = env_copy.move_down()
        elif action == 2:
            moved = env_copy.move_left()
        elif action == 3:
            moved = env_copy.move_right()

        # sim_state, sim_score, sim_done, _ = env_copy.step(a)
        reward = env_copy.score - env.score
        value_est = reward + approximator.value(env_copy.board)
        # print("Value estimation of state is:", approximator.value(sim_state))
        if value_est > best_value:
            best_value = value_est
            best_action = action

    return best_action # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


