import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math

from randomstate_env import Game2048Env

import copy
import random
import math
import numpy as np
import os
import pickle
from collections import defaultdict

# -------------------------------
# Transformation functions for symmetric sampling on a 4x4 board.

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

def const_factory():
    return 160000.000001 / 8

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


def td_learning(env, approximator, num_episodes=50000, alpha=0.01, gamma=0.99,
                epsilon=0.0001, save_interval=10000, save_dir="checkpoints"):
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
        previous_score = 0
        done = False
        max_tile = np.max(state)

        while not done:
            curr_state = copy.deepcopy(state)
            env.add_random_tile()

            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

            # --- Action Selection (ε-greedy) ---
            if random.random() < epsilon:
                # Exploration: choose a random legal move.
                action = random.choice(legal_moves)
            else:
                # Exploitation: choose the move with the highest estimated value.
                best_value = -float('inf')
                best_action = None
                # Evaluate each legal move by simulating the step.
                for a in legal_moves:
                    env_copy = copy.deepcopy(env)
                    # Note: We use previous_score from the main env for consistency.
                    sim_state, sim_score, sim_done, _ = env_copy.step(a)
                    reward = sim_score - previous_score
                    # Estimate value using immediate reward and discounted next state value.
                    value_est = reward + gamma * approximator.value(sim_state)
                    if value_est > best_value:
                        best_value = value_est
                        best_action = a
                action = best_action

            # --- Take action in the real environment ---
            next_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(next_state))

            # --- TD Update ---
            v_current = approximator.value(curr_state)
            print(f"Current state value: {v_current}")
            v_next = 0 if done else approximator.value(next_state)
            delta = incremental_reward + gamma * v_next - v_current
            approximator.update(curr_state, delta, alpha)

            # Optionally, you could store trajectory information here.
            
            state = env.board

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % 500 == 0:
            avg_score = np.mean(final_scores[-500:])
            success_rate = np.sum(success_flags[-5000:]) / 500
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
        # --- Save Checkpoint ---
        if (episode + 1) % save_interval == 0:
            checkpoint_filename = os.path.join(save_dir, f"approximator_checkpoint_episode_{episode+1}.pkl")
            with open(checkpoint_filename, "wb") as f:
                pickle.dump(approximator, f)
            print(f"Checkpoint saved at episode {episode+1} to {checkpoint_filename}")

    return final_scores

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
        
        if value_est > best_value:
            best_value = value_est
            best_action = action

    return best_action


if __name__ == "__main__":
    

    with open('./8x6_afterstate_approximator/approximator_checkpoint_episode_50000.pkl', 'rb') as f:
        approximator = pickle.load(f)
    
    total_score = 0
    for _ in range(20):
        
        env = Game2048Env()
        state = env.reset()
        # env.render()
        done = False

        while not done:
            # Get the action from the approximator
            action = get_action(state, env.score)
            # Take the action in the environment
            next_state, new_score, done, _ = env.step(action)
            state = next_state
            # Print the current state and score
            # print("Current state:\n", state)
            # print("Current score:", env.score)
            # env.render()


        # Print final game results
        print("Game over, final score:", env.score)
        total_score += env.score
    print("Average score over 20 games:", total_score / 20)