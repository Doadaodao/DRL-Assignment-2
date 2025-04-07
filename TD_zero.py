import copy
import numpy as np
import os
import pickle


def td_learning(env, approximator, num_episodes=50000, alpha=0.1, gamma=0.99,
                epsilon=0.0001, save_interval=10000, print_interval = 100, save_dir="checkpoints"):
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

        trajectory = []

        while not done:
            curr_state = copy.deepcopy(state)
            env.add_random_tile()

            legal_moves = [a for a in range(4) if env.is_move_legal(a)]
            if not legal_moves:
                break

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
            new_state, new_score, done, _ = env.step(action)
            incremental_reward = new_score - previous_score
            previous_score = new_score
            max_tile = max(max_tile, np.max(new_state))

            # Store the trajectory information
            next_state = copy.deepcopy(new_state)
            trajectory.append((curr_state, incremental_reward, next_state, done))

            # --- TD Update ---
            # v_current = approximator.value(curr_state)
            # v_next = 0 if done else approximator.value(next_state)
            # delta = incremental_reward + gamma * v_next - v_current
            # approximator.update(curr_state, delta, alpha)
            # Optionally, you could store trajectory information here.
            
            state = env.board
            
        
        # --- Update using the entire trajectory in reverse ---
        trajectory.reverse()
        for (s, r, s_next, terminal_flag) in trajectory:
            v_current = approximator.value(s)
            v_next = 0 if terminal_flag else approximator.value(s_next)
            delta = r + gamma * v_next - v_current
            approximator.update(s, delta, alpha)

        final_scores.append(env.score)
        success_flags.append(1 if max_tile >= 2048 else 0)

        if (episode + 1) % print_interval == 0:
            avg_score = np.mean(final_scores[-print_interval:])
            success_rate = np.sum(success_flags[-print_interval:]) / print_interval
            print(f"Episode {episode+1}/{num_episodes} | Avg Score: {avg_score:.2f} | Success Rate: {success_rate:.2f}")
        # --- Save Checkpoint ---
        if (episode + 1) % save_interval == 0:
            checkpoint_filename = os.path.join(save_dir, f"approximator_checkpoint_episode_{episode+1}.pkl")
            with open(checkpoint_filename, "wb") as f:
                pickle.dump(approximator, f)
            print(f"Checkpoint saved at episode {episode+1} to {checkpoint_filename}")

    return final_scores