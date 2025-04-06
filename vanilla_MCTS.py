import os
import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import the Connect6Game environment (assumes connect6.py is in the same directory)
from connect6 import Connect6Game

###########################
# Helper functions
###########################

def clone_game(game):
    """Create a deep copy of the current game state."""
    new_game = Connect6Game(game.size)
    new_game.board = np.copy(game.board)
    new_game.turn = game.turn
    new_game.game_over = game.game_over
    return new_game

def get_legal_moves(game):
    """Return a list of legal moves as (row, col) tuples."""
    moves = []
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] == 0:
                moves.append((r, c))
    return moves

def board_to_tensor(game):
    """
    Convert the board state to a tensor with 2 channels.
    Channel 0: positions of the current player's stones.
    Channel 1: positions of the opponent's stones.
    Output shape: [1, 2, board_size, board_size]
    """
    board = game.board
    size = game.size
    current = game.turn
    if current == 1:
        current_stones = (board == 1).astype(np.float32)
        opp_stones = (board == 2).astype(np.float32)
    else:
        current_stones = (board == 2).astype(np.float32)
        opp_stones = (board == 1).astype(np.float32)
    state = np.stack([current_stones, opp_stones], axis=0)
    state_tensor = torch.tensor(state).unsqueeze(0)  # add batch dimension
    return state_tensor

def apply_move(game, move):
    """
    Apply a move (row, col) to the game state.
    This function updates the board and turn without printing anything.
    """
    r, c = move
    # Place stone of current player (1 or 2)
    game.board[r, c] = game.turn
    # Check win condition
    if game.check_win() != 0:
        game.game_over = True
    # Change turn: note that in Connect6, if not the very first move, a player places two stones
    # and turn only changes after both moves are placed.
    # For simulation, we assume each call applies one stone and the calling code handles sequential moves.
    game.turn = 3 - game.turn

def sample_move_from_pi(pi_dict):
    """
    Sample a move from the probability distribution.
    pi_dict: dictionary mapping (r, c) moves to probabilities.
    Returns one move (r, c).
    """
    moves = list(pi_dict.keys())
    probs = np.array([pi_dict[m] for m in moves], dtype=np.float64)
    # Normalize (should be already normalized, but be safe)
    probs /= probs.sum()
    move = random.choices(moves, weights=probs, k=1)[0]
    return move

def network_policy_to_dict(policy_tensor, game):
    """
    Given the raw policy output (a tensor of shape [1, board_size*board_size]),
    extract the probabilities for legal moves and return as a dictionary.
    """
    board_size = game.size
    policy_np = policy_tensor.detach().cpu().numpy().flatten()
    legal_moves = get_legal_moves(game)
    move_probs = {}
    # Collect the network's probability for each legal move
    for (r, c) in legal_moves:
        idx = r * board_size + c
        move_probs[(r, c)] = policy_np[idx]
    # Normalize over legal moves
    total = sum(move_probs.values())
    if total > 0:
        for move in move_probs:
            move_probs[move] /= total
    else:
        # Fallback to uniform probabilities if network output is degenerate
        uniform = 1.0 / len(legal_moves)
        for move in legal_moves:
            move_probs[move] = uniform
    return move_probs

###########################
# Neural Network Definition
###########################

class PolicyValueNet(nn.Module):
    def __init__(self, board_size, channels=32):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        # Input channels = 2 (current player's stones and opponent's stones)
        self.conv1 = nn.Conv2d(2, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # Shared convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)  # log probabilities over board_size^2 moves
        
        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))  # output in range [-1, 1]
        return p, v

###########################
# MCTS Implementation
###########################

class TreeNode:
    def __init__(self, prior):
        self.P = prior           # prior probability (from network)
        self.N = 0               # visit count
        self.W = 0.0             # total value
        self.Q = 0.0             # mean value
        self.children = {}       # map from move (tuple) to TreeNode

class MCTS:
    def __init__(self, network, device, c_puct=1.0, num_simulations=50):
        self.network = network
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations
        
    def search(self, game, num_simulations=None):
        """Run MCTS simulations starting from the current game state.
           Returns a dictionary mapping legal moves to normalized visit counts.
        """
        if num_simulations is None:
            num_simulations = self.num_simulations
        root = TreeNode(0)
        for _ in range(num_simulations):
            game_copy = clone_game(game)
            self._simulate(root, game_copy)
        # Collect visit counts for legal moves from root.children
        counts = {}
        for move, child in root.children.items():
            counts[move] = child.N
        total = sum(counts.values())
        if total > 0:
            pi = {move: count / total for move, count in counts.items()}
        else:
            # If no simulations were run (should not happen), use uniform distribution
            legal = get_legal_moves(game)
            uniform = 1.0 / len(legal)
            pi = {move: uniform for move in legal}
        return pi

    def _simulate(self, node, game):
        """
        Recursively simulate a game from the current node.
        Returns the value of the current state from the perspective of the current player.
        """
        # Check for terminal state
        winner = game.check_win()
        if game.game_over or winner != 0 or len(get_legal_moves(game)) == 0:
            # Terminal state: if current player wins, return 1; if loses, -1; draw 0.
            if winner == 0:
                return 0
            # The value is from the perspective of the player who just played, so flip sign.
            return 1 if winner == (3 - game.turn) else -1

        # If node is a leaf (not expanded)
        if not node.children:
            state_tensor = board_to_tensor(game).to(self.device)
            # Get network outputs; policy: log probabilities
            log_policy, value = self.network(state_tensor)
            # Exponentiate to get probabilities
            policy = torch.exp(log_policy).view(-1)
            policy = policy.detach()
            # Get legal moves and assign priors
            legal_moves = get_legal_moves(game)
            board_size = game.size
            priors = {}
            # Collect network probabilities for legal moves
            total_prior = 0.0
            for move in legal_moves:
                idx = move[0] * board_size + move[1]
                priors[move] = policy[idx].item()
                total_prior += priors[move]
            if total_prior > 0:
                for move in priors:
                    priors[move] /= total_prior
            else:
                # fallback: uniform probability
                uniform = 1.0 / len(legal_moves)
                for move in legal_moves:
                    priors[move] = uniform

            # Expand the node
            for move in legal_moves:
                node.children[move] = TreeNode(prior=priors[move])
            # Return the network's value estimate (from current player's perspective)
            return value.item()

        # Selection: choose the move with highest UCB score
        best_score = -float('inf')
        best_move = None
        for move, child in node.children.items():
            ucb = child.Q + self.c_puct * child.P * math.sqrt(node.N + 1) / (1 + child.N)
            if ucb > best_score:
                best_score = ucb
                best_move = move
        # Simulate the move on a cloned game
        apply_move(game, best_move)
        # Recursively simulate from the child node (value is from opponent's perspective, hence the minus sign)
        value = -self._simulate(node.children[best_move], game)
        # Backpropagate
        node.children[best_move].N += 1
        node.children[best_move].W += value
        node.children[best_move].Q = node.children[best_move].W / node.children[best_move].N
        node.N += 1
        return value

###########################
# Self-play and Training Loop
###########################

def self_play_game(network, mcts, device):
    """
    Play one self-play game using MCTS-guided moves.
    Returns a list of training examples: each is a tuple (state, pi, z)
    where state is a numpy array, pi is a move probability dictionary,
    and z is the final outcome from the perspective of the player at that state.
    """
    game = Connect6Game()
    training_examples = []
    move_count = 0
    # For Connect6, Black's first move is a single stone; later moves require two stones.
    while not game.game_over and len(get_legal_moves(game)) > 0:
        current_player = game.turn
        # Get state representation (as tensor, but record numpy array for training)
        state_tensor = board_to_tensor(game)
        state_np = state_tensor.squeeze(0).numpy()  # shape [2, board_size, board_size]
        # Determine number of stones to place this turn:
        stones_to_place = 1 if (move_count == 0 and current_player == 1) else 2
        for _ in range(stones_to_place):
            # Run MCTS to get move probabilities
            pi = mcts.search(game)
            # Record the training example (state, pi, current_player); outcome z is filled later.
            training_examples.append((state_np, pi, current_player))
            # Select a move (here we sample from the distribution; alternatively, use argmax)
            move = sample_move_from_pi(pi)
            apply_move(game, move)
            move_count += 1
            if game.game_over:
                break
    # Determine game outcome for training labels: winner is from Connect6Game.check_win()
    winner = game.check_win()
    examples = []
    for state, pi, player in training_examples:
        if winner == 0:
            z = 0
        else:
            z = 1 if winner == player else -1
        examples.append((state, pi, z))
    return examples

def train_network(network, optimizer, training_data, device, batch_size=32, epochs=5, l2_coef=1e-4):
    """
    Train the network on the self-play training data.
    training_data: list of (state, pi, z) tuples.
    The policy target pi is converted to a vector over board positions.
    """
    network.train()
    board_size = network.board_size
    num_samples = len(training_data)
    for epoch in range(epochs):
        random.shuffle(training_data)
        losses = []
        for i in range(0, num_samples, batch_size):
            batch = training_data[i:i+batch_size]
            states = []
            target_pis = []
            target_vs = []
            for state_np, pi_dict, z in batch:
                states.append(state_np)
                # Convert pi_dict (mapping move->prob) to a flat vector of length board_size^2
                target_pi = np.zeros(board_size * board_size, dtype=np.float32)
                for (r, c), prob in pi_dict.items():
                    idx = r * board_size + c
                    target_pi[idx] = prob
                target_pis.append(target_pi)
                target_vs.append(z)
            # Convert to tensors
            state_tensor = torch.tensor(np.array(states)).to(device)  # shape: [batch, 2, board_size, board_size]
            target_pi_tensor = torch.tensor(np.array(target_pis)).to(device)  # [batch, board_size^2]
            target_v_tensor = torch.tensor(np.array(target_vs), dtype=torch.float32).to(device)  # [batch]
            
            optimizer.zero_grad()
            log_pi, v = network(state_tensor)  # log_pi: [batch, board_size^2], v: [batch, 1]
            v = v.squeeze(1)
            # Policy loss: negative log likelihood (cross entropy)
            loss_pi = -torch.mean(torch.sum(target_pi_tensor * log_pi, dim=1))
            # Value loss: mean squared error
            loss_v = F.mse_loss(v, target_v_tensor)
            # L2 regularization
            l2_loss = sum(torch.sum(param ** 2) for param in network.parameters())
            loss = loss_pi + loss_v + l2_coef * l2_loss
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses) / len(losses)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

def main():
    # Hyperparameters
    board_size = 19
    num_selfplay_games = 10      # Number of self-play games per training iteration
    mcts_simulations = 50        # Number of MCTS simulations per move
    training_epochs = 5          # Training epochs per iteration
    batch_size = 32
    learning_rate = 1e-3
    l2_coef = 1e-4
    total_iterations = 100     # Total training iterations (self-play + training)
    checkpoint_path = "connect6_checkpoint.pth"

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize network and optimizer
    network = PolicyValueNet(board_size=board_size).to(device)
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    # Initialize MCTS module
    mcts = MCTS(network, device, c_puct=1.0, num_simulations=mcts_simulations)

    # Replay buffer for training examples
    replay_buffer = []

    for iteration in range(1, total_iterations + 1):
        print(f"\n--- Training Iteration {iteration} ---")
        # Self-play: generate training examples from several games
        iteration_examples = []
        for game_idx in range(num_selfplay_games):
            examples = self_play_game(network, mcts, device)
            iteration_examples.extend(examples)
            print(f"Game {game_idx+1} finished, {len(examples)} examples collected.")
        # Append to replay buffer (for simplicity, we use only recent examples)
        replay_buffer.extend(iteration_examples)
        # Optionally: limit the size of the replay buffer by sampling or truncating

        # Train the network with the collected data
        train_network(network, optimizer, replay_buffer, device,
                      batch_size=batch_size, epochs=training_epochs, l2_coef=l2_coef)

        # Save checkpoint every few iterations
        if iteration % 10 == 0:
            torch.save(network.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration} to {checkpoint_path}")

if __name__ == "__main__":
    main()
