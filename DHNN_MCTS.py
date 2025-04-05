import os
import math
import copy
import random
import numpy as np
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import the Connect6 game environment.
from connect6 import Connect6Game

# Set up device to use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



##############################
# Dual-Headed Neural Network #
##############################

class Connect6Net(nn.Module):
    def __init__(self, board_size, num_channels=64):
        super(Connect6Net, self).__init__()
        self.board_size = board_size
        self.input_channels = 3  # We'll use three channels: black, white, and turn indicator.
        
        # Common convolutional layers.
        self.conv1 = nn.Conv2d(self.input_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        
        # Policy head.
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # Value head.
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        # x: (batch, 3, board_size, board_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Policy head.
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy = F.log_softmax(p, dim=1)  # log probabilities
        
        # Value head.
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value

##########################
# Board and State Helper #
##########################

def board_to_tensor(game):
    """
    Convert a Connect6Game board into a tensor representation.
    We use 3 channels:
      - Channel 0: Black stones (1 if board==1, else 0)
      - Channel 1: White stones (1 if board==2, else 0)
      - Channel 2: Current player's turn indicator (all ones if game.turn==1, else all zeros)
    """
    board = game.board.astype(np.float32)
    black = (board == 1).astype(np.float32)
    white = (board == 2).astype(np.float32)
    turn_plane = np.ones_like(black) if game.turn == 1 else np.zeros_like(black)
    
    state = np.stack([black, white, turn_plane], axis=0)
    return torch.tensor(state).unsqueeze(0)  # shape: (1, 3, board_size, board_size)

###################################
# MCTS with Dual-Headed Guidance  #
###################################

# Define an MCTS node for the guided search.
class MCTSNode:
    def __init__(self, game, parent=None, move=None):
        self.game = game  # a deepcopy of the game state at this node
        self.parent = parent
        self.move = move  # move that led to this node (None for root)
        self.children = {}  # move -> MCTSNode
        
        # Statistics for PUCT.
        self.visit_count = 0
        self.total_value = 0.0
        self.prior = {}  # move -> prior probability (from network)
        
        # Cache legal moves.
        self.untried_moves = get_legal_moves(game)
    
    def is_leaf(self):
        return len(self.children) == 0

    def q_value(self):
        # Return average value.
        if self.visit_count == 0:
            return 0
        return self.total_value / self.visit_count

def get_legal_moves(game):
    """Return list of legal moves as (row, col) tuples."""
    moves = []
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] == 0:
                moves.append((r, c))
    return moves

def apply_move(game, move):
    """
    Apply a move (row, col) to the game state.
    This function assumes the move is legal.
    """
    r, c = move
    if game.board[r, c] != 0:
        raise ValueError("Illegal move attempted")
    game.board[r, c] = game.turn
    winner = game.check_win()
    if winner != 0:
        game.game_over = True
    game.turn = 3 - game.turn

def mcts_search(root, net, num_simulations=100, c_puct=1.4):
    """
    Perform MCTS search guided by the dual-headed network.
    Returns a probability distribution over moves (as a dict: move -> probability).
    """
    for _ in range(num_simulations):
        node = root
        state = copy.deepcopy(root.game)
        
        # --- Selection ---
        # Traverse until we reach a leaf.
        while node.untried_moves == [] and node.children:
            best_score = -float('inf')
            best_move = None
            for move, child in node.children.items():
                # PUCT: Q + U, where U = c_puct * P * sqrt(N_parent) / (1 + N_child)
                P = node.prior.get(move, 0)
                U = c_puct * P * math.sqrt(node.visit_count) / (1 + child.visit_count)
                score = child.q_value() + U
                if score > best_score:
                    best_score = score
                    best_move = move
            node = node.children[best_move]
            apply_move(state, best_move)
        
        # --- Expansion & Evaluation ---
        if not state.game_over:
            # Evaluate the leaf node with the neural network.
            state_tensor = board_to_tensor(state).to(device)
            with torch.no_grad():
                log_policy, value = net(state_tensor)
            policy = log_policy.exp().cpu().numpy().flatten()  # convert log probs to probabilities
            # Mask illegal moves.
            legal = get_legal_moves(state)
            mask = np.zeros(state.board.size, dtype=np.float32)
            for move in legal:
                idx = move[0] * state.size + move[1]
                mask[idx] = 1
            policy = policy * mask
            if policy.sum() > 0:
                policy = policy / policy.sum()
            else:
                # if no legal moves according to network, assign uniform probability.
                policy = mask / mask.sum()
            
            # Set priors for expansion.
            node.prior = {}
            for move in legal:
                idx = move[0] * state.size + move[1]
                node.prior[move] = policy[idx]
            
            # Expand one random move from the untried moves.
            move_to_expand = random.choice(node.untried_moves)
            apply_move(state, move_to_expand)
            child_node = MCTSNode(copy.deepcopy(state), parent=node, move=move_to_expand)
            node.children[move_to_expand] = child_node
            node.untried_moves.remove(move_to_expand)
            
            # The value from the network is our evaluation.
            leaf_value = value.item()
        else:
            # If state is terminal, determine the outcome.
            winner = state.check_win()
            # From the perspective of the parent, value is 1 if parent.player (the one who moved) wins.
            leaf_value = 1.0 if winner == (3 - state.turn) else -1.0
        
        # --- Backpropagation ---
        while node is not None:
            node.visit_count += 1
            node.total_value += leaf_value
            leaf_value = -leaf_value  # switch perspective
            node = node.parent
    
    # After simulations, return the visit counts as improved probabilities.
    visits = np.zeros(root.game.board.size, dtype=np.float32)
    for move, child in root.children.items():
        idx = move[0] * root.game.size + move[1]
        visits[idx] = child.visit_count
    if visits.sum() > 0:
        visits = visits / visits.sum()
    return visits  # shape: (board_size * board_size,)

#########################################
# Self-Play and Training Data Collection#
#########################################

def self_play_game(net, mcts_simulations=100):
    """
    Play one game using MCTS guided by the network.
    Returns a list of training examples: (state tensor, target policy (flattened), outcome)
    Outcome is +1 if current player eventually wins, -1 if loses.
    """
    game = Connect6Game(size=19)
    game.reset_board()
    examples = []
    states = []
    mcts_policies = []
    players = []
    
    while not game.game_over and get_legal_moves(game):
        state_tensor = board_to_tensor(game)
        root = MCTSNode(copy.deepcopy(game))
        # Run MCTS from current state.
        mcts_policy = mcts_search(root, net, num_simulations=mcts_simulations)
        
        # Store training example (state, policy, current player).
        examples.append((state_tensor, mcts_policy, game.turn))
        
        # Choose move proportional to visit count.
        flat_moves = mcts_policy.flatten()
        move_idx = np.random.choice(np.arange(len(flat_moves)), p=flat_moves)
        move = (move_idx // game.size, move_idx % game.size)
        
        apply_move(game, move)
    
    # Determine game outcome from the perspective of the player at each step.
    winner = game.check_win()
    outcome = 1 if winner == 1 else -1 if winner == 2 else 0
    # Adjust outcomes for each example based on which player was at turn.
    training_examples = []
    for state, policy, player in examples:
        # If the current player at that state is the winner, outcome=+1, else -1.
        adjusted_outcome = outcome if player == 1 else -outcome
        training_examples.append((state, torch.tensor(policy, dtype=torch.float32), adjusted_outcome))
    
    return training_examples

##########################
# Training the Network   #
##########################

def train_network(net, optimizer, training_data, epochs=1, batch_size=32):
    """
    Train the network on the training_data.
    training_data is a list of tuples: (state_tensor, target_policy, outcome)
    """
    net.train()
    loss_total = 0.0
    n = len(training_data)
    for epoch in range(epochs):
        random.shuffle(training_data)
        for i in range(0, n, batch_size):
            batch = training_data[i:i+batch_size]
            states = torch.cat([example[0] for example in batch], dim=0).to(device)
            target_policies = torch.stack([example[1] for example in batch]).to(device)
            target_values = torch.tensor([example[2] for example in batch], dtype=torch.float32).unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            log_policy_pred, value_pred = net(states)
            # Policy loss: negative log likelihood.
            policy_loss = -torch.mean(torch.sum(target_policies * log_policy_pred, dim=1))
            # Value loss: mean squared error.
            value_loss = F.mse_loss(value_pred, target_values)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
    return loss_total / n

##########################
# Main Training Loop     #
##########################

def main_training(num_iterations=10, games_per_iteration=5, mcts_simulations=100, epochs=1, checkpoint_interval=5):
    board_size = 19
    net = Connect6Net(board_size).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    # Directory to save checkpoints.
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    all_training_data = []
    
    for iteration in range(1, num_iterations+1):
        print(f"Iteration {iteration}/{num_iterations}")
        iteration_data = []
        # Self-play phase.
        for game in range(games_per_iteration):
            examples = self_play_game(net, mcts_simulations=mcts_simulations)
            iteration_data.extend(examples)
            print(f"  Completed self-play game {game+1}/{games_per_iteration}, collected {len(examples)} examples.")
        
        all_training_data.extend(iteration_data)
        # Training phase.
        loss = train_network(net, optimizer, iteration_data, epochs=epochs)
        print(f"  Training loss: {loss:.4f}")
        
        # Save checkpoint.
        if iteration % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"connect6_net_iter{iteration}.pth")
            torch.save(net.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved to {checkpoint_path}")
    
    # Save final model.
    final_path = os.path.join(checkpoint_dir, "connect6_net_final.pth")
    torch.save(net.state_dict(), final_path)
    print("Training complete. Final model saved.")

if __name__ == "__main__":
    main_training(num_iterations=10, games_per_iteration=5, mcts_simulations=100, epochs=1, checkpoint_interval=5)
