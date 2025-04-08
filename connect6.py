import sys
import numpy as np
import random

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Policy-Value Network definition (should match your training code)
# -----------------------------
class PolicyValueNet(nn.Module):
    def __init__(self, board_size, channels=32):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
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
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Policy head
        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.log_softmax(p, dim=1)
        # Value head
        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

# -----------------------------
# MCTS Implementation (simplified)
# -----------------------------
class TreeNode:
    def __init__(self, prior):
        self.P = prior
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.children = {}  # map: move (tuple) -> TreeNode

class MCTS:
    def __init__(self, network, device, c_puct=1.0, num_simulations=50):
        self.network = network
        self.device = device
        self.c_puct = c_puct
        self.num_simulations = num_simulations

    def search(self, game, num_simulations=None):
        if num_simulations is None:
            num_simulations = self.num_simulations
        root = TreeNode(0)
        for _ in range(num_simulations):
            game_copy = clone_game(game)
            self._simulate(root, game_copy)
        counts = {}
        for move, child in root.children.items():
            counts[move] = child.N
        total = sum(counts.values())
        if total > 0:
            return {move: count/total for move, count in counts.items()}
        else:
            legal = get_legal_moves(game)
            uniform = 1.0 / len(legal)
            return {move: uniform for move in legal}

    def _simulate(self, node, game):
        winner = game.check_win()
        if game.game_over or winner != 0 or len(get_legal_moves(game)) == 0:
            if winner == 0:
                return 0
            return 1 if winner == (3 - game.turn) else -1

        if not node.children:
            state_tensor = board_to_tensor(game).to(self.device)
            log_policy, value = self.network(state_tensor)
            policy = torch.exp(log_policy).view(-1).detach()
            legal_moves = get_legal_moves(game)
            board_size = game.size
            priors = {}
            total_prior = 0.0
            for move in legal_moves:
                idx = move[0] * board_size + move[1]
                priors[move] = policy[idx].item()
                total_prior += priors[move]
            if total_prior > 0:
                for move in priors:
                    priors[move] /= total_prior
            else:
                uniform = 1.0 / len(legal_moves)
                for move in legal_moves:
                    priors[move] = uniform
            for move in legal_moves:
                node.children[move] = TreeNode(priors[move])
            return value.item()

        best_score = -float('inf')
        best_move = None
        for move, child in node.children.items():
            score = child.Q + self.c_puct * child.P * math.sqrt(node.N+1)/(1+child.N)
            if score > best_score:
                best_score = score
                best_move = move
        apply_move(game, best_move)
        value = -self._simulate(node.children[best_move], game)
        node.children[best_move].N += 1
        node.children[best_move].W += value
        node.children[best_move].Q = node.children[best_move].W / node.children[best_move].N
        node.N += 1
        return value

# -----------------------------
# Utility Functions used by MCTS & RLAgent
# -----------------------------
def clone_game(game):
    new_game = Connect6Game(game.size)
    new_game.board = np.copy(game.board)
    new_game.turn = game.turn
    new_game.game_over = game.game_over
    return new_game

def get_legal_moves(game):
    moves = []
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] == 0:
                moves.append((r, c))
    return moves

def board_to_tensor(game):
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
    state_tensor = torch.tensor(state).unsqueeze(0)  # shape: [1, 2, size, size]
    return state_tensor

def apply_move(game, move):
    r, c = move
    game.board[r, c] = game.turn
    if game.check_win() != 0:
        game.game_over = True
    game.turn = 3 - game.turn

# -----------------------------
# RLAgent class: Wraps network and MCTS for move selection.
# -----------------------------
class RLAgent:
    def __init__(self, board_size, checkpoint_path, mcts_simulations=50, c_puct=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = PolicyValueNet(board_size).to(self.device)
        if os.path.exists(checkpoint_path):
            self.network.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        else:
            print("Checkpoint not found at", checkpoint_path)
        self.network.eval()
        self.mcts = MCTS(self.network, self.device, c_puct=c_puct, num_simulations=mcts_simulations)

    def select_move(self, game):
        # Use MCTS search to get move probabilities
        move_probs = self.mcts.search(game)
        if move_probs:
            # For testing, select the move with the highest visit probability.
            best_move = max(move_probs, key=move_probs.get)
            return best_move
        else:
            return None

def load_rl_agent(board_size, checkpoint_path):
    return RLAgent(board_size, checkpoint_path)


class Connect6Game:
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False

    def reset_board(self):
        """Clears the board and resets the game."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Sets the board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won.
        Returns:
        0 - No winner yet
        1 - Black wins
        2 - White wins
        """
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != 0:
                    current_color = self.board[r, c]
                    for dr, dc in directions:
                        prev_r, prev_c = r - dr, c - dc
                        if 0 <= prev_r < self.size and 0 <= prev_c < self.size and self.board[prev_r, prev_c] == current_color:
                            continue
                        count = 0
                        rr, cc = r, c
                        while 0 <= rr < self.size and 0 <= cc < self.size and self.board[rr, cc] == current_color:
                            count += 1
                            rr += dr
                            cc += dc
                        if count >= 6:
                            return current_color
        return 0

    def index_to_label(self, col):
        """Converts column index to letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))  # Skips 'I'

    def label_to_index(self, col_char):
        """Converts letter to column index (accounting for missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Places stones and checks the game status."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            if len(stone) < 2:
                print("? Invalid format")
                return
            col_char = stone[0].upper()
            if not col_char.isalpha():
                print("? Invalid format")
                return
            col = self.label_to_index(col_char)
            try:
                row = int(stone[1:]) - 1
            except ValueError:
                print("? Invalid format")
                return
            if not (0 <= row < self.size and 0 <= col < self.size):
                print("? Move out of board range")
                return
            if self.board[row, col] != 0:
                print("? Position already occupied")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.turn = 3 - self.turn
        print('= ', end='', flush=True)

    # def generate_move(self, color):
    #     """Generates a random move for the computer."""
    #     if self.game_over:
    #         print("? Game over")
    #         return

    #     empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
    #     selected = random.sample(empty_positions, 1)
    #     move_str = ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in selected)
        
    #     self.play_move(color, move_str)

    #     print(f"{move_str}\n\n", end='', flush=True)
    #     print(move_str, file=sys.stderr)

    def generate_move(self, color):
        """Generates a move using the trained RL network via MCTS."""
        if self.game_over:
            print("? Game over")
            return

        # Lazy-load the RL agent if not already loaded
        if self.rl_agent is None:
            # Adjust checkpoint_path as needed
            checkpoint_path = "connect6_checkpoint.pth"
            self.rl_agent = load_rl_agent(self.size, checkpoint_path)
            print("RL agent loaded from checkpoint.")

        # Use the RL agent to select the best move using MCTS
        best_move = self.rl_agent.select_move(self)
        if best_move is None:
            print("? Failed to generate a move")
            return

        # Convert the chosen move (row, col) to move notation (e.g., "J10")
        move_str = f"{self.index_to_label(best_move[1])}{best_move[0] + 1}"

        # Play the move
        self.play_move(color, move_str)

        # Output the move (for GTP output and stderr logging)
        print(f"{move_str}\n\n", end='', flush=True)
        print(move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board as text."""
        print("= ")
        for row in range(self.size - 1, -1, -1):
            line = f"{row+1:2} " + " ".join("X" if self.board[row, col] == 1 else "O" if self.board[row, col] == 2 else "." for col in range(self.size))
            print(line)
        col_labels = "   " + " ".join(self.index_to_label(i) for i in range(self.size))
        print(col_labels)
        print(flush=True)

    def list_commands(self):
        """Lists all available commands."""
        print("= ", flush=True)  

    def process_command(self, command):
        """Parses and executes GTP commands."""
        command = command.strip()
        if command == "get_conf_str env_board_size:":
            return "env_board_size=19"

        if not command:
            return
        
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "boardsize":
            try:
                size = int(parts[1])
                self.set_board_size(size)
            except ValueError:
                print("? Invalid board size")
        elif cmd == "clear_board":
            self.reset_board()
        elif cmd == "play":
            if len(parts) < 3:
                print("? Invalid play command format")
            else:
                self.play_move(parts[1], parts[2])
                print('', flush=True)
        elif cmd == "genmove":
            if len(parts) < 2:
                print("? Invalid genmove command format")
            else:
                self.generate_move(parts[1])
        elif cmd == "showboard":
            self.show_board()
        elif cmd == "list_commands":
            self.list_commands()
        elif cmd == "quit":
            print("= ", flush=True)
            sys.exit(0)
        else:
            print("? Unsupported command")

    def run(self):
        """Main loop that reads GTP commands from standard input."""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                self.process_command(line)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"? Error: {str(e)}")

if __name__ == "__main__":
    game = Connect6Game()
    game.run()
