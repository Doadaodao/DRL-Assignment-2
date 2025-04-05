import copy
import math
import random
from connect6 import Connect6Game  # assuming connect6.py is in the same directory

# ---------- Helper Functions for Game Simulation ----------

def get_legal_moves(game):
    """Return a list of legal moves as (row, col) tuples for the given game state."""
    moves = []
    for r in range(game.size):
        for c in range(game.size):
            if game.board[r, c] == 0:
                moves.append((r, c))
    return moves

def do_move(game, move):
    """
    Apply a move (row, col) to the game state without printing.
    Uses the current player's turn in the game state.
    """
    r, c = move
    if game.board[r, c] != 0:
        raise ValueError("Illegal move attempted")
    # Set the stone: 1 for Black, 2 for White (game.turn holds the current player's code)
    game.board[r, c] = game.turn
    # Check for a win after the move.
    winner = game.check_win()
    if winner != 0:
        game.game_over = True
    # Toggle turn.
    game.turn = 3 - game.turn

def rollout(game):
    """
    Simulate a random playout from the current game state until terminal.
    Returns the winner (1 or 2), or 0 if no winner (should not occur in Connect6).
    """
    while not game.game_over and get_legal_moves(game):
        move = random.choice(get_legal_moves(game))
        do_move(game, move)
    return game.check_win()

# ---------- MCTS Node Class ----------

class MCTSNode:
    def __init__(self, game, parent=None, move=None, playerJustMoved=None):
        """
        game: a deep-copied Connect6Game instance representing the state at this node.
        parent: the parent MCTSNode.
        move: the move (row, col) that led from the parent to this node.
        playerJustMoved: the player (1 or 2) who made the move to reach this node.
                         For the root node, this is set to 3 - game.turn.
        """
        self.game = game
        self.parent = parent
        self.move = move
        self.playerJustMoved = playerJustMoved  # for root, set as the opponent of current turn.
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = get_legal_moves(game)

    def ucb_score(self, child, c_param=1.4):
        """Calculate the UCB1 score for a child node."""
        return (child.wins / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)

    def select_child(self):
        """Select a child node with the highest UCB score."""
        return max(self.children, key=lambda child: self.ucb_score(child))

    def update(self, result):
        """
        Update this node's statistics.
        'result' is the winner (1 or 2) from the simulation.
        If the result equals the player who just moved at this node, count it as a win.
        """
        self.visits += 1
        if self.playerJustMoved == result:
            self.wins += 1

# ---------- MCTS Search Function ----------

def mcts_search(root, itermax=100):
    """
    Run MCTS starting from the root node for a given number of iterations.
    Returns the move (row, col) of the child of root with the highest visit count.
    """
    for _ in range(itermax):
        node = root
        state = copy.deepcopy(root.game)

        # --- Selection ---
        while node.untried_moves == [] and node.children:
            node = node.select_child()
            do_move(state, node.move)

        # --- Expansion ---
        if node.untried_moves:
            move = random.choice(node.untried_moves)
            do_move(state, move)
            node.untried_moves.remove(move)
            # The move was made by the player who was about to move at 'node'
            # After do_move, state.turn has switched, so the player who made the move is 3 - state.turn.
            child = MCTSNode(game=copy.deepcopy(state),
                             parent=node,
                             move=move,
                             playerJustMoved=3 - state.turn)
            node.children.append(child)
            node = child

        # --- Simulation (Rollout) ---
        result = rollout(state)

        # --- Backpropagation ---
        while node is not None:
            node.update(result)
            node = node.parent

    # Return the move from the child with the highest visits.
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.move

# ---------- Self-Play Training Loop ----------

def self_play_training(num_games=10, mcts_iter=100):
    """
    Run self-play games using MCTS to choose moves.
    For each move, MCTS is run from the current state for a fixed number of iterations.
    """
    results = {1: 0, 2: 0, 0: 0}  # wins for Black (1), White (2), and draws (0)
    for game_num in range(1, num_games + 1):
        game_state = Connect6Game(size=19)
        game_state.reset_board()  # Clear board and reset game state

        # For the root node, since no move has been made, we set playerJustMoved as the opponent.
        current_player = game_state.turn
        root_player = 3 - current_player

        # Create a copy of the game state for MCTS root.
        root = MCTSNode(game=copy.deepcopy(game_state), parent=None, move=None, playerJustMoved=root_player)

        move_count = 0
        while not game_state.game_over and get_legal_moves(game_state):
            # Create an MCTS root for the current game state.
            root = MCTSNode(game=copy.deepcopy(game_state),
                            parent=None,
                            move=None,
                            playerJustMoved=3 - game_state.turn)
            # Run MCTS to choose the next move.
            best_move = mcts_search(root, itermax=mcts_iter)
            # Apply the chosen move.
            do_move(game_state, best_move)
            move_count += 1

            # Optionally, you could print the move or board occasionally:
            print(f"Move {move_count}: Player {3 - game_state.turn} played {best_move}")

            # Check if the game has been won.
            if game_state.check_win() != 0:
                game_state.game_over = True

        winner = game_state.check_win()
        results[winner] += 1
        print(f"Game {game_num} finished in {move_count} moves. Winner: {winner if winner != 0 else 'Draw'}")

    print("\nSelf-play training complete.")
    print("Results over {} games:".format(num_games))
    print("  Black wins: ", results[1])
    print("  White wins: ", results[2])
    print("  Draws:      ", results[0])

if __name__ == "__main__":
    # You can adjust num_games and mcts_iter for longer training.
    self_play_training(num_games=10, mcts_iter=100)
