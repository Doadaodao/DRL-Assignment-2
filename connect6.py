import sys
import numpy as np
import random
import math
import copy

# MCTS Node definition
class MCTSNode:
    def __init__(self, board, turn, moves_left, parent=None, move=None):
        self.board = board  # numpy array copy
        self.turn = turn    # whose turn it is at this node
        self.parent = parent
        self.move = move    # the move (a list of positions) that was applied from the parent's state
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = self.get_moves()

        self.moves_left = moves_left
    
    # Helper: Get candidate positions from the board (restrict search to near existing stones)
    def candidate_moves(self, board, margin):
        size = board.shape[0]
        moves_set = set()
        stones = np.argwhere(board != 0)
        # print(stones.size, file=sys.stderr)
        if stones.size == 0:
            return [(size // 2, size // 2)]
        for (r, c) in stones:
            for dr in range(-margin, margin + 1):
                for dc in range(-margin, margin + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                        moves_set.add((nr, nc))
        if not moves_set:
            return [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]
        return list(moves_set)

    def get_moves(self):
        # Determine the number of stones this move should place given the board state:
        s = 1 if np.count_nonzero(self.board) == 0 else 2
        poss = self.candidate_moves(self.board, margin=1)
        moves = []
        for pos in poss:
            moves.append(pos)
        return moves

class Connect6Game:
    
    def __init__(self, size=19):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)  # 0: Empty, 1: Black, 2: White
        self.turn = 1  # 1: Black, 2: White
        self.game_over = False
        self.last_opponent_move = None

        self.root_node = None

    def reset_board(self):
        """Resets the board and game state."""
        self.board.fill(0)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def set_board_size(self, size):
        """Changes board size and resets the game."""
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.turn = 1
        self.game_over = False
        print("= ", flush=True)

    def check_win(self):
        """Checks if a player has won. Returns 1 (Black wins), 2 (White wins), or 0 (no winner)."""
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
        """Converts a column index to a letter (skipping 'I')."""
        return chr(ord('A') + col + (1 if col >= 8 else 0))

    def label_to_index(self, col_char):
        """Converts a column letter to an index (handling the missing 'I')."""
        col_char = col_char.upper()
        if col_char >= 'J':  # 'I' is skipped
            return ord(col_char) - ord('A') - 1
        else:
            return ord(col_char) - ord('A')

    def play_move(self, color, move):
        """Processes a move and updates the board."""
        if self.game_over:
            print("? Game over")
            return

        stones = move.split(',')
        positions = []

        for stone in stones:
            stone = stone.strip()
            col_char = stone[0].upper()
            col = self.label_to_index(col_char)
            row = int(stone[1:]) - 1
            if not (0 <= row < self.size and 0 <= col < self.size) or self.board[row, col] != 0:
                print("? Invalid move")
                return
            positions.append((row, col))

        for row, col in positions:
            self.board[row, col] = 1 if color.upper() == 'B' else 2

        self.last_opponent_move = positions[-1]  # Track the opponent's last move
        self.turn = 3 - self.turn
        print('= ', end='', flush=True)
    
    # ----------------------------------
    # Heuristic evaluation function
    # ----------------------------------

    def evaluate_board(self, board, color):
        """
        Evaluates the board from the perspective of the given 'color' using pattern matching.
        Converts each row, column, and diagonal to a string representation:
        - '1' for our stone (color),
        - '2' for the opponent's stone,
        - '0' for an empty cell.
        Then, scans for specific patterns—including those with gaps such as "11011"—and returns
        the differential score (our potential minus opponent's potential).
        """
        opponent_color = 3 - color
        size = board.shape[0]
        # Create a character representation of the board.
        # Our stone becomes '1'; opponent's becomes '2'; empty remains '0'.
        board_rep = np.full((size, size), '0', dtype='<U1')
        for r in range(size):
            for c in range(size):
                if board[r, c] == color:
                    board_rep[r, c] = '1'
                elif board[r, c] == opponent_color:
                    board_rep[r, c] = '2'

        # Helper to convert an array (row of cells) to string.
        def array_to_str(arr):
            return ''.join(arr)

        # Gather all the lines (rows, columns, diagonals, and anti-diagonals) for pattern matching.
        lines = []
        # Rows.
        for r in range(size):
            lines.append(array_to_str(board_rep[r, :]))
        # Columns.
        for c in range(size):
            lines.append(array_to_str(board_rep[:, c]))
        # Diagonals (top-left to bottom-right).
        for offset in range(-size + 1, size):
            diag = board_rep.diagonal(offset=offset)
            if len(diag) >= 5:
                lines.append(array_to_str(diag))
        # Anti-diagonals (top-right to bottom-left).
        flipped = np.fliplr(board_rep)
        for offset in range(-size + 1, size):
            diag = flipped.diagonal(offset=offset)
            if len(diag) >= 5:
                lines.append(array_to_str(diag))

        # Define a dictionary of pattern strings and their corresponding scores.
        # These patterns capture various threats and opportunities.
        # For instance, the pattern "11011" represents a nearly-connected sequence
        # (two stones, a gap, followed by two stones) that you may exploit.
        patterns = {
            "111111":   10000,       # Six in a row (winning pattern).
            "0111110":  20000,     # Five in a row with open ends (winning threat).
            # "111011":   10000,     # Five in a row with open ends (winning threat).
            # "110111":   10000,     # Five in a row with open ends (winning threat).
            # "101111":   10000,     # Five in a row with open ends (winning threat).
            # "111101":   10000,     # Five in a row with open ends (winning threat).
            "011110":   20000,     # Open four: four contiguous stones with empty ends.
            "211110":   10000,     # Open four: four contiguous stones with empty ends.
            "011112":   10000,     # Open four: four contiguous stones with empty ends.
            # "0110110":  5000,     # A broken pattern: e.g. pattern "11011" with open ends.
            # "0101110":  5000,     # Nearly complete pattern.
            # "0110110":  5000,     # Nearly complete pattern.
            # "0111010":  5000,     # Variation on nearly complete pattern.
            "01110":    1000,     # Open three.
            # "010110":   1000,     # A split pattern with a gap in between.
            # "011010":   1000,     # A split pattern with a gap in between.
            # "0110":     100,     # Open two.
            # "01010":    100,     # Open two.
        }

        # Scan through all lines and accumulate scores.
        my_score = 0
        opp_score = 0
        for line in lines:
            for pat, sc in patterns.items():
                # Count occurrences for our stones.
                count = line.count(pat)
                my_score += count * sc
                # For the opponent, convert pattern '1's to '2's.
                opp_pat = pat.replace('1', '3')
                opp_pat = opp_pat.replace('2', '1')
                opp_pat = opp_pat.replace('3', '2')
                count_opp = line.count(opp_pat)
                opp_score += count_opp * sc
        # print("My score:", my_score, "Opponent score:", opp_score, file=sys.stderr)
        return my_score - opp_score

    # ----------------------------------
    # MCTS-based move generator
    # ----------------------------------
    def generate_move(self, color):
        """Generates the best move using a Monte Carlo tree search with board pattern evaluation."""
        if self.game_over:
            print("? Game over", flush=True)
            return

        my_color = 1 if color.upper() == 'B' else 2
        
        # Helper: Convert board coordinate move (list of (r, c)) to string move
        def move_to_str(move):
            return ",".join(f"{self.index_to_label(c)}{r+1}" for r, c in move)

        # Helper: Get candidate positions from the board (restrict search to near existing stones)
        def candidate_moves(board, margin):
            size = board.shape[0]
            moves_set = set()
            stones = np.argwhere(board != 0)
            # print(stones.size, file=sys.stderr)
            if stones.size == 0:
                return [(size // 2, size // 2)]
            for (r, c) in stones:
                for dr in range(-margin, margin + 1):
                    for dc in range(-margin, margin + 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < size and 0 <= nc < size and board[nr, nc] == 0:
                            moves_set.add((nr, nc))
            if not moves_set:
                return [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]
            return list(moves_set)

        # Helper: Check for win in a given board state
        def check_win_state(board):
            size = board.shape[0]
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for r in range(size):
                for c in range(size):
                    if board[r, c] != 0:
                        current_color = board[r, c]
                        for dr, dc in directions:
                            prev_r, prev_c = r - dr, c - dc
                            if 0 <= prev_r < size and 0 <= prev_c < size and board[prev_r, prev_c] == current_color:
                                continue
                            count = 0
                            rr, cc = r, c
                            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == current_color:
                                count += 1
                                rr += dr
                                cc += dc
                            if count >= 6:
                                return current_color
            return 0
        
        def evaluate_position(board, size, r, c, color):
            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            score = 0
            # print("Evaluating position", r, c, file=sys.stderr)

            for dr, dc in directions:
                count = 1
                rr, cc = r + dr, c + dc
                while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
                    count += 1
                    rr += dr
                    cc += dc
                rr, cc = r - dr, c - dc
                while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == color:
                    count += 1
                    rr -= dr
                    cc -= dc

                if count >= 5:
                    score += 10000
                elif count == 4:
                    score += 5000
                elif count == 3:
                    score += 1000
                elif count == 2:
                    score += 100
        
            return score
        
        def rule_based_generate_move(board, my_color):
            current_board = copy.deepcopy(board)

            size = board.shape[0]

            opponent_color = 3 - my_color
            # empty_positions = [(r, c) for r in range(size) for c in range(size) if board[r, c] == 0]
            empty_positions = candidate_moves(current_board, 1)
            # print("Current board", current_board, file=sys.stderr)
            # print("Empty positions", empty_positions, file=sys.stderr)

            # 1. Winning move
            # print("Checking winning move", file=sys.stderr)
            for r, c in empty_positions:
                current_board[r, c] = my_color
                if check_win_state(current_board) == my_color:
                    return (r, c)
                current_board[r, c] = 0

            # 2. Block opponent's winning move
            # print("Checking blocking move", file=sys.stderr)
            for r, c in empty_positions:
                current_board[r, c] = opponent_color
                if check_win_state(current_board) == opponent_color:
                    return (r, c)
                current_board[r, c] = 0

            # 3. Attack: prioritize strong formations
            # print("Checking attack move", file=sys.stderr)
            best_move = None
            best_score = -1
            for r, c in empty_positions:
                score = evaluate_position(current_board, size, r, c, my_color)
                if score > best_score:
                    best_score = score
                    best_move = (r, c)

            # 4. Defense: prevent opponent from forming strong positions
            # print("Checking defense move", file=sys.stderr)
            for r, c in empty_positions:
                opponent_score = evaluate_position(current_board, size, r, c, opponent_color)
                if opponent_score >= best_score:
                    best_score = opponent_score
                    best_move = (r, c)
        
            # 5. Execute best move
            if best_move:
                return best_move
            else:
                return random.choice(empty_positions)
        
        # Helper: Rollout (simulation) from a given state until terminal or a rollout limit is reached.
        def rollout(board, rollout_turn, moves_left, rollout_limit, factor = 100000):
            size = board.shape[0]
            current_board = copy.deepcopy(board)
            current_turn = rollout_turn

            if np.count_nonzero(current_board) == 0:
                current_board[size // 2, size // 2] = current_turn
            elif moves_left == 1:
                r, c = rule_based_generate_move(current_board, current_turn)
                current_board[r, c] = current_turn
                winner = check_win_state(current_board)
                if winner != 0:
                    # value = self.evaluate_board(current_board, my_color) / factor
                    # return value
                    return 1 if winner == my_color else -1
            
            current_turn = 3 - current_turn

            # Perform rollouts until a terminal state is reached or the limit is hit.
            for i in range(rollout_limit):
                for _ in range(2):
                    r, c = rule_based_generate_move(current_board, current_turn)
                    current_board[r, c] = current_turn
                    winner = check_win_state(current_board)
                    if winner != 0:
                        # value = self.evaluate_board(current_board, my_color) / factor
                        # return value
                        return 1 if winner == my_color else -1
                current_turn = 3 - current_turn

            # If no terminal state found, evaluate the board
            value = self.evaluate_board(current_board, my_color) / factor
            return value
            return 0

        # MCTS Node definition
        class MCTSNode:
            def __init__(self, board, turn, moves_left, parent=None, move=None):
                self.board = board  # numpy array copy
                self.turn = turn    # whose turn it is at this node
                self.parent = parent
                self.move = move    # the move (a list of positions) that was applied from the parent's state
                self.children = []
                self.wins = 0
                self.visits = 0
                self.untried_moves = self.get_moves()

                self.moves_left = moves_left

            def get_moves(self):
                # Determine the number of stones this move should place given the board state:
                s = 1 if np.count_nonzero(self.board) == 0 else 2
                poss = candidate_moves(self.board, margin=1)
                moves = []
                for pos in poss:
                    moves.append(pos)
                # if s == 1:
                #     for pos in poss:
                #         moves.append([pos])
                # else:
                #     # To restrict the branching factor, if there are many candidates, sample a subset.
                #     # if len(poss) > 10:
                #     #     poss = random.sample(poss, 10)
                #     n = len(poss)
                #     for i in range(n):
                #         for j in range(i + 1, n):
                #             moves.append([poss[i], poss[j]])
                return moves

        # UCT selection from a node’s children.
        def uct_select_child(node):
            return max(node.children,
                       key=lambda child: child.wins / child.visits +
                    1.41 * math.sqrt(2 * math.log(node.visits) / child.visits))

        def backpropagate(node, result):
            # Propagate result up the tree; flip sign as we move up.
            while node is not None:
                node.visits += 1
                node.wins += result
                if node.parent and node.parent.turn != node.turn:
                    result = -result
                node = node.parent

        def pick_untried_move(node):
            current_board = copy.deepcopy(node.board)

            size = current_board.shape[0]

            opponent_color = 3 - my_color

            positions = node.untried_moves
            
            # 1. Winning move
            # print("Checking winning move", file=sys.stderr)
            for r, c in positions:
                current_board[r, c] = my_color
                if check_win_state(current_board) == my_color:
                    return (r, c)
                current_board[r, c] = 0

            # 2. Block opponent's winning move
            # print("Checking blocking move", file=sys.stderr)
            for r, c in positions:
                current_board[r, c] = opponent_color
                if check_win_state(current_board) == opponent_color:
                    return (r, c)
                current_board[r, c] = 0

            # 3. Attack: prioritize strong formations
            # print("Checking attack move", file=sys.stderr)
            best_move = None
            best_score = -1
            for r, c in positions:
                score = evaluate_position(current_board, size, r, c, my_color)
                if score > best_score:
                    best_score = score
                    best_move = (r, c)

            # 4. Defense: prevent opponent from forming strong positions
            # print("Checking defense move", file=sys.stderr)
            for r, c in positions:
                opponent_score = evaluate_position(current_board, size, r, c, opponent_color)
                if opponent_score >= best_score:
                    best_score = opponent_score
                    best_move = (r, c)
        
            # 5. Execute best move
            if best_move:
                return best_move
            else:
                return random.choice(positions)
            
        # Main MCTS search procedure.
        def mcts_search(root_node, root_turn, moves_left, iterations):
            
            for _ in range(iterations):   
                # Selection
                node = root_node
                
                if root_node.untried_moves == []:
                    # Descend to a leaf node (one that has untried moves or is terminal)
                    while len(node.children) >= 5:
                        node = uct_select_child(node)
                
                # Expansion
                if node.untried_moves:
                    m = pick_untried_move(node)
                    new_board = np.copy(node.board)
                    # Apply the move for the current node’s turn.
                    
                    new_board[m[0], m[1]] = node.turn

                    if node.moves_left == 1:
                        child_node = MCTSNode(new_board, node.turn, 0, parent=node, move=m)
                    else:
                        child_node = MCTSNode(new_board, 3 - node.turn, 1, parent=node, move=m)

                    # print("New node size", np.count_nonzero(new_board),"turn", node.turn, file=sys.stderr)
                    node.children.append(child_node)
                    node.untried_moves.remove(m)
                    node = child_node
                # Simulation (rollout)
                result = rollout(node.board, node.turn, node.moves_left, rollout_limit=10)
                move_str = move_to_str([m])
                print("Simulated move:", move_str, "Rollout result:", result, file=sys.stderr)
                # Backpropagation
                backpropagate(node, result)
            # Choose the move from the root with the highest visit count.
            best_child = max(root_node.children, key=lambda c: c.visits) if root_node.children else None
            return [best_child.move] if best_child is not None else [random.choice(root_node.untried_moves)]

        # Run MCTS starting from the current board state.
        if np.count_nonzero(self.board) == 0:
            best_move = [(self.size // 2, self.size // 2)]
        else:
            if self.root_node is None:
                self.root_node = MCTSNode(np.copy(self.board), my_color, 1)
            else:
                root = None
                for child_node in self.root_node.children:
                    for grandchild_node in child_node.children:
                        if grandchild_node.board == self.board:
                            root = grandchild_node
                            break
                if root is None:
                    root = MCTSNode(np.copy(self.board), my_color, 1)
                self.root_node = root

            best_move_1 = mcts_search(self.root_node, my_color, moves_left=1, iterations=30)
            # copy_board = np.copy(self.board)
            # for r, c in best_move_1:
            #     copy_board[r, c] = my_color
            for child in self.root_node.children:
                if child.move == best_move_1[0]:
                    self.root_node = child
                    break
            # root_node = MCTSNode(np.copy(self.board), my_color, 1)
            best_move_2 = mcts_search(self.root_node, my_color, moves_left=0, iterations=30)
            for child in self.root_node.children:
                if child.move == best_move_2[0]:
                    self.root_node = child
                    break
            best_move = best_move_1 + best_move_2

        if best_move:
            move_str = move_to_str(best_move)
            self.play_move(color, move_str)
            print(move_str, flush=True)
            print("MCTS", move_str, file=sys.stderr)
        else:
            # Fallback to a random move if MCTS fails (should rarely happen)
            empty_positions = [(r, c) for r in range(self.size) for c in range(self.size) if self.board[r, c] == 0]
            selected = random.choice(empty_positions)
            move_str = f"{self.index_to_label(selected[1])}{selected[0] + 1}"
            self.play_move(color, move_str)
            print(move_str, flush=True)
            print("Random", move_str, file=sys.stderr)

    def show_board(self):
        """Displays the board in text format."""
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
            print("env_board_size=19", flush=True)

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
