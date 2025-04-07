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
    def __init__(self, board_size, patterns, v_init = 0.0):
        """ Initializes the N-Tuple approximator. 'patterns' is a list of base tuple patterns (each a list of (row, col) tuples). """
        self.board_size = board_size
        self.patterns = patterns
        # Create one weight dictionary per base pattern (shared across its symmetric variants)
        if v_init:
            def constant():
                return v_init
            self.weights = [defaultdict(constant) for _ in patterns]
        else:
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
            total_value += (group_value)
        return total_value

    def update(self, board, delta, alpha):
        # Update weights based on the TD error.
        num_patterns = len(self.patterns)
        for group_idx, group in enumerate(self.symmetry_groups):
            num_sym = len(group)
            for sym in group:
                feature = self.get_feature(board, sym)
                self.weights[group_idx][feature] += (alpha / num_sym) / num_patterns * delta