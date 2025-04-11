import numpy as np
from game import Board, UP, RIGHT, DOWN, LEFT, action_name
from game import IllegalAction, GameOver

# TUPLES = [
#     [0, 1, 2, 4, 5, 6],
#     [1, 2, 5, 6, 9, 13],
#     [0, 1, 2, 3, 4, 5],
#     [0, 1, 5, 6, 7, 10],
#     [0, 1, 2, 5, 9, 10],
#     [0, 1, 5, 9, 13, 14],
#     [0, 1, 5, 8, 9, 13],
#     [0, 1, 2, 4, 6, 10]
# ]

class nTupleNewrok:
    def __init__(self, tuples, v_init):
        self.TUPLES = self.generate_symmetric_tuples(tuples)   
        self.TARGET_PO2 = 15
        self.v_init = v_init
        self.LUTS = self.initialize_LUTS(tuples)
        

    def rotate_tuple(self, tp):
        """rotate a tuple on a 4x4 board by 90 degrees"""
        rotated = []
        for coord in tp:
            x, y = coord // 4, coord % 4
            rotated.append(y * 4 + (3 - x))
        return rotated
    
    def horizontal_reflect_tuple(self, tp):
        """reflect a tuple on the horizontal axis of a 4x4 board"""
        reflected = []
        for coord in tp:
            x, y = coord // 4, coord % 4
            reflected.append((3 - x) * 4 + y)
        return reflected

    def generate_symmetric_tuples(self, tuples):
        """generate all the symmetric tuples of the given tuples"""
        sym_tuples = []
        for tp in tuples:
            group = []
            group.append(tp)
            group.append(self.horizontal_reflect_tuple(tp))
            for _ in range(3):
                tp = self.rotate_tuple(tp)
                group.append(tp)
                group.append(self.horizontal_reflect_tuple(tp))
            sym_tuples.append(group)
        return sym_tuples

    def initialize_LUTS(self, tuples):
        LUTS = []
        for tp in tuples:
            LUTS.append(np.full((self.TARGET_PO2 + 1) ** len(tp), self.v_init))
        # print(f"initialize LUTS with {len(tuples)} tuples")
        return LUTS

    def tuple_id(self, values):
        values = values[::-1]
        k = 1
        n = 0
        for v in values:
            if v >= self.TARGET_PO2:
                raise ValueError(
                    "digit %d should be smaller than the base %d" % (v, self.TARGET_PO2)
                )
            n += v * k
            k *= self.TARGET_PO2
        return n

    def V(self, board, delta=None, debug=False):
        """Return the expected total future rewards of the board.
        Updates the LUTs if a delta is given and return the updated value.
        """
        if debug:
            print(f"V({board})")
        vals = []
        for i, (sym_group, LUT) in enumerate(zip(self.TUPLES, self.LUTS)):
            for tp in sym_group:
                tiles = [board[i] for i in tp]
                tpid = self.tuple_id(tiles)
                if delta is not None:
                    LUT[tpid] += delta
                v = LUT[tpid]
                if debug:
                    print(f"LUTS[{i}][{tiles}]={v}")
                vals.append(v)
        return np.mean(vals)

    def evaluate(self, s, a):
        "Return expected total rewards of performing action (a) on the given board state (s)"
        b = Board(s)
        try:
            r = b.act(a)
            s_after = b.copyboard()
        except IllegalAction:
            return 0
        return r + self.V(s_after)

    def best_action(self, s):
        "returns the action with the highest expected total rewards on the state (s)"
        a_best = None
        r_best = -1
        for a in [UP, RIGHT, DOWN, LEFT]:
            r = self.evaluate(s, a)
            if r > r_best:
                r_best = r
                a_best = a
        return a_best

    def learn(self, s, a, r, s_after, s_next, alpha=0.01, debug=False):
        """Learn from a transition experience by updating the belief
        on the after state (s_after) towards the sum of the next transition rewards (r_next) and
        the belief on the next after state (s_after_next).

        """
        a_next = self.best_action(s_next)
        b = Board(s_next)
        try:
            r_next = b.act(a_next)
            s_after_next = b.copyboard()
            v_after_next = self.V(s_after_next)
        except IllegalAction:
            r_next = 0
            v_after_next = 0

        delta = r_next + v_after_next - self.V(s_after)

        if debug:
            print("s_next")
            Board(s_next).display()
            print("a_next", action_name(a_next), "r_next", r_next)
            print("s_after_next")
            Board(s_after_next).display()
            self.V(s_after_next, debug=True)
            print(
                f"delta ({delta:.2f}) = r_next ({r_next:.2f}) + v_after_next ({v_after_next:.2f}) - V(s_after) ({V(s_after):.2f})"
            )
            print(
                f"V(s_after) <- V(s_after) ({V(s_after):.2f}) + alpha * delta ({alpha} * {delta:.1f})"
            )
        self.V(s_after, alpha * delta)
