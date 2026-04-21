# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent
# Weight tuning script — tests evaluation weight combinations against a tokens-only baseline.
# Usage: python tune_weights.py
# Run from the project root directory.

import sys
import time

from referee.game import (
    PlayerColor, Coord, CARDINAL_DIRECTIONS,
    PlaceAction, MoveAction, EatAction, CascadeAction,
    PLACEMENT_TURNS, MAX_TURNS,
)

from agent.game import (
    apply_place, apply_move, apply_eat, apply_cascade,
    get_actions, token_count, in_bounds,
)

from agent.search import best_place, order_actions, SEARCH_DEPTH, INF

# W_TOKENS is fixed at 10 throughout; only W_HEIGHTS and W_THREATS are varied.
W_TOKENS = 10

WEIGHT_GRID = [
    (0, 0),   # baseline: tokens only
    (0, 2),
    (0, 5),
    (3, 0),
    (3, 2),   # current configuration
    (3, 5),
    (6, 0),
    (6, 2),
    (6, 5),
]

TUNE_GAMES = 1    # games per colour assignment per combination (2 total per pair)


def make_evaluate(w_heights, w_threats):
    """
    Return an evaluate function closed over the given weights.
    W_TOKENS is fixed at 10.
    """
    def evaluate(board, my_colour):
        opp = my_colour.opponent
        my_tokens = token_count(board, my_colour)
        opp_tokens = token_count(board, opp)

        if opp_tokens == 0:
            return INF
        if my_tokens == 0:
            return -INF

        token_diff = my_tokens - opp_tokens

        # Taller stacks can eat more and cascade further
        my_max = max((cs.height for cs in board.values() if cs.color == my_colour), default=0)
        opp_max = max((cs.height for cs in board.values() if cs.color == opp), default=0)
        height_diff = my_max - opp_max

        # Count adjacent enemy stacks that can be eaten
        my_threats = 0
        opp_threats = 0
        for coord, cs in board.items():
            for d in CARDINAL_DIRECTIONS:
                nr, nc = coord.r + d.r, coord.c + d.c
                if not in_bounds(nr, nc):
                    continue
                target = board.get(Coord(nr, nc))
                if target is None:
                    continue
                if cs.color == my_colour and target.color == opp and cs.height >= target.height:
                    my_threats += 1
                elif cs.color == opp and target.color == my_colour and cs.height >= target.height:
                    opp_threats += 1

        threat_diff = my_threats - opp_threats

        return W_TOKENS * token_diff + w_heights * height_diff + w_threats * threat_diff

    return evaluate


def make_alpha_beta(eval_fn):
    """
    Return an alpha_beta function closed over a custom evaluate function.
    """
    def alpha_beta(board, depth, alpha, beta, maximizing, my_colour):
        my_tokens = token_count(board, my_colour)
        opp_tokens = token_count(board, my_colour.opponent)

        if opp_tokens == 0:
            return INF
        if my_tokens == 0:
            return -INF
        if depth == 0:
            return eval_fn(board, my_colour)

        colour = my_colour if maximizing else my_colour.opponent
        actions = order_actions(get_actions(board, colour))

        if not actions:
            return 0

        if maximizing:
            value = -INF
            for action, new_board in actions:
                value = max(value, alpha_beta(new_board, depth - 1, alpha, beta, False, my_colour))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        else:
            value = INF
            for action, new_board in actions:
                value = min(value, alpha_beta(new_board, depth - 1, alpha, beta, True, my_colour))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value

    return alpha_beta


def make_weighted_agent(w_heights, w_threats):
    """
    Factory: returns an Agent class using the given evaluation weights.
    """
    eval_fn = make_evaluate(w_heights, w_threats)
    alpha_beta = make_alpha_beta(eval_fn)

    def best_action(board, colour):
        best_act = None
        best_val = -INF
        actions = order_actions(get_actions(board, colour))
        if not actions:
            return None
        for action, new_board in actions:
            val = alpha_beta(new_board, SEARCH_DEPTH - 1, -INF, INF, False, colour)
            if val > best_val:
                best_val = val
                best_act = action
        return best_act

    class WeightedAgent:
        def __init__(self, color, **referee):
            self._colour = color
            self._board = {}
            self._turn_count = 0

        def action(self, **referee):
            if self._turn_count < PLACEMENT_TURNS:
                return best_place(self._board, self._colour)
            return best_action(self._board, self._colour)

        def update(self, colour, action, **referee):
            self._turn_count += 1
            if isinstance(action, PlaceAction):
                self._board = apply_place(self._board, action.coord, colour)
            elif isinstance(action, MoveAction):
                self._board = apply_move(self._board, action.coord, action.direction)
            elif isinstance(action, EatAction):
                self._board = apply_eat(self._board, action.coord, action.direction)
            elif isinstance(action, CascadeAction):
                self._board = apply_cascade(self._board, action.coord, action.direction, colour)

    return WeightedAgent


def apply_action(board, action, colour):
    """
    Apply a single action to the board for colour.
    Returns the updated board dict.
    """
    if isinstance(action, PlaceAction):
        return apply_place(board, action.coord, colour)
    elif isinstance(action, MoveAction):
        return apply_move(board, action.coord, action.direction)
    elif isinstance(action, EatAction):
        return apply_eat(board, action.coord, action.direction)
    elif isinstance(action, CascadeAction):
        return apply_cascade(board, action.coord, action.direction, colour)
    return board


def play_game(red_class, blue_class):
    """
    Simulate a full game between two agent classes.
    Returns the winning PlayerColor, or None for a draw.
    """
    board = {}
    red = red_class(color=PlayerColor.RED)
    blue = blue_class(color=PlayerColor.BLUE)
    agents = {PlayerColor.RED: red, PlayerColor.BLUE: blue}
    colour = PlayerColor.RED

    for turn in range(PLACEMENT_TURNS + MAX_TURNS):
        action = agents[colour].action()

        if action is None:
            break

        board = apply_action(board, action, colour)

        for agent in agents.values():
            agent.update(colour, action)

        # Win conditions only apply after the placement phase is complete
        if turn >= PLACEMENT_TURNS - 1:
            if token_count(board, PlayerColor.BLUE) == 0:
                return PlayerColor.RED
            if token_count(board, PlayerColor.RED) == 0:
                return PlayerColor.BLUE

        colour = colour.opponent

    # Turn limit or stalemate: tiebreak by token count
    red_tokens = token_count(board, PlayerColor.RED)
    blue_tokens = token_count(board, PlayerColor.BLUE)
    if red_tokens > blue_tokens:
        return PlayerColor.RED
    elif blue_tokens > red_tokens:
        return PlayerColor.BLUE
    return None


def run_tuning():
    """
    Test each weight combination against the tokens-only baseline.
    Each combination plays TUNE_GAMES games as RED and TUNE_GAMES as BLUE.
    """
    baseline_class = make_weighted_agent(0, 0)
    results = {}

    print(f"Weight tuning: {len(WEIGHT_GRID)} combinations vs baseline (10, 0, 0)")
    print(f"Games per combination: {TUNE_GAMES * 2} ({TUNE_GAMES} as each colour)")
    print("-" * 55)

    for w_heights, w_threats in WEIGHT_GRID:
        label = f"(10, {w_heights}, {w_threats})"
        variant_class = make_weighted_agent(w_heights, w_threats)

        wins = 0
        losses = 0
        draws = 0
        total_time = 0

        for game_i in range(TUNE_GAMES):
            # Play as RED
            start = time.time()
            winner = play_game(variant_class, baseline_class)
            total_time += time.time() - start
            if winner == PlayerColor.RED:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1

            # Play as BLUE
            start = time.time()
            winner = play_game(baseline_class, variant_class)
            total_time += time.time() - start
            if winner == PlayerColor.BLUE:
                wins += 1
            elif winner is None:
                draws += 1
            else:
                losses += 1

        total_games = TUNE_GAMES * 2
        win_rate = wins / total_games
        results[(w_heights, w_threats)] = win_rate
        marker = " <-- current" if (w_heights, w_threats) == (3, 2) else ""
        print(f"  {label:<18} W:{wins} L:{losses} D:{draws}  "
              f"win rate: {win_rate:.2f}  ({total_time:.1f}s){marker}")

    print("-" * 55)
    best = max(results, key=results.get)
    print(f"Best combination: (W_TOKENS=10, W_HEIGHTS={best[0]}, W_THREATS={best[1]})  "
          f"win rate: {results[best]:.2f}")

    return results


if __name__ == "__main__":
    run_tuning()
