# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent
# Benchmark script — runs games directly between two agent classes without the referee subprocess.
# Usage: python benchmark.py [n_games]   (default: 20)
# Run from the project root directory.

import sys
import time

from referee.game import (
    PlayerColor, PlaceAction, MoveAction, EatAction, CascadeAction,
    PLACEMENT_TURNS, MAX_TURNS,
)

from agent.game import (
    apply_place, apply_move, apply_eat, apply_cascade, token_count,
)

from agent.program import Agent as AlphaBetaAgent
from mcts_agent.program import Agent as MCTSAgent


def apply_action(board, action, colour):
    """
    Apply a single action to board for the given colour.
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

        # Stalemate: no legal move available
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


def run_benchmark(n_games):
    """
    Run n_games games, alternating which agent plays RED and BLUE.
    Prints per-game results and a final summary.
    """
    ab_wins = 0
    mcts_wins = 0
    draws = 0
    game_times = []

    print(f"Running {n_games} games: Alpha-Beta vs MCTS")
    print("-" * 45)

    for i in range(n_games):
        # Alternate colours so neither agent has a structural advantage
        if i % 2 == 0:
            red_class = AlphaBetaAgent
            blue_class = MCTSAgent
            ab_colour = PlayerColor.RED
        else:
            red_class = MCTSAgent
            blue_class = AlphaBetaAgent
            ab_colour = PlayerColor.BLUE

        start = time.time()
        winner = play_game(red_class, blue_class)
        elapsed = time.time() - start
        game_times.append(elapsed)

        if winner == ab_colour:
            ab_wins += 1
            result_str = "Alpha-Beta wins"
        elif winner is None:
            draws += 1
            result_str = "Draw"
        else:
            mcts_wins += 1
            result_str = "MCTS wins"

        print(f"  Game {i + 1:2d}: {result_str:<18} ({elapsed:.1f}s)")

    avg_time = sum(game_times) / len(game_times)
    print("-" * 45)
    print(f"Results over {n_games} games:")
    print(f"  Alpha-Beta wins : {ab_wins}")
    print(f"  MCTS wins       : {mcts_wins}")
    print(f"  Draws           : {draws}")
    print(f"  Avg game time   : {avg_time:.1f}s")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    run_benchmark(n)
