# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent
# Monte Carlo Tree Search agent — comparison baseline only, not submitted as final agent
# Iden Patrick McElhone (1543498)
# Meidelline Surya (1492043)

import math
import random
import time

from referee.game import (
    PlayerColor, Action, PlaceAction, MoveAction, EatAction, CascadeAction,
    PLACEMENT_TURNS,
)

from agent.game import (
    apply_place, apply_move, apply_eat, apply_cascade,
    get_actions, token_count,
)

from agent.search import best_place

# MCTS tuning constants
MCTS_TIME = 0.2      # CPU seconds allocated per move (0.2s used for benchmarking)
UCB_C = 1.41         # exploration constant (sqrt(2))
ROLLOUT_DEPTH = 20   # max steps in each random rollout


class MCTSNode:
    """
    Node in the MCTS search tree.
    A class is used here because the tree structure requires mutable
    parent/child links and per-node statistics.
    """

    def __init__(self, board, colour, parent=None, action=None):
        self.board = board
        self.colour = colour    # player to move from this state
        self.parent = parent
        self.action = action    # action taken to reach this node
        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried = None     # lazily initialised list of unexpanded actions


def ucb1(node, parent_visits):
    """
    UCB1 selection score for a node.
    Returns +inf for unvisited nodes to guarantee initial exploration.
    """
    if node.visits == 0:
        return float('inf')
    exploit = node.wins / node.visits
    explore = UCB_C * math.sqrt(math.log(parent_visits) / node.visits)
    return exploit + explore


def select(node):
    """
    Descend the tree following the highest UCB1 child.
    Stops at the first node with untried actions or no children.
    """
    while node.children:
        if node.untried is None:
            node.untried = get_actions(node.board, node.colour)[:]
        if node.untried:
            break
        node = max(node.children, key=lambda c: ucb1(c, node.visits))
    return node


def expand(node):
    """
    Add one child by trying an untried action from node.
    Returns the new child node.
    """
    if node.untried is None:
        node.untried = get_actions(node.board, node.colour)[:]
    rand_action, new_board = node.untried.pop()
    child = MCTSNode(new_board, node.colour.opponent, parent=node, action=rand_action)
    node.children.append(child)
    return child


def rollout(board, colour, depth):
    """
    Random playout from the given state up to depth steps.
    Returns 1.0 if colour wins, 0.0 if colour loses, 0.5 for a draw.
    """
    for step in range(depth):
        if token_count(board, colour.opponent) == 0:
            return 1.0
        if token_count(board, colour) == 0:
            return 0.0
        actions = get_actions(board, colour)
        if not actions:
            return 0.5
        rand_action, board = random.choice(actions)
        colour = colour.opponent

    # Leaf: use token advantage as a proxy for win probability
    my_tokens = token_count(board, colour)
    opp_tokens = token_count(board, colour.opponent)
    if my_tokens > opp_tokens:
        return 0.6
    elif my_tokens < opp_tokens:
        return 0.4
    return 0.5


def backpropagate(node, result):
    """
    Propagate rollout result up the tree.
    Result flips at each level to account for alternating players.
    """
    while node is not None:
        node.visits += 1
        node.wins += result
        result = 1.0 - result
        node = node.parent


def mcts_best_action(board, colour, time_limit):
    """
    Run MCTS from the given board state within the CPU time budget.
    Returns the most-visited child's action as the best move.
    """
    root = MCTSNode(board, colour)
    deadline = time.process_time() + time_limit

    while time.process_time() < deadline:
        node = select(root)

        if node.untried is None:
            node.untried = get_actions(node.board, node.colour)[:]

        # Expand if there are untried actions and node is not terminal
        if node.untried:
            node = expand(node)

        result = rollout(node.board, node.colour, ROLLOUT_DEPTH)
        backpropagate(node, result)

    if not root.children:
        return None

    # Most-visited child is the most robust choice
    return max(root.children, key=lambda c: c.visits).action


class Agent:
    """
    MCTS-based agent used for comparison against alpha-beta.
    Uses the same placement strategy as the main agent.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        Initialise the agent with a colour and an empty board.
        """
        self._colour = color
        self._board = {}
        self._turn_count = 0

    def action(self, **referee: dict) -> Action:
        """
        Return a placement during the placement phase, else run MCTS.
        """
        if self._turn_count < PLACEMENT_TURNS:
            return best_place(self._board, self._colour)

        return mcts_best_action(self._board, self._colour, MCTS_TIME)

    def update(self, colour: PlayerColor, action: Action, **referee: dict):
        """
        Apply the last action to the internal board.
        """
        self._turn_count += 1
        if isinstance(action, PlaceAction):
            self._board = apply_place(self._board, action.coord, colour)
        elif isinstance(action, MoveAction):
            self._board = apply_move(self._board, action.coord, action.direction)
        elif isinstance(action, EatAction):
            self._board = apply_eat(self._board, action.coord, action.direction)
        elif isinstance(action, CascadeAction):
            self._board = apply_cascade(self._board, action.coord, action.direction, colour)
