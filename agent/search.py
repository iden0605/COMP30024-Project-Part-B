# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent
# Iden Patrick McElhone (1543498)
# Meidelline Surya (1492043)

from referee.game import (
    Coord, CARDINAL_DIRECTIONS,
    EatAction, CascadeAction,
    PLACEMENT_TURNS,
)

from .game import (
    get_actions, get_place_actions, token_count,
    in_bounds,
)

# Alpha-beta search depth
SEARCH_DEPTH = 3
INF = float('inf')

# Evaluation weights
W_TOKENS = 10
W_HEIGHTS = 3
W_THREATS = 2


def evaluate(board, my_colour):
    """
    Heuristic board score from my_colour's perspective.
    Uses a linear weighted sum of token advantage, max stack height advantage,
    and eat-threat advantage.
    Returns +INF or -INF for terminal states.
    """
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

    return W_TOKENS * token_diff + W_HEIGHTS * height_diff + W_THREATS * threat_diff


def order_actions(actions):
    """
    Sort actions for alpha-beta pruning efficiency.
    First try Eat, then Cascade, then Move.
    """
    def priority(item):
        action, new_board = item
        if isinstance(action, EatAction):
            return 0
        elif isinstance(action, CascadeAction):
            return 1
        else:
            return 2

    return sorted(actions, key=priority)


def alpha_beta(board, depth, alpha, beta, maximizing, my_colour):
    """
    Minimax search with alpha-beta pruning for my_colour.
    Returns the heuristic value of the board at the given search depth.
    Prunes branches where alpha >= beta.
    """
    my_tokens = token_count(board, my_colour)
    opp_tokens = token_count(board, my_colour.opponent)

    # Terminal state, one player has no tokens remaining
    if opp_tokens == 0:
        return INF
    if my_tokens == 0:
        return -INF

    # Depth cutoff, switch to heuristic evaluation
    if depth == 0:
        return evaluate(board, my_colour)

    colour = my_colour if maximizing else my_colour.opponent
    actions = order_actions(get_actions(board, colour))

    # No legal moves available, draw
    if not actions:
        return 0

    # Expand children, updating bounds and pruning when needed
    if maximizing:
        value = -INF
        for action, new_board in actions:
            value = max(value, alpha_beta(new_board, depth - 1, alpha, beta, False, my_colour))
            alpha = max(alpha, value)
            # beta cutoff
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for action, new_board in actions:
            value = min(value, alpha_beta(new_board, depth - 1, alpha, beta, True, my_colour))
            beta = min(beta, value)
            # alpha cutoff
            if beta <= alpha:
                break
        return value


def best_action(board, colour):
    """
    Return the best action for colour using alpha-beta search.
    Searches to SEARCH_DEPTH and returns the action with the highest score.
    """
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


def best_place(board, colour):
    """
    Return the best placement action for colour.
    Cells close to the board centre with good spread from own stacks are preferred.
    """
    options = get_place_actions(board, colour)
    if not options:
        return None

    center = 3.5
    best_act = None
    best_val = -INF

    # Collect own stack coordinates for spread calculation
    own = []
    for own_coord, cs in board.items():
        if cs.color == colour:
            own.append(own_coord)

    for action, new_board in options:
        r, c = action.coord.r, action.coord.c
        center_dist = abs(r - center) + abs(c - center)

        # Reward spread from own stacks for more board coverage
        if own:
            spread = min(abs(r - s.r) + abs(c - s.c) for s in own)
        else:
            spread = 0

        val = -center_dist * 2 + spread * 0.5
        if val > best_val:
            best_val = val
            best_act = action

    return best_act
