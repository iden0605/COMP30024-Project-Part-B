# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent
# Iden Patrick McElhone (1543498)
# Meidelline Surya (1492043)

from referee.game import (
    CellState, Coord, CARDINAL_DIRECTIONS,
    PlaceAction, MoveAction, EatAction, CascadeAction,
    BOARD_N, INITIAL_STACK_HEIGHT,
)


def in_bounds(r, c):
    """
    Checks if coordinates (r, c) are within the board boundaries.
    """
    return 0 <= r < BOARD_N and 0 <= c < BOARD_N


def push_chain(board, r, c, dr, dc):
    """
    Recursively push the stack at (r, c) one step in direction (dr, dc).
    If the destination is taken, that stack is pushed first before moving.
    Stacks pushed off the board are removed.
    """
    nr = r + dr
    nc = c + dc
    src = Coord(r, c)

    # Remove stack if it goes off the board
    if not in_bounds(nr, nc):
        del board[src]
        return

    dst = Coord(nr, nc)

    # Push the stack at the destination if it exists
    if dst in board:
        push_chain(board, nr, nc, dr, dc)

    # Move the source stack to the destination
    board[dst] = board.pop(src)


def apply_move(board, coord, d):
    """
    Move the stack at coord one step in direction d.
    If the destination has a friendly stack, merge and add height.
    Returns a new board dict.
    """
    new = dict(board)
    dst = Coord(coord.r + d.r, coord.c + d.c)
    src = new.pop(coord)

    # Merge if destination stack is friendly
    if dst in new:
        new[dst] = CellState(src.color, src.height + new[dst].height)
    else:
        new[dst] = src

    return new


def apply_eat(board, coord, d):
    """
    Capture the enemy stack adjacent in direction d.
    The attacker moves into the enemy cell and the enemy tokens are removed.
    Returns a new board dict.
    """
    new = dict(board)
    dst = Coord(coord.r + d.r, coord.c + d.c)
    new[dst] = new.pop(coord)
    return new


def apply_cascade(board, coord, d, colour):
    """
    Cascade the stack at coord, spreading one token per cell along direction d.
    Any stack in a token's path is chain-pushed forward.
    Tokens and stacks that go off the board are eliminated.
    Returns a new board dict.
    """
    new = dict(board)
    height = new.pop(coord).height
    dr, dc = d.r, d.c

    # Spread tokens one by one
    for i in range(1, height + 1):
        tr, tc = coord.r + dr * i, coord.c + dc * i
        if not in_bounds(tr, tc):
            break

        target = Coord(tr, tc)
        if target in new:
            push_chain(new, tr, tc, dr, dc)

        new[target] = CellState(colour, 1)

    return new


def apply_place(board, coord, colour):
    """
    Place a new stack of height INITIAL_STACK_HEIGHT of the given colour at coord.
    Returns a new board dict.
    """
    new = dict(board)
    new[coord] = CellState(colour, INITIAL_STACK_HEIGHT)
    return new


def get_actions(board, colour):
    """
    Get (action, new_board) for every legal action for colour in the play phase.
    Possible actions are Move, Eat, and Cascade over all directions.
    """
    results = []
    opp = colour.opponent

    for coord, cs in board.items():
        if cs.color != colour:
            continue

        for d in CARDINAL_DIRECTIONS:
            nr, nc = coord.r + d.r, coord.c + d.c
            dst = board.get(Coord(nr, nc)) if in_bounds(nr, nc) else None

            # Move: target must be empty or a friendly stack
            if in_bounds(nr, nc) and (dst is None or dst.color == colour):
                results.append((MoveAction(coord, d), apply_move(board, coord, d)))

            # Eat: target must be an enemy stack with at most our height
            if dst is not None and dst.color == opp and cs.height >= dst.height:
                results.append((EatAction(coord, d), apply_eat(board, coord, d)))

            # Cascade: stack must have at least 2 tokens to spread
            if cs.height >= 2:
                results.append((CascadeAction(coord, d), apply_cascade(board, coord, d, colour)))

    return results


def get_place_actions(board, colour):
    """
    Get (action, new_board) for every legal placement for the given colour.
    Cannot place on occupied cells or adjacent to any opponent stack.
    """
    opp = colour.opponent
    results = []

    for r in range(BOARD_N):
        for c in range(BOARD_N):
            coord = Coord(r, c)
            if coord in board:
                continue

            # Block placement adjacent to opponent stacks
            blocked = False
            for d in CARDINAL_DIRECTIONS:
                nr, nc = r + d.r, c + d.c
                if not in_bounds(nr, nc):
                    continue
                neighbor = board.get(Coord(nr, nc))
                if neighbor is not None and neighbor.color == opp:
                    blocked = True
                    break

            if not blocked:
                results.append((PlaceAction(coord), apply_place(board, coord, colour)))

    return results


def token_count(board, colour):
    """
    Return the sum of stack heights for the given colour.
    """
    return sum(cs.height for cs in board.values() if cs.color == colour)
