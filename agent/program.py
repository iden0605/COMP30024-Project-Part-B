# COMP30024 Artificial Intelligence, Semester 1 2026
# Project Part B: Game Playing Agent
# Iden Patrick McElhone (1543498)
# Meidelline Surya (1492043)

from referee.game import (
    PlayerColor, Action, PlaceAction, MoveAction, EatAction, CascadeAction,
    PLACEMENT_TURNS,
)

from .game import apply_place, apply_move, apply_eat, apply_cascade
from .search import best_action, best_place


class Agent:
    """
    Entry point for the agent, called by the referee to play Cascade.
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
        Return the best action for the current turn.
        Places during placement phase, searches during play phase.
        """
        if self._turn_count < PLACEMENT_TURNS:
            return best_place(self._board, self._colour)

        return best_action(self._board, self._colour)

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
