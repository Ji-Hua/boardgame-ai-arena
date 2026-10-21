"""Validated analysis positions and display-only shortest-path projections."""
from collections import deque
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from quoridor_engine import Action, Orientation, Player, RawState, RuleEngine, calculation

ENGINE = RuleEngine.standard()
Cell = Annotated[StrictInt, Field(ge=0, le=8)]
Anchor = Annotated[StrictInt, Field(ge=0, le=7)]
Stock = Annotated[StrictInt, Field(ge=0, le=10)]


class Wall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: Anchor
    y: Anchor
    orientation: Literal["horizontal", "vertical"]


class Position(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pawns: tuple[tuple[Cell, Cell], tuple[Cell, Cell]] = ((4, 0), (4, 8))
    walls_remaining: tuple[Stock, Stock] = (10, 10)
    walls: list[Wall] = Field(default_factory=list, max_length=20)
    turn: Literal[1, 2] = 1

    @model_validator(mode="after")
    def validate_inventory(self):
        if self.pawns[0] == self.pawns[1]:
            raise ValueError("双方棋子不能在同一格。")
        if self.pawns[0][1] == 8 and self.pawns[1][1] == 0:
            raise ValueError("双方不能同时处于获胜终点。")
        if sum(self.walls_remaining) + len(self.walls) != 20:
            raise ValueError("剩余墙数与棋盘上的墙数之和必须为 20。")
        return self


class Move(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["pawn", "horizontal", "vertical"]
    target: tuple[Cell, Cell]
    player: Literal[1, 2]

    def engine_action(self):
        player = Player.P1 if self.player == 1 else Player.P2
        x, y = self.target
        if self.type == "pawn":
            return Action.move_pawn(player, x, y)
        if x > 7 or y > 7:
            raise ValueError("墙锚点必须在 a1 到 h8 之间。")
        orientation = Orientation.Horizontal if self.type == "horizontal" else Orientation.Vertical
        return Action.place_wall(player, x, y, orientation)


def wire_action(action):
    kind = "pawn" if action.kind == "MovePawn" else (
        "horizontal" if action.coordinate_kind == "Horizontal" else "vertical")
    return {"player": 1 if action.player == Player.P1 else 2,
            "type": kind, "target": [action.target_x, action.target_y]}


def action_id(action):
    a = wire_action(action) if not isinstance(action, dict) else action
    return f"{a['type']}:{a['target'][0]}:{a['target'][1]}"


def notation(action):
    a = wire_action(action) if not isinstance(action, dict) else action
    x, y = a["target"]
    return f"{chr(97+x)}{y+1}" + {"pawn": "", "horizontal": "h", "vertical": "v"}[a["type"]]


def build_state(position: Position):
    """Reject malformed geometry before passing user input into the Rust FFI.

    Non-terminal wall configurations are additionally reconstructed through
    Rust wall placements. Editor positions need not have a known game history.
    """
    h_edges = v_edges = h_heads = v_heads = 0
    for wall in position.walls:
        x, y = wall.x, wall.y
        bit = 1 << (x * 8 + y)
        if (h_heads | v_heads) & bit:
            raise ValueError("墙不能重叠或在同一个锚点交叉。")
        if wall.orientation == "horizontal":
            edges = (1 << (x * 8 + y)) | (1 << ((x + 1) * 8 + y))
            if h_edges & edges:
                raise ValueError("横墙发生重叠。")
            h_edges |= edges
            h_heads |= bit
        else:
            edges = (1 << (x * 9 + y)) | (1 << (x * 9 + y + 1))
            if v_edges & edges:
                raise ValueError("竖墙发生重叠。")
            v_edges |= edges
            v_heads |= bit
    p1, p2 = position.pawns
    state = RawState(*p1, *p2, *position.walls_remaining,
                     h_edges, v_edges, h_heads, v_heads,
                     Player.P1 if position.turn == 1 else Player.P2)
    for player in (Player.P1, Player.P2):
        if not ENGINE.path_exists(state, player):
            raise ValueError("局面封死了某一方通往终点的所有路径。")
    if not ENGINE.is_game_over(state):
        probe = RawState(*p1, *p2, 10, 10, 0, 0, 0, 0, Player.P1)
        for wall in position.walls:
            # Stock and turn are reset only while validating an edited layout.
            probe = RawState(*p1, *p2, 10, 10, probe.horizontal_edges,
                             probe.vertical_edges, probe.horizontal_heads,
                             probe.vertical_heads, Player.P1)
            move = Move(type=wall.orientation, target=(wall.x, wall.y), player=1)
            probe = ENGINE.apply_action(probe, move.engine_action())
        assert (probe.horizontal_edges, probe.vertical_edges) == (h_edges, v_edges)
    return state


def to_position(state):
    walls = []
    for orientation, bits in (("horizontal", state.horizontal_heads),
                              ("vertical", state.vertical_heads)):
        for x in range(8):
            for y in range(8):
                if bits & (1 << (x * 8 + y)):
                    walls.append({"orientation": orientation, "x": x, "y": y})
    return {"pawns": [list(state.pawn_pos(Player.P1)), list(state.pawn_pos(Player.P2))],
            "walls_remaining": [state.walls_remaining(Player.P1), state.walls_remaining(Player.P2)],
            "walls": walls, "turn": 1 if state.current_player == Player.P1 else 2}


def distances(state):
    return [calculation.shortest_path_len(state, p, ENGINE.topology)
            for p in (Player.P1, Player.P2)]


def display_path(state, player):
    """A witness of the Rust distance heuristic, not a legal pawn-move plan.

    The heuristic ignores the other pawn. Its projected length is checked
    against Rust; legality and all real moves remain exclusively in Rust.
    """
    start = tuple(state.pawn_pos(player))
    goal = 8 if player == Player.P1 else 0
    queue, previous = deque([start]), {start: None}
    while queue:
        x, y = queue.popleft()
        if y == goal:
            path = [(x, y)]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            path.reverse()
            if len(path)-1 != calculation.shortest_path_len(state, player, ENGINE.topology):
                raise RuntimeError("Display path disagrees with the Rust heuristic")
            return path
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x+dx, y+dy
            if not (0 <= nx < 9 and 0 <= ny < 9) or (nx, ny) in previous:
                continue
            blocked = (state.horizontal_edges & (1 << (x*8+min(y, ny)))) if dy else (
                state.vertical_edges & (1 << (min(x, nx)*9+y)))
            if not blocked:
                previous[(nx, ny)] = (x, y)
                queue.append((nx, ny))
    raise ValueError("No path to goal")


def snapshot(state):
    winner = ENGINE.winner(state)
    return {"position": to_position(state), "distances": distances(state),
            "paths": [display_path(state, p) for p in (Player.P1, Player.P2)],
            "winner": None if winner is None else (1 if winner == Player.P1 else 2),
            "legal_actions": [wire_action(a) for a in ENGINE.legal_actions(state)]}


def presets():
    return [
        {"id": "opening", "title": "标准开局", "description": "P1 e1 / P2 e9，无墙。",
         "position": Position().model_dump(mode="json")},
        {"id": "threat", "title": "P2 终局威胁", "description": "P1 e3 / P2 e2，无墙，P1 行动。",
         "position": Position(pawns=((4, 2), (4, 1))).model_dump(mode="json")},
        {"id": "finish", "title": "P1 一步获胜", "description": "P1 e8 / P2 e4，无墙，P1 行动。",
         "position": Position(pawns=((4, 7), (4, 3))).model_dump(mode="json")},
        {"id": "jump", "title": "相邻棋子 / 横墙", "description": "P1 e4 / P2 e5，横墙 d5，P1 行动。",
         "position": Position(pawns=((4, 3), (4, 4)), walls_remaining=(9, 10),
                              walls=[Wall(x=3, y=4, orientation="horizontal")]).model_dump(mode="json")},
    ]
