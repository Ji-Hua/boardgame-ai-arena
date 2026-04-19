"""Aggregation logic — compute win-rate matrix from game records."""

from __future__ import annotations

from collections import defaultdict

from agent_system.evaluation.arena.models import GameRecord


def compute_win_rate_matrix(
    games: list[GameRecord],
) -> dict[tuple[str, str], float]:
    """Compute win rates from a list of game records.

    Returns a dict mapping (agent_a, agent_b) -> win_rate_of_a.
    Each pair appears once; win_rate_of_b = 1 - win_rate_of_a (ignoring draws).
    """
    wins: dict[tuple[str, str], int] = defaultdict(int)
    totals: dict[tuple[str, str], int] = defaultdict(int)

    for game in games:
        # Normalize key so (A, B) is always alphabetically ordered
        a, b = sorted([game.agent_a, game.agent_b])
        key = (a, b)
        totals[key] += 1
        if game.winner == a:
            wins[key] += 1
        # If winner == b or None (draw), a gets no win credit

    matrix: dict[tuple[str, str], float] = {}
    for key, total in totals.items():
        matrix[key] = wins[key] / total if total > 0 else 0.0

    return matrix


def format_matrix_text(
    matrix: dict[tuple[str, str], float],
    agents: list[str],
) -> str:
    """Format win-rate matrix as a human-readable text table.

    Rows represent agent_a, columns represent agent_b.
    Cell value is win rate of row agent against column agent.
    """
    # Build a lookup for quick access: (row, col) -> win_rate of row agent
    lookup: dict[tuple[str, str], float] = {}
    for (a, b), rate in matrix.items():
        lookup[(a, b)] = rate
        lookup[(b, a)] = 1.0 - rate

    col_width = max(len(a) for a in agents) + 2
    header = " " * col_width + "".join(a.rjust(col_width) for a in agents)
    lines = [header]

    for row in agents:
        cells = []
        for col in agents:
            if row == col:
                cells.append("—".rjust(col_width))
            else:
                rate = lookup.get((row, col), 0.0)
                cells.append(f"{rate:.2f}".rjust(col_width))
        lines.append(row.ljust(col_width) + "".join(cells))

    return "\n".join(lines)


def format_pairwise_text(
    matrix: dict[tuple[str, str], float],
) -> str:
    """Format results as pairwise lines: 'A vs B → X% / Y%'."""
    lines = []
    for (a, b), rate_a in sorted(matrix.items()):
        rate_b = 1.0 - rate_a
        lines.append(f"{a} vs {b} → {rate_a:.0%} / {rate_b:.0%}")
    return "\n".join(lines)
