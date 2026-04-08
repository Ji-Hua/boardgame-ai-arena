"""Turn Orchestrator — routes agent turns after state transitions.

After each successful action (human or agent), the orchestrator checks
whether the next turn belongs to an agent and triggers automatic action.

Flow: Backend → Agent Service → Backend → Engine → Broadcast

Retry contract (per agent-interface.md):
  - Engine REJECT with reject_kind="GAME_END" → terminate immediately
  - Engine REJECT with reject_kind="INVALID_ACTION" → retry (all agent categories)
  - AdvanceCursor is called ONLY after Engine ACCEPT
"""

from __future__ import annotations

import asyncio
import logging
import time

from backend.adapters.agent_service_adapter import AgentServiceAdapter
from backend.application.game_manager import GameManager
from backend.application.room_manager import RoomManager, Room
from backend.runtime.broadcast import BroadcastHub

logger = logging.getLogger(__name__)

# Maximum number of RequestAction retries per turn before force-ending the game.
MAX_AGENT_RETRIES = 10


async def maybe_trigger_agent_turn(
    room: Room,
    room_id: str,
    gm: GameManager,
    rm: RoomManager,
    hub: BroadcastHub,
    agent_adapter: AgentServiceAdapter,
) -> None:
    """Check if the current turn belongs to an agent and execute it.

    This runs in a loop to handle consecutive agent turns (e.g., agent vs agent).
    Stops when:
    - It's a human player's turn
    - The game is over
    - An unrecoverable error occurs (timeout, agent crash, retry limit exhausted)
    """
    while True:
        if room.status != "using" or room.current_game_id is None:
            break

        game = gm.get_game(room.current_game_id)
        if game is None or game.phase != "running":
            break

        state = game.get_state()
        current_seat = state.get("current_player")
        if current_seat not in (1, 2):
            break

        # Check if the current seat is an agent
        if room.seats[current_seat].actor_type != "agent":
            break

        if not agent_adapter.has_agent(room_id, current_seat):
            break

        # Get legal actions for the agent
        legal_actions = game.engine.legal_pawn_actions()

        # Request initial action from agent
        try:
            action = await agent_adapter.request_action(
                room_id=room_id,
                seat=current_seat,
                game_state=state,
                legal_actions=legal_actions,
            )
        except (TimeoutError, RuntimeError) as e:
            logger.error("Agent action request failed for room=%s seat=%s: %s", room_id, current_seat, e)
            _force_end(game, room, room_id, gm, rm)
            await hub.broadcast(room_id, {
                "type": "game_ended",
                "game_id": game.game_id,
                "result": game.result,
            })
            break

        # Retry loop: apply contract retry semantics for all agent categories.
        # - REJECT with reject_kind="GAME_END"       → terminate immediately
        # - REJECT with reject_kind="INVALID_ACTION" → retry up to MAX_AGENT_RETRIES
        # - AdvanceCursor MUST NOT be called on reject
        accepted = False
        retries = 0
        while True:
            result = gm.submit_action(game, action)

            if result["success"]:
                accepted = True
                break

            if result.get("reject_kind") == "GAME_END":
                logger.warning(
                    "Agent action rejected (GAME_END) — terminating room=%s seat=%s",
                    room_id, current_seat,
                )
                break

            # INVALID_ACTION: retry
            retries += 1
            if retries > MAX_AGENT_RETRIES:
                logger.error(
                    "Agent retry limit (%d) exhausted for room=%s seat=%s",
                    MAX_AGENT_RETRIES, room_id, current_seat,
                )
                break

            logger.warning(
                "Agent action rejected (retry %d/%d) room=%s seat=%s: %s",
                retries, MAX_AGENT_RETRIES, room_id, current_seat, result.get("error"),
            )

            # Re-request action without advancing cursor (state is unchanged)
            try:
                action = await agent_adapter.request_action(
                    room_id=room_id,
                    seat=current_seat,
                    game_state=state,
                    legal_actions=legal_actions,
                )
            except (TimeoutError, RuntimeError) as e:
                logger.error(
                    "Agent retry request failed for room=%s seat=%s: %s", room_id, current_seat, e
                )
                break

        if not accepted:
            _force_end(game, room, room_id, gm, rm)
            await hub.broadcast(room_id, {
                "type": "game_ended",
                "game_id": game.game_id,
                "result": game.result,
            })
            break

        # Advance cursor ONLY after Engine ACCEPT.
        # This is a no-op for non-replay agents (service guards on hasattr "advance").
        agent_adapter.advance_agent(room_id, current_seat)

        # Broadcast state update (same path as human actions)
        await hub.broadcast(room_id, {
            "type": "state_update",
            "game_id": game.game_id,
            "state": result["state"],
            "last_action": action,
            "step_count": game.step_count,
        })

        if result.get("game_over"):
            await hub.broadcast(room_id, {
                "type": "game_ended",
                "game_id": game.game_id,
                "result": result["result"],
            })
            rm.set_config(room)
            agent_adapter.destroy_room_agents(room_id)
            break

        # Apply step delay for agent-vs-agent and replay modes (both seats are agents).
        # Speed control is ONLY applied here in the backend orchestration layer.
        # The engine and agent modules are never delayed.
        if room.seats[1].actor_type == "agent" and room.seats[2].actor_type == "agent":
            interval = 0.5 / game.speed_multiplier
            logger.info(
                "[SPEED] room=%s game=%s speed_multiplier=%.2f interval=%.4fs ts=%.3f",
                room_id,
                game.game_id,
                game.speed_multiplier,
                interval,
                time.monotonic(),
            )
            await asyncio.sleep(interval)
        else:
            # Yield to allow other coroutines (e.g., human input) to run
            await asyncio.sleep(0)


def _force_end(game, room, room_id: str, gm: GameManager, rm: RoomManager) -> None:
    """Force-end the game and reset the room to config state."""
    gm.force_end(game)
    rm.set_config(room)

