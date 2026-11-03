"""Run a local adapter smoke match without touching platform services or catalogs."""
import argparse
import json
from pathlib import Path
from agent_system.games.contracts import QUORIDOR, MR_JACK
from agent_system.games.registry import create_game
from agent_system.games.match import RandomAgent, play, replay

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["quoridor", "mr_jack"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-decisions", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True, help="Private research replay, never a public download")
    args = parser.parse_args()
    ref = QUORIDOR if args.game == "quoridor" else MR_JACK
    game = create_game(ref.wire())
    result = play(game, {r: RandomAgent() for r in game.roles}, environment_seed=args.seed,
                  agent_seeds={r: args.seed + i + 1 for i, r in enumerate(game.roles)},
                  max_decisions=args.max_decisions)
    for role in game.roles:
        replay(game, result, role)
    # Exclusive creation: existing evidence is never overwritten.
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"game": ref.game_id, "termination": result["termination"],
                      "returns": result["returns"], "decisions": result["decisions"]}))

if __name__ == "__main__":
    main()
