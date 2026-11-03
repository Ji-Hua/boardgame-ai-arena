"""Durable role-scoped game sessions. Private engine state never leaves this module."""
import copy
import hashlib
import hmac
import json
from pathlib import Path
import random
import secrets
import sqlite3
import threading

from agent_system.games.contracts import Decision
from agent_system.games.match import RandomAgent
from agent_system.games.registry import create_game, validate_ref
from agent_system.evaluation.mrjack_policies import load_policy
from .games import Busy, Conflict, Denied

def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()

def tuples(value):
    return tuple(tuples(v) for v in value) if isinstance(value, list) else value

class ResearchGames:
    def __init__(self, path, data_root, limit=64):
        self.root = Path(data_root)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("pragma journal_mode=wal")
        self.db.execute("create table if not exists research_games(id text primary key, revision integer not null, status text not null, private text not null)")
        self.lock = threading.RLock()
        self.limit = limit

    def _load(self, identifier, token):
        row = self.db.execute("select * from research_games where id=?", (identifier,)).fetchone()
        if row is None or row["status"] == "closed":
            raise Denied("Game unavailable or invalid capability")
        record = json.loads(row["private"])
        hashed = token_hash(token)
        viewer = next((r for r,h in record["tokens"].items() if hmac.compare_digest(h, hashed)), None)
        if viewer is None:
            raise Denied("Game unavailable or invalid capability")
        return dict(row), record, viewer

    def _agent(self, kind, project, ref):
        if kind == "heuristic":
            from agent_system.games.contracts import MR_JACK
            if ref!=MR_JACK.wire():raise ValueError("Unsupported heuristic")
            from agent_system.games.mrjack_agents import MrJackHeuristic
            return MrJackHeuristic()
        if kind == "random":
            return RandomAgent()
        return load_policy(self.root, kind, project, ref)

    @staticmethod
    def _settle(game, state, rng, events):
        count = 0
        while game.actor(state) == "chance":
            count += 1
            if count > 100:
                raise ValueError("Chance transition budget exceeded")
            outcomes = game.chance_outcomes(state)
            action = rng.choices([a for a,_ in outcomes], weights=[p for _,p in outcomes])[0]
            events.append(dict(actor="chance", action=action))
            state = game.apply(state, action)
        return state

    def create(self, project_id, game_ref, human_role, opponent="random"):
        validate_ref(game_ref)
        if not isinstance(project_id,str) or not project_id or len(project_id)>80:
            raise ValueError("Invalid project identity")
        projects=json.loads((self.root/"jobs"/"projects.json").read_text())
        if projects.get("schema_version")!=1:
            raise ValueError("Unsupported project catalog")
        project=next((p for p in projects["projects"] if p["id"]==project_id),None)
        if project is None or project["game_ref"]!=game_ref:
            raise ValueError("Unknown or incompatible project")
        game = create_game(game_ref)
        if human_role not in game.roles:
            raise ValueError("Invalid role")
        self._agent(opponent, project_id, game_ref)  # fail before creating a session
        token, watch = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        identifier = secrets.token_hex(16)
        initial = game.initial_state()
        rng = random.Random(secrets.randbits(128))
        events = []
        state = self._settle(game, initial, rng, events)
        record = dict(game_ref=game_ref, project_id=project_id, human_role=human_role,
            opponent=opponent, initial=game.serialize(initial), state=game.serialize(state),
            events=events, offsets=[len(events)], env_rng=rng.getstate(),
            agent_rng={r:random.Random(secrets.randbits(128)).getstate() for r in game.roles},
            tokens={human_role:token_hash(token), "watch":token_hash(watch)})
        with self.lock:
            self.db.execute("begin immediate")
            try:
                if self.db.execute("select count(*) from research_games where status='active'").fetchone()[0] >= self.limit:
                    raise Busy("Active game limit reached")
                self.db.execute("insert into research_games values(?,?,?,?)",
                    (identifier, 0, "active", json.dumps(record)))
                self.db.execute("commit")
            except Exception:
                self.db.execute("rollback"); raise
        return dict(id=identifier, token=token, watch_token=watch,
                    view=self.view(identifier,token))

    @staticmethod
    def _projection(game, state, viewer, can_act=True):
        actor = game.actor(state)
        observation = game.public_observe(state) if viewer == "watch" else game.observe(state,viewer)
        legal = game.legal_actions(state) if can_act and viewer == actor else ()
        return dict(actor=actor, observation=copy.deepcopy(observation),
            legal_actions=[dict(id=a,label=game.action_label(state,a)) for a in legal],
            returns=game.returns(state))

    def _view(self, row, record, viewer):
        game = create_game(record["game_ref"])
        state = game.restore(record["state"])
        return dict(id=row["id"], revision=row["revision"], status=row["status"],
            game_ref=record["game_ref"], project_id=record["project_id"], viewer=viewer,
            human_role=record["human_role"], opponent=record["opponent"],
            **self._projection(game,state,viewer))

    def view(self, identifier, token):
        with self.lock:
            return self._view(*self._load(identifier,token))

    def step(self, identifier, token, revision, action=None):
        with self.lock:
            row, record, viewer = self._load(identifier,token)
            if viewer != record["human_role"]:
                raise Denied("Read-only capability")
            if revision != row["revision"] or row["status"] != "active":
                raise Conflict("Game revision changed or game finished")
            game = create_game(record["game_ref"]); state = game.restore(record["state"])
            actor = game.actor(state)
            if actor == record["human_role"]:
                if type(action) is not int or action not in game.legal_actions(state):
                    raise ValueError("Invalid action")
            else:
                if action is not None:
                    raise ValueError("Agent action must be selected by its runtime")
                agent = self._agent(record["opponent"],record["project_id"],record["game_ref"])
                agent.reset(0); agent.rng.setstate(tuples(record["agent_rng"][actor]))
                action = agent.act(Decision(actor,copy.deepcopy(game.observe(state,actor)),
                                           game.legal_actions(state),row["revision"]))
                record["agent_rng"][actor] = agent.rng.getstate()
            state = game.apply(state,action)
            record["events"].append(dict(actor=actor,action=action))
            rng = random.Random(); rng.setstate(tuples(record["env_rng"]))
            state = self._settle(game,state,rng,record["events"])
            record["env_rng"] = rng.getstate()
            record["state"] = game.serialize(state)
            record["offsets"].append(len(record["events"]))
            status = "finished" if game.actor(state) is None else "truncated" if row["revision"]+1>=500 else "active"
            changed = self.db.execute("update research_games set revision=revision+1,status=?,private=? where id=? and revision=? and status='active'",
                (status,json.dumps(record),identifier,revision)).rowcount
            if changed != 1:
                raise Conflict("Game revision changed")
            row.update(revision=revision+1,status=status)
            return self._view(row,record,viewer)

    def replay(self, identifier, token):
        with self.lock:
            row, record, viewer = self._load(identifier,token)
            game = create_game(record["game_ref"]); state = game.restore(record["initial"])
            frames=[]; offset=0
            for boundary in record["offsets"]:
                for event in record["events"][offset:boundary]:
                    if game.actor(state) != event["actor"]:
                        raise ValueError("Corrupt game history")
                    state = game.apply(state,event["action"])
                offset=boundary
                frames.append(self._projection(game,state,viewer,can_act=False))
            if game.serialize(state) != record["state"]:
                raise ValueError("Corrupt game history")
            return dict(id=identifier,game_ref=record["game_ref"],viewer=viewer,
                        revision=row["revision"],frames=frames)

    def context(self, identifier, token):
        result=self.replay(identifier,token)
        return dict(schema_version=2,game_ref=result["game_ref"],viewer=result["viewer"],
                    revision=result["revision"],visible_history=result["frames"],
                    memory_mode="visible_history",complete=True)

    def close(self, identifier, token):
        with self.lock:
            row,record,viewer=self._load(identifier,token)
            if viewer!=record["human_role"]:
                raise Denied("Read-only capability")
            self.db.execute("update research_games set status='closed' where id=?",(identifier,))
            return dict(closed=True)
