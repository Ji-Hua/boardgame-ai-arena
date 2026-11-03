import json
from pathlib import Path
import tempfile
import unittest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from agent_system.games.contracts import KUHN_POKER
from agent_system.games.openspiel import KuhnPoker
from apps.gameplay.research import ResearchGames
from apps.gameplay.research_api import router

class ResearchGamesTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        from apps.jobs.projects import Projects
        projects=Projects(self.root/"jobs"/"projects.sqlite")
        projects.create(dict(id="poker",title="Poker",question="test",game_ref=KUHN_POKER.wire()))
        self.games=ResearchGames(self.root/"games.sqlite",self.root)
        app=FastAPI();app.include_router(router(self.games));self.api=TestClient(app)

    def create(self, role="p1"):
        return self.games.create("poker",KUHN_POKER.wire(),role)

    def test_role_and_watch_capabilities_never_expose_internal_state(self):
        created=self.create();gid=created["id"]
        for token in (created["token"],created["watch_token"]):
            for path in ("", "/replay", "/context"):
                response=self.api.get("/api/v1/play-v2/"+gid+path,headers={"X-Game-Token":token})
                self.assertEqual(response.status_code,200)
                text=response.text
                for key in ('"env_rng"','"agent_rng"','"tokens"','"initial"','"events"','"digest"'):
                    self.assertNotIn(key,text)
        watch=self.games.view(gid,created["watch_token"])
        self.assertNotIn("information_state",watch["observation"])
        self.assertEqual(watch["legal_actions"],[])
        denied=self.api.post("/api/v1/play-v2/"+gid+"/step",json=dict(revision=0,action=0),
                             headers={"X-Game-Token":created["watch_token"]})
        self.assertEqual(denied.status_code,403)
        self.assertEqual(self.api.get("/api/v1/play-v2/"+gid).status_code,403)

    def test_same_information_set_same_player_and_watch_projection(self):
        created=self.create();gid=created["id"];game=KuhnPoker()
        row=self.games.db.execute("select private from research_games where id=?",(gid,)).fetchone()
        record=json.loads(row[0]);views=[]
        for other in (1,2):
            state=game.apply(game.apply(game.initial_state(),0),other)
            record["state"]=game.serialize(state)
            record["events"]=[dict(actor="chance",action=0),dict(actor="chance",action=other)]
            self.games.db.execute("update research_games set private=? where id=?",(json.dumps(record),gid))
            views.append((self.games.view(gid,created["token"]),self.games.view(gid,created["watch_token"]),
                          self.games.replay(gid,created["token"])))
        self.assertEqual(views[0],views[1])

    def test_restart_preserves_agent_rng_and_revision_conflicts(self):
        created=self.create("p2");gid=created["id"];token=created["token"]
        other=ResearchGames(self.root/"clone.sqlite",self.root)
        self.games.db.backup(other.db)
        a=self.games.step(gid,token,0)
        b=other.step(gid,token,0)
        self.assertEqual(a,b)
        denied=self.api.post("/api/v1/play-v2/"+gid+"/step",json=dict(revision=0),
                             headers={"X-Game-Token":token})
        self.assertEqual(denied.status_code,409)
        restarted=ResearchGames(self.root/"games.sqlite",self.root)
        while a["status"]=="active":
            action=a["legal_actions"][0]["id"] if a["actor"]=="p2" else None
            a=restarted.step(gid,token,a["revision"],action)
        replay=restarted.replay(gid,token)
        self.assertEqual(len(replay["frames"]),a["revision"]+1)
        self.assertEqual(replay["frames"][-1]["observation"],a["observation"])

    def test_bad_policy_path_and_bool_action_fail_closed(self):
        with self.assertRaises(ValueError):
            self.games.create("poker",KUHN_POKER.wire(),"p1","../secret")
        created=self.create()
        result=self.api.post("/api/v1/play-v2/"+created["id"]+"/step",
            json=dict(revision=0,action=True),headers={"X-Game-Token":created["token"]})
        self.assertEqual(result.status_code,422)
        self.assertEqual(self.games.view(created["id"],created["token"])["revision"],0)


    def test_mrjack_private_projection_and_restore(self):
        from agent_system.games.contracts import MR_JACK
        from apps.jobs.projects import Projects
        Projects(self.root/"jobs"/"projects.sqlite").create(
            dict(id="jack",title="Jack",question="test",game_ref=MR_JACK.wire()))
        created=self.games.create("jack",MR_JACK.wire(),"detective","heuristic")
        gid,token=created["id"],created["token"]
        original=self.games.view(gid,token)
        assert original["observation"]["identity"] is None
        assert "identity" not in self.games.view(gid,created["watch_token"])["observation"]
        restarted=ResearchGames(self.root/"games.sqlite",self.root)
        assert original==restarted.view(gid,token)
        view=original
        while view["status"]=="active":
            action=view["legal_actions"][0]["id"] if view["actor"]=="detective" else None
            view=restarted.step(gid,token,view["revision"],action)
        assert restarted.replay(gid,token)["frames"][-1]["observation"]==view["observation"]
        assert restarted.context(gid,token)["complete"] is True
        with self.assertRaises(ValueError):
            restarted.create("poker",MR_JACK.wire(),"detective")
