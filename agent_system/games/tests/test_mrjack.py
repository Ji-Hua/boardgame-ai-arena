import copy
import random
import pytest
from agent_system.games.mrjack import MrJack
from agent_system.games.mrjack_board import EXITS, INITIAL_LIGHTS, SEWERS
from agent_system.games.mrjack_agents import MrJackHeuristic
from agent_system.games.match import play, replay
from agent_system.games.contracts import Decision

def ready(character=0, role="detective"):
    game=MrJack()
    s=game.apply(game.initial_state(),4)
    for c in (0,2,4,6):s=game.apply(s,c)
    s.update(hand=list(range(8)),selected=character,phase="move",turn=0 if role=="detective" else 1)
    return game,s

def test_setup_chance_and_private_information():
    g,s=ready()
    assert g.visible(s)==[True,False,True,False,True,True,False,False]
    other=copy.deepcopy(s);other["jack"]=5;other["alibis"]=[0,1,2,3,4,6,7]
    assert g.observe(s,"detective")==g.observe(other,"detective")
    assert g.public_observe(s)==g.public_observe(other)
    assert g.observe(s,"jack")!=g.observe(other,"jack")
    assert g.legal_actions(s)==g.legal_actions(other)
    s["private"]["jack"]=[1]
    assert 1 not in g.observe(s,"detective")["private_alibis"]

@pytest.mark.parametrize("character,phase",[(0,"ability"),(2,"ability"),(3,"orient"),(4,"alibi"),(5,"ability"),(6,"ability"),(7,"choose"),(1,"choose")])
def test_all_character_move_continuations(character,phase):
    g,s=ready(character)
    a=next(a for a in g.legal_actions(s) if 100<=a<186 and a-100 not in s["positions"])
    result=g.apply(s,a)
    assert result["phase"]==phase
    assert s["positions"]!=result["positions"]
    assert s["moved"] is False

@pytest.mark.parametrize("character,key",[(0,"cordons"),(2,"covered"),(5,"lights")])
def test_mandatory_abilities_before_and_after(character,key):
    for before in (True,False):
        g,s=ready(character);s["phase"]="order"
        s=g.apply(s,10 if before else 11)
        if not before:
            s=g.apply(s,next(a for a in g.legal_actions(s) if 100<=a<186 and a-100 not in s["positions"]))
        old=s[key][:]
        s=g.apply(s,g.legal_actions(s)[0])
        assert old!=s[key]
        assert s["phase"]==("move" if before else "choose")

def test_holmes_alibi_only_activating_role():
    g,s=ready(4,"jack")
    s=g.apply(s,next(a for a in g.legal_actions(s) if a-100 not in s["positions"]))
    card=g.legal_actions(s)[0]
    assert card!=s["jack"]
    s=g.apply(s,card)
    assert g.observe(s,"jack")["private_alibis"]==[card]
    assert g.observe(s,"detective")["private_alibis"]==[]
    assert "private_alibis" not in g.public_observe(s)

def test_accusation_correct_and_wrong_and_jack_cannot_accuse():
    g,s=ready(0)
    s["positions"][0]=5;s["positions"][4]=6;s["jack"]=4
    assert 106 in g.legal_actions(s)
    assert g.apply(s,106)["winner"]=="detective"
    s["jack"]=1
    assert g.apply(s,106)["winner"]=="jack"
    s["turn"]=1
    assert 106 not in g.legal_actions(s)

def test_escape_requires_jack_identity_invisible_and_open_exit():
    g,s=ready(0,"jack");s["jack"]=0;s["positions"][0]=EXITS[0]
    assert 230 not in g.legal_actions(s)
    s["witness"]=False
    assert g.apply(s,230)["winner"]=="jack"
    s["cordons"][0]=EXITS[0]
    assert 230 not in g.legal_actions(s)
    s["jack"]=1
    assert not any(230<=a<234 for a in g.legal_actions(s))

def test_gull_swap_and_watson_orientation():
    g,s=ready(7)
    before=s["positions"][:];s=g.apply(s,203)
    assert s["positions"][7]==before[3] and s["positions"][3]==before[7]
    assert s["phase"]=="orient"
    s=g.apply(s,220)
    assert s["phase"]=="choose" and s["orientation"]==0

def test_goodley_budget_surface_distance_and_orientation():
    g,s=ready(6);s.update(phase="ability",pull_remaining=3)
    while s["phase"]=="ability":
        actions=g.legal_actions(s)
        assert actions
        a=actions[0];who,remainder=divmod(a-1000,258);dest,k=divmod(remainder,3)
        distance=g.distances(s,s["positions"][6],tunnels=False)
        assert distance[dest]<distance[s["positions"][who]]
        budget=s["pull_remaining"]
        s=g.apply(s,a)
        assert s["pull_remaining"]==budget-k-1
        if s["phase"]=="orient":s=g.apply(s,220)
    assert s["phase"]=="move"

def test_witness_elimination_lamp_number_and_eight_round_survival():
    g,s=ready(7);s.update(turn=3,hand=[7],round=1)
    # Exchange does not create an accusation.
    s=g.apply(s,200)
    assert s["round"]==2 and s["turn"]==0 and s["lights"][1]==-1
    assert s["jack"] in s["suspects"]
    g,s=ready(7);s.update(turn=3,hand=[7],round=8)
    assert g.apply(s,200)["winner"]=="jack"

def test_tunnel_and_stealth_paths():
    g,s=ready(1)
    assert g.distances(s,6)[15]==1
    s["covered"]=[6,15]
    assert g.distances(s,6).get(15,99)>1
    normal=g.distances(s,2);stealth=g.distances(s,2,stealth=True)
    assert all(stealth[k]<=v for k,v in normal.items())
    assert any(k not in normal for k in stealth)

def test_seeded_matches_replay_and_heuristic_never_peeks():
    g=MrJack()
    for seed in range(20):
        record=play(g,{r:MrJackHeuristic() for r in g.roles},environment_seed=seed)
        assert record["termination"]=="engine_terminal"
        assert sum(record["returns"].values())==0
        for role in g.roles:
            assert replay(g,record,role)==g.observe(g.restore(record["final"]),role)
    g,s=ready()
    for action in (True,-1,3072):
        with pytest.raises(ValueError):g.apply(s,action)
