"""One role per episode; opponent/chance transitions are not learner steps."""
import copy
import random
import gymnasium as gym
from gymnasium import spaces
from agent_system.games.mrjack import MrJack,ACTION_COUNT
from agent_system.games.mrjack_encoding import encode,mask
from agent_system.games.mrjack_agents import MrJackHeuristic
from agent_system.games.contracts import Decision

class MrJackEnv(gym.Env):
    def __init__(self):
        self.game=MrJack()
        self.action_space=spaces.Discrete(ACTION_COUNT)
        size=len(encode(self.game.observe(self.game.initial_state(),"detective"),"detective"))
        self.observation_space=spaces.Box(-1,1,(size,),dtype="float32")
        self.state=None
        self.completed=[]
        self.total_engine_decisions=0

    def _advance(self):
        while self.game.actor(self.state) not in (self.role,None):
            actor=self.game.actor(self.state)
            if actor=="chance":
                outcomes=self.game.chance_outcomes(self.state)
                action=self.chance_rng.choices([a for a,_ in outcomes],weights=[p for _,p in outcomes])[0]
            else:
                action=self.opponent.act(Decision(actor,copy.deepcopy(self.game.observe(self.state,actor)),
                    self.game.legal_actions(self.state),self.decisions))
                self.decisions+=1;self.total_engine_decisions+=1
            self.state=self.game.apply(self.state,action)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        for _ in range(100):
            self.role=self.game.roles[int(self.np_random.integers(2))]
            self.chance_rng=random.Random(int(self.np_random.integers(2**32)))
            self.opponent=MrJackHeuristic();self.opponent.reset(int(self.np_random.integers(2**32)))
            self.state=self.game.initial_state();self.decisions=0
            self._advance()
            if self.game.actor(self.state) is not None:
                return encode(self.game.observe(self.state,self.role),self.role),{}
        raise RuntimeError("No learner decision reached in 100 resets")

    def action_masks(self):
        return mask(self.game.legal_actions(self.state))

    def step(self,action):
        self.state=self.game.apply(self.state,int(action))
        self.decisions+=1;self.total_engine_decisions+=1
        self._advance()
        terminated=self.game.actor(self.state) is None
        reward=self.game.returns(self.state).get(self.role,0.0)
        if terminated:self.completed.append(dict(role=self.role,return_=reward,engine_decisions=self.decisions))
        return encode(self.game.observe(self.state,self.role),self.role),reward,terminated,False,{}
