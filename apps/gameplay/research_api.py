"""Separate v2 API: old Quoridor play capabilities cannot authorize these games."""
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from .games import Busy, Conflict, Denied

class NewResearchGame(BaseModel):
    model_config=ConfigDict(extra="forbid")
    project_id:str=Field(min_length=1,max_length=80)
    game_ref:dict
    human_role:str
    opponent:str=Field(default="random",max_length=64)

class ResearchStep(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    revision:int=Field(ge=0)
    action:int|None=None

def router(games):
    api=APIRouter(prefix="/api/v1/play-v2")
    def call(fn,*args):
        try:
            return fn(*args)
        except Denied:
            raise HTTPException(403,"Game unavailable or invalid capability")
        except Conflict as exc:
            raise HTTPException(409,str(exc))
        except Busy as exc:
            raise HTTPException(429,str(exc))
        except (ValueError,KeyError,TypeError):
            raise HTTPException(422,"Invalid or incompatible game request")
        except (OSError,RuntimeError,ImportError):
            raise HTTPException(503,"Required game or exact policy is unavailable")
    @api.post("")
    def create(body:NewResearchGame):
        return call(games.create,body.project_id,body.game_ref,body.human_role,body.opponent)
    @api.get("/{identifier}")
    def view(identifier:str,x_game_token:str=Header(default="")):
        return call(games.view,identifier,x_game_token)
    @api.post("/{identifier}/step")
    def step(identifier:str,body:ResearchStep,x_game_token:str=Header(default="")):
        return call(games.step,identifier,x_game_token,body.revision,body.action)
    @api.get("/{identifier}/replay")
    def replay(identifier:str,x_game_token:str=Header(default="")):
        return call(games.replay,identifier,x_game_token)
    @api.get("/{identifier}/context")
    def context(identifier:str,x_game_token:str=Header(default="")):
        return call(games.context,identifier,x_game_token)
    @api.delete("/{identifier}")
    def close(identifier:str,x_game_token:str=Header(default="")):
        return call(games.close,identifier,x_game_token)
    return api
