"""Append-only manual research eligibility; never modifies deployed rosters."""
import json
import sqlite3
import threading
import time
from agent_system.definition.research_spec import policy_spec
from agent_system.games.contracts import stable_hash

class ResearchEligibility:
    def __init__(self,path,controller):
        self.controller=controller
        self.db=sqlite3.connect(path,check_same_thread=False,isolation_level=None)
        self.lock=threading.RLock()
        self.db.execute("create table if not exists decisions(seq integer primary key,project text not null,payload text not null)")
    def listing(self,project):
        self.controller.projects.get(project)
        with self.lock:
            rows=[json.loads(r[0]) for r in self.db.execute(
                "select payload from decisions where project=? order by seq",(project,))]
        current={}
        for row in rows:current[row["scope_hash"]]=row
        return dict(history=rows,current=list(current.values()),deployment_changed=False)
    def record(self,project,body):
        required={"sha256","role","target","evaluation_job","decision","note"}
        if set(body)!=required or any(not isinstance(v,str) for v in body.values()):raise ValueError("Explicit policy, role, target, evaluation, decision and note required")
        if body["target"]!="research-play":raise ValueError("Only research-play eligibility is supported")
        if body["decision"] not in ("approve","withdraw"):raise ValueError("Unknown decision")
        if not isinstance(body["note"],str) or not 1<=len(body["note"].strip())<=2000:
            raise ValueError("A human decision note is required")
        _,raw=self.controller.policies.get(project,body["sha256"])
        spec=policy_spec(json.loads(raw),body["sha256"],body["role"])
        job=self.controller.store.get(body["evaluation_job"])
        if job is None or job["project_id"]!=project or job["status"]!="succeeded":
            raise ValueError("A succeeded evaluation in this project is required")
        report=self.controller.research_evaluation(job)
        if report["config"]["candidate_sha"]!=body["sha256"]:
            raise ValueError("Evaluation candidate mismatch")
        scope=dict(project_id=project,game_ref=spec["game_ref"],role=body["role"],
                   target=body["target"],agent_spec_hash=spec["hash"])
        row=dict(**scope,scope_hash=stable_hash(scope),agent_spec=spec,decision=body["decision"],
            evaluation_job=job["id"],evaluation_digest=report["digest"],protocol=report["protocol"],
            note=body["note"].strip(),created_at=time.time(),deployment_changed=False,
            gate_semantics="manual_review_of_completed_evaluation")
        with self.lock:
            self.db.execute("insert into decisions(project,payload) values(?,?)",(project,json.dumps(row)))
        return row
