import json
from types import SimpleNamespace
import pytest
from agent_system.games.contracts import MR_JACK
from agent_system.definition.research_spec import policy_spec
from apps.jobs.research_eligibility import ResearchEligibility

def test_scoped_identity_and_manual_evidence(tmp_path):
    sha="a"*64
    bundle=dict(game_ref=MR_JACK.wire(),format="mrjack_mlp_tanh_v1",output_semantics="policy_logits")
    spec=policy_spec(bundle,sha,"detective")
    assert spec["hash"]!=policy_spec(bundle,sha,"jack")["hash"]
    with pytest.raises(ValueError):policy_spec(bundle,sha,"p1")
    job=dict(id="e",project_id="p",status="succeeded")
    report=dict(config=dict(candidate_sha=sha),digest="digest",protocol="mrjack-policy-paired-v1")
    c=SimpleNamespace(projects=SimpleNamespace(get=lambda p:None),
       policies=SimpleNamespace(get=lambda p,s:({},json.dumps(bundle))),
       store=SimpleNamespace(get=lambda j:job),research_evaluation=lambda j:report)
    audit=ResearchEligibility(tmp_path/"audit.sqlite",c)
    body=dict(sha256=sha,role="detective",target="research-play",evaluation_job="e",decision="approve",note="reviewed")
    row=audit.record("p",body)
    assert row["deployment_changed"] is False
    assert audit.listing("p")["current"][0]["decision"]=="approve"
    audit.record("p",dict(body,decision="withdraw"))
    assert len(audit.listing("p")["history"])==2
    assert len(audit.listing("p")["current"])==1
    with pytest.raises(ValueError):audit.record("p",dict(body,target="production"))
    with pytest.raises(ValueError):audit.record("other",body)
    report["config"]["candidate_sha"]="b"*64
    with pytest.raises(ValueError):audit.record("p",body)
