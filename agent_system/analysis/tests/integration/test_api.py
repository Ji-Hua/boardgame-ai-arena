# TEST_CLASSIFICATION: SPECIFIED
# Spec: documents/agent/analysis/decision-analysis-interface.md
from fastapi.testclient import TestClient

from agent_system.analysis.app import app

client = TestClient(app)
POSITION = {"pawns": [[4, 0], [4, 8]], "walls_remaining": [10, 10], "walls": [], "turn": 1}


def test_independent_app_assets_and_presets():
    assert client.get("/health").json()["component"] == "decision-lab"
    page = client.get("/")
    assert page.status_code == 200
    assert '/assets/app.js' in page.text
    for asset in ("app.js", "board.js", "neural.js", "style.css"):
        assert client.get(f"/assets/{asset}").status_code == 200
    meta = client.get("/api/meta").json()
    assert len(meta["provenance"]["source_sha256"]) == 64
    for preset in meta["presets"]:
        response = client.post("/api/position", json=preset["position"])
        assert response.status_code == 200
        assert response.json()["legal_actions"]


def test_http_analyze_inspect_apply_and_import_roundtrip():
    request = {"position": POSITION, "algorithm": "minimax", "depth": 2}
    analyzed = client.post("/api/analyze", json=request)
    assert analyzed.status_code == 200
    result = analyzed.json()
    candidate = next(c for c in result["candidates"] if c["id"] == result["selected_id"])
    inspected = client.post("/api/branch", json={**request, "action": candidate["action"]}).json()
    assert inspected["bound"] == "exact"
    assert inspected["position_hash"] == result["position_hash"]
    applied = client.post("/api/apply", json={**request, "action": candidate["action"]}).json()
    assert applied == inspected["after"]
    assert applied["position"]["turn"] == 2
    imported = client.post("/api/position", json=applied["position"]).json()
    assert imported == applied
    # Analysis never modifies the supplied input, nor shares state with later requests.
    assert client.post("/api/position", json=POSITION).json()["position"] == POSITION


def test_invalid_inputs_and_illegal_actions_are_rejected():
    assert client.post("/api/analyze", json={"position": POSITION, "depth": 10}).status_code == 422
    assert client.post("/api/analyze", json={"position": POSITION, "algorithm": "cnn"}).status_code == 422
    assert client.post("/api/position", json={**POSITION, "walls_remaining": [0, 0]}).status_code == 422
    for endpoint in ("/api/branch", "/api/apply"):
        for action in [
            {"type": "pawn", "target": [4, 7], "player": 1},
            {"type": "pawn", "target": [4, 7], "player": 2},
            {"type": "horizontal", "target": [8, 7], "player": 1},
        ]:
            assert client.post(endpoint, json={"position": POSITION, "action": action}).status_code == 422
