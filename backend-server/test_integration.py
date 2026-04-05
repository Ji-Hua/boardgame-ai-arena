"""Integration test — human vs human full game flow."""

import requests
import json

BASE = "http://localhost:8000"

def test_full_flow():
    # 1. Create session
    r = requests.post(f"{BASE}/session/create")
    assert r.status_code == 200, f"Create failed: {r.text}"
    data = r.json()
    sid = data["session_id"]
    assert data["status"] == "waiting"
    print(f"1. Created session: {sid}")

    # 2. Join seat 1
    r = requests.post(f"{BASE}/session/{sid}/join",
                      json={"client_id": "alice", "seat": 1})
    assert r.status_code == 200, f"Join 1 failed: {r.text}"
    print("2. Alice joined seat 1")

    # 3. Join seat 2
    r = requests.post(f"{BASE}/session/{sid}/join",
                      json={"client_id": "bob", "seat": 2})
    assert r.status_code == 200, f"Join 2 failed: {r.text}"
    print("3. Bob joined seat 2")

    # 4. Start game
    r = requests.post(f"{BASE}/session/{sid}/start")
    assert r.status_code == 200, f"Start failed: {r.text}"
    game = r.json()
    assert game["status"] == "active"
    print(f"4. Game started: {json.dumps(game['game_state'], indent=2)}")

    # 5. Get state
    r = requests.get(f"{BASE}/session/{sid}/state")
    assert r.status_code == 200
    state = r.json()
    assert state["status"] == "active"
    gs = state["game_state"]
    assert gs["current_player"] == 1
    assert gs["pawns"]["1"] == {"x": 4, "y": 0}
    assert gs["pawns"]["2"] == {"x": 4, "y": 8}
    print(f"5. State OK: P1 at (4,0), P2 at (4,8)")

    # 6. Get legal actions
    r = requests.get(f"{BASE}/session/{sid}/legal_actions")
    assert r.status_code == 200
    la = r.json()["legal_actions"]
    assert len(la) == 131, f"Expected 131 legal actions, got {len(la)}"
    print(f"6. Legal actions: {len(la)}")

    # 7. Submit action: P1 moves forward (4,0) -> (4,1)
    r = requests.post(f"{BASE}/session/{sid}/action",
                      json={"player": 1, "type": "pawn", "target": [4, 1]})
    assert r.status_code == 200, f"Action failed: {r.text}"
    res = r.json()
    assert res["success"]
    assert res["state"]["current_player"] == 2
    assert res["state"]["pawns"]["1"] == {"x": 4, "y": 1}
    print("7. P1 moved to (4,1)")

    # 8. Wrong turn (P1 tries again)
    r = requests.post(f"{BASE}/session/{sid}/action",
                      json={"player": 1, "type": "pawn", "target": [4, 2]})
    assert r.status_code == 400, f"Expected 400, got {r.status_code}"
    print(f"8. Wrong turn correctly rejected: {r.json()['detail']}")

    # 9. P2 moves forward (4,8) -> (4,7)
    r = requests.post(f"{BASE}/session/{sid}/action",
                      json={"player": 2, "type": "pawn", "target": [4, 7]})
    assert r.status_code == 200, f"Action failed: {r.text}"
    res = r.json()
    assert res["success"]
    assert res["state"]["pawns"]["2"] == {"x": 4, "y": 7}
    print("9. P2 moved to (4,7)")

    # 10. P1 places a wall
    r = requests.post(f"{BASE}/session/{sid}/action",
                      json={"player": 1, "type": "horizontal", "target": [3, 1]})
    assert r.status_code == 200, f"Wall failed: {r.text}"
    res = r.json()
    assert res["success"]
    assert res["state"]["walls_remaining"]["1"] == 9
    print("10. P1 placed horizontal wall at (3,1), walls remaining: 9")

    # 11. Test surrender
    r = requests.post(f"{BASE}/session/{sid}/surrender",
                      json={"seat": 2})
    assert r.status_code == 200
    res = r.json()
    assert res["status"] == "ended"
    assert res["result"]["winner_seat"] == 1
    assert res["result"]["termination"] == "surrender"
    print("11. P2 surrendered. Winner: seat 1")

    # 12. Verify session is ended
    r = requests.get(f"{BASE}/session/{sid}/state")
    state = r.json()
    assert state["status"] == "ended"
    assert state["result"]["winner_seat"] == 1
    print("12. Session ended correctly")

    # 13. New game
    r = requests.post(f"{BASE}/session/{sid}/new_game")
    assert r.status_code == 200
    assert r.json()["status"] == "waiting"
    print("13. New game — back to waiting")

    # 14. List sessions
    r = requests.get(f"{BASE}/session/list")
    assert r.status_code == 200
    sessions = r.json()["sessions"]
    assert any(s["session_id"] == sid for s in sessions)
    print(f"14. List sessions: {len(sessions)} session(s)")

    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    test_full_flow()
