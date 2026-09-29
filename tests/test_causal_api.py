"""跨节点日志因果核验接口（POST /api/v1/events/causal-verification）测试。"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PATH = "/api/v1/events/causal-verification"


def test_causal_api_feasible_returns_offsets_witness_and_event_real_times():
    payload = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 1, "max": 5}
        ],
        "events": [
            {"id": "e2", "node": "B", "timestamp": 1004},
            {"id": "e1", "node": "A", "timestamp": 1000},  # 乱序提交
        ],
        "relations": [
            {"id": "r1", "before_event_id": "e1", "after_event_id": "e2",
             "min_interval": 0}
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["feasible"] is True
    assert data["reference"] == "A"

    offsets = {o["node"]: o["offset"] for o in data["offsets"]}
    assert [o["node"] for o in data["offsets"]] == ["A", "B"]  # 按节点排序
    assert offsets["A"] == 0.0  # 参考偏移恒为 0

    # 见证满足观测：1 <= x_B - x_A <= 5
    assert 1.0 - 1e-9 <= offsets["B"] <= 5.0 + 1e-9

    times = {t["event_id"]: t["real_time"] for t in data["event_real_times"]}
    assert [t["event_id"] for t in data["event_real_times"]] == ["e1", "e2"]
    assert times["e1"] == 1000.0
    # 真实时间 = 本地时间戳 − 偏移；且关系 r1：real(e2) - real(e1) >= 0
    assert times["e2"] == 1004.0 - offsets["B"]
    assert times["e2"] - times["e1"] >= -1e-9


def test_causal_api_name_echoed():
    data = client.post(
        PATH, json={"reference": "A", "name": "trace-7"}
    ).json()
    assert data["feasible"] is True and data["name"] == "trace-7"
    assert data["offsets"] == [{"node": "A", "offset": 0.0}]
    assert data["event_real_times"] == []


def test_causal_api_infeasible_mixed_cycle_fields_and_closure():
    payload = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10}
        ],
        "events": [
            {"id": "e1", "node": "A", "timestamp": 1000},
            {"id": "e2", "node": "B", "timestamp": 1005},
        ],
        "relations": [
            {"id": "r1", "before_event_id": "e1", "after_event_id": "e2",
             "min_interval": 10}
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["feasible"] is False
    assert "offsets" not in data and "event_real_times" not in data
    cycle = data["cycle"]
    assert cycle["accumulated_bound"] == -5.0
    assert cycle["contradiction"] == "0 <= -5.0"
    assert {s["type"] for s in cycle["steps"]} == {"observation", "relation"}

    rel = next(s for s in cycle["steps"] if s["type"] == "relation")
    assert set(rel) == {
        "type", "relation_id", "before_event_id", "after_event_id",
        "from", "to", "weight", "min_interval",
    }
    assert rel == {
        "type": "relation",
        "relation_id": "r1",
        "before_event_id": "e1",
        "after_event_id": "e2",
        "from": "A",
        "to": "B",
        "weight": -5.0,
        "min_interval": 10.0,
    }

    obs = next(s for s in cycle["steps"] if s["type"] == "observation")
    assert set(obs) == {
        "type", "observation_id", "from", "to", "constraint", "weight", "bound"
    }

    # 环闭合：每步 from 等于上一步 to
    for i, step in enumerate(cycle["steps"]):
        assert step["from"] == cycle["steps"][i - 1]["to"]


def test_causal_api_relation_only_cycle_on_same_node():
    payload = {
        "reference": "A",
        "events": [
            {"id": "e1", "node": "A", "timestamp": 0},
            {"id": "e2", "node": "A", "timestamp": 5},
        ],
        "relations": [
            {"id": "r1", "before_event_id": "e1", "after_event_id": "e2",
             "min_interval": 6}
        ],
    }
    data = client.post(PATH, json=payload).json()
    assert data["feasible"] is False
    (step,) = data["cycle"]["steps"]
    assert step["type"] == "relation" and step["weight"] == -1.0
    assert data["cycle"]["accumulated_bound"] == -1.0


def test_causal_api_event_nodes_join_node_universe():
    payload = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10}
        ],
        "events": [
            {"id": "e1", "node": "B", "timestamp": 5},
            {"id": "e2", "node": "D", "timestamp": 42},
        ],
    }
    data = client.post(PATH, json=payload).json()
    assert data["feasible"] is True
    nodes = [o["node"] for o in data["offsets"]]
    assert nodes == ["A", "B", "D"]
    times = {t["event_id"]: t for t in data["event_real_times"]}
    offsets = {o["node"]: o["offset"] for o in data["offsets"]}
    assert times["e2"]["node"] == "D"
    assert times["e2"]["real_time"] == 42.0 - offsets["D"]


def _expect_422(payload, fragment=None):
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422, resp.text
    if fragment is not None:
        assert fragment in resp.text, (fragment, resp.text)
    return resp


def test_causal_api_422_duplicate_event_id():
    _expect_422(
        {"reference": "A",
         "events": [
             {"id": "e", "node": "A", "timestamp": 1},
             {"id": "e", "node": "A", "timestamp": 2},
         ]},
        "重复",
    )


def test_causal_api_422_duplicate_relation_id():
    events = [
        {"id": "e1", "node": "A", "timestamp": 1},
        {"id": "e2", "node": "B", "timestamp": 2},
    ]
    _expect_422(
        {"reference": "A", "events": events,
         "relations": [
             {"id": "r", "before_event_id": "e1", "after_event_id": "e2",
              "min_interval": 0},
             {"id": "r", "before_event_id": "e1", "after_event_id": "e2",
              "min_interval": 0},
         ]},
        "重复",
    )


def test_causal_api_422_dangling_event_reference():
    _expect_422(
        {"reference": "A",
         "events": [{"id": "e1", "node": "A", "timestamp": 1}],
         "relations": [
             {"id": "r1", "before_event_id": "ghost", "after_event_id": "e1",
              "min_interval": 0}
         ]},
        "不存在的事件",
    )


def test_causal_api_422_negative_min_interval():
    events = [
        {"id": "e1", "node": "A", "timestamp": 1},
        {"id": "e2", "node": "A", "timestamp": 2},
    ]
    _expect_422(
        {"reference": "A", "events": events,
         "relations": [
             {"id": "r1", "before_event_id": "e1", "after_event_id": "e2",
              "min_interval": -0.5}
         ]},
    )


def test_causal_api_422_non_finite_values_do_not_500():
    body = (
        '{"reference":"A","events":[{"id":"e1","node":"A","timestamp":NaN}]}'
    )
    resp = client.post(
        PATH, content=body, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422

    body = (
        '{"reference":"A",'
        '"events":[{"id":"e1","node":"A","timestamp":1},'
        '{"id":"e2","node":"A","timestamp":2}],'
        '"relations":[{"id":"r1","before_event_id":"e1",'
        '"after_event_id":"e2","min_interval":Infinity}]}'
    )
    resp = client.post(
        PATH, content=body, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422


def test_causal_api_422_more_than_twenty_events():
    _expect_422(
        {"reference": "A",
         "events": [
             {"id": f"e{i}", "node": "A", "timestamp": i} for i in range(21)
         ]},
        "20",
    )


def test_causal_api_422_more_than_forty_relations():
    events = [
        {"id": "a", "node": "A", "timestamp": 1},
        {"id": "b", "node": "A", "timestamp": 2},
    ]
    _expect_422(
        {"reference": "A", "events": events,
         "relations": [
             {"id": f"r{i}", "before_event_id": "a", "after_event_id": "b",
              "min_interval": 0}
             for i in range(41)
         ]},
        "40",
    )


def test_causal_api_422_limits_inclusive_at_boundary():
    ok = client.post(PATH, json={
        "reference": "A",
        "events": [
            {"id": f"e{i}", "node": "A", "timestamp": i} for i in range(20)
        ],
        "relations": [
            {"id": f"r{i}",
             "before_event_id": "e0", "after_event_id": "e1", "min_interval": 0}
            for i in range(40)
        ],
    })
    assert ok.status_code == 200, ok.text


def test_causal_api_422_unknown_field():
    _expect_422({"reference": "A", "bogus": 1})


def test_causal_api_422_duplicate_observation_id_still_enforced():
    _expect_422(
        {"reference": "A",
         "observations": [
             {"id": "o", "source": "A", "target": "B", "min": 0, "max": 1},
             {"id": "o", "source": "A", "target": "B", "min": 0, "max": 1},
         ]},
    )


def test_causal_api_ids_only_unique_within_category():
    # 观测、事件、关系三类 ID 可以重名，互不冲突。
    payload = {
        "reference": "A",
        "observations": [
            {"id": "x", "source": "A", "target": "B", "min": 0, "max": 10}
        ],
        "events": [
            {"id": "x", "node": "A", "timestamp": 1},
            {"id": "y", "node": "B", "timestamp": 5},
        ],
        "relations": [
            {"id": "x", "before_event_id": "x", "after_event_id": "y",
             "min_interval": 0}
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    assert resp.json()["feasible"] is True


def test_causal_api_cycle_independent_of_submission_order():
    payload = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10},
            {"id": "o2", "source": "B", "target": "C", "min": 0, "max": 10},
        ],
        "events": [
            {"id": "e1", "node": "A", "timestamp": 0},
            {"id": "e2", "node": "B", "timestamp": 105},
            {"id": "e3", "node": "C", "timestamp": 200},
        ],
        "relations": [
            {"id": "r1", "before_event_id": "e1", "after_event_id": "e2",
             "min_interval": 100},
            {"id": "r2", "before_event_id": "e2", "after_event_id": "e3",
             "min_interval": 100},
            {"id": "r3", "before_event_id": "e3", "after_event_id": "e1",
             "min_interval": 0},
        ],
    }
    seen = set()
    for events_order in [
        ["e1", "e2", "e3"], ["e3", "e1", "e2"], ["e2", "e3", "e1"],
    ]:
        shuffled = dict(payload)
        shuffled["events"] = [
            next(e for e in payload["events"] if e["id"] == eid)
            for eid in events_order
        ]
        shuffled["relations"] = list(reversed(payload["relations"]))
        data = client.post(PATH, json=shuffled).json()
        assert data["feasible"] is False
        seen.add(
            tuple((s["type"], s.get("relation_id") or s.get("observation_id"))
                  for s in data["cycle"]["steps"])
        )
    assert len(seen) == 1
