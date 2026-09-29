"""跨节点日志因果一致性接口（POST /api/v1/events/causal-consistency）测试。"""

import json

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PATH = "/api/v1/events/causal-consistency"


def _feasible_payload(**overrides):
    payload = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 100}
        ],
        "events": [
            {"id": "e1", "node": "A", "timestamp": 1000},
            {"id": "e2", "node": "B", "timestamp": 1060},
            {"id": "e3", "node": "B", "timestamp": 1120},
        ],
        "relations": [
            {"id": "r1", "before": "e1", "after": "e2", "min_interval": 10},
            {"id": "r2", "before": "e2", "after": "e3", "min_interval": 50},
        ],
    }
    payload.update(overrides)
    return payload


def test_feasible_returns_offset_witness_and_real_times():
    resp = client.post(PATH, json=_feasible_payload())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["feasible"] is True
    assert data["reference"] == "A"
    assert [o["node"] for o in data["node_offsets"]] == ["A", "B"]
    offsets = {o["node"]: o["offset"] for o in data["node_offsets"]}
    assert offsets["A"] == 0.0
    assert 0 <= offsets["B"] <= 100

    assert [e["event_id"] for e in data["event_times"]] == ["e1", "e2", "e3"]
    real = {e["event_id"]: e["real_time"] for e in data["event_times"]}
    # 真实时间 = 本地时间戳 − 偏移；间隔关系均被见证满足
    assert real["e1"] == 1000.0 - offsets["A"]
    assert real["e2"] == 1060.0 - offsets["B"]
    assert real["e3"] == 1120.0 - offsets["B"]
    assert real["e2"] - real["e1"] >= 10
    assert real["e3"] - real["e2"] >= 50


def test_name_echoed():
    data = client.post(PATH, json=_feasible_payload(name="trace-9")).json()
    assert data["name"] == "trace-9"


def test_node_union_covers_event_only_node():
    # C 只出现在事件节点，须并入节点集合并给出偏移见证
    payload = _feasible_payload(
        observations=[],
        events=[{"id": "e1", "node": "C", "timestamp": 7}],
        relations=[],
    )
    data = client.post(PATH, json=payload).json()
    assert data["feasible"] is True
    assert [o["node"] for o in data["node_offsets"]] == ["A", "C"]


def test_empty_events_and_relations_accepted():
    payload = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 1, "max": 5}
        ],
        "events": [],
        "relations": [],
    }
    data = client.post(PATH, json=payload).json()
    assert data["feasible"] is True
    assert data["event_times"] == []
    offsets = {o["node"]: o["offset"] for o in data["node_offsets"]}
    assert offsets["A"] == 0.0 and 1 <= offsets["B"] <= 5


def test_infeasible_mixed_cycle_step_fields_and_closure():
    payload = {
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10}
        ],
        "events": [
            {"id": "e1", "node": "A", "timestamp": 1000},
            {"id": "e2", "node": "B", "timestamp": 1050},
        ],
        "relations": [
            {"id": "r1", "before": "e1", "after": "e2", "min_interval": 100}
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["feasible"] is False
    assert "node_offsets" not in data and "event_times" not in data
    cycle = data["cycle"]
    assert cycle["accumulated_bound"] == -50.0
    assert cycle["contradiction"] == "0 <= -50.0"
    assert {s["type"] for s in cycle["steps"]} == {"observation", "relation"}

    rel_step = next(s for s in cycle["steps"] if s["type"] == "relation")
    assert set(rel_step) == {
        "type", "relation_id", "before_event_id", "after_event_id",
        "from", "to", "weight", "min_interval",
    }
    assert rel_step["relation_id"] == "r1"
    assert rel_step["before_event_id"] == "e1"
    assert rel_step["after_event_id"] == "e2"
    assert rel_step["from"] == "A" and rel_step["to"] == "B"
    assert rel_step["weight"] == -50.0 and rel_step["min_interval"] == 100.0

    obs_step = next(s for s in cycle["steps"] if s["type"] == "observation")
    assert set(obs_step) == {
        "type", "observation_id", "from", "to", "constraint", "weight", "bound"
    }

    # 每步首尾闭合，累计权重为负
    steps = cycle["steps"]
    for i, step in enumerate(steps):
        assert step["from"] == steps[i - 1]["to"]
    assert sum(s["weight"] for s in steps) == -50.0


def test_infeasible_same_node_relation_self_cycle():
    payload = {
        "reference": "A",
        "observations": [],
        "events": [
            {"id": "e1", "node": "A", "timestamp": 1000},
            {"id": "e2", "node": "A", "timestamp": 1020},
        ],
        "relations": [
            {"id": "r1", "before": "e1", "after": "e2", "min_interval": 30}
        ],
    }
    data = client.post(PATH, json=payload).json()
    assert data["feasible"] is False
    steps = data["cycle"]["steps"]
    assert len(steps) == 1 and steps[0]["type"] == "relation"
    assert steps[0]["from"] == "A" and steps[0]["to"] == "A"
    assert data["cycle"]["accumulated_bound"] == -10.0


def test_cycle_independent_of_submission_order():
    payload = _feasible_payload(
        observations=[
            {"id": "o1", "source": "A", "target": "B", "min": 0, "max": 10}
        ],
        events=[
            {"id": "e1", "node": "A", "timestamp": 1000},
            {"id": "e2", "node": "B", "timestamp": 1050},
        ],
        relations=[
            {"id": "r1", "before": "e1", "after": "e2", "min_interval": 100}
        ],
    )
    r1 = client.post(PATH, json=payload).json()
    shuffled = {
        "reference": "A",
        "observations": list(reversed(payload["observations"])),
        "events": list(reversed(payload["events"])),
        "relations": list(reversed(payload["relations"])),
    }
    r2 = client.post(PATH, json=shuffled).json()
    assert r1 == r2


def _expect_422(payload, fragment=None):
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 422, resp.text
    if fragment is not None:
        assert fragment in resp.text, (fragment, resp.text)
    return resp


def test_422_more_than_twenty_events():
    _expect_422(
        {
            "reference": "A", "observations": [],
            "events": [
                {"id": f"e{i:02d}", "node": "A", "timestamp": i}
                for i in range(21)
            ],
            "relations": [],
        },
        "20",
    )


def test_422_exactly_twenty_events_accepted():
    payload = {
        "reference": "A", "observations": [],
        "events": [
            {"id": f"e{i:02d}", "node": "A", "timestamp": i} for i in range(20)
        ],
        "relations": [],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["event_times"]) == 20


def test_422_more_than_forty_relations():
    _expect_422(
        {
            "reference": "A", "observations": [],
            "events": [{"id": "e0", "node": "A", "timestamp": 0}],
            "relations": [
                {"id": f"r{i:02d}", "before": "e0", "after": "e0", "min_interval": 0}
                for i in range(41)
            ],
        }
    )


def test_422_duplicate_event_id():
    _expect_422(
        {
            "reference": "A", "observations": [],
            "events": [
                {"id": "x", "node": "A", "timestamp": 1},
                {"id": "x", "node": "B", "timestamp": 2},
            ],
            "relations": [],
        },
        "重复",
    )


def test_422_duplicate_relation_id():
    _expect_422(
        {
            "reference": "A", "observations": [],
            "events": [{"id": "e1", "node": "A", "timestamp": 1}],
            "relations": [
                {"id": "x", "before": "e1", "after": "e1", "min_interval": 0},
                {"id": "x", "before": "e1", "after": "e1", "min_interval": 1},
            ],
        },
        "重复",
    )


def test_422_dangling_event_reference():
    _expect_422(
        {
            "reference": "A", "observations": [],
            "events": [{"id": "e1", "node": "A", "timestamp": 1}],
            "relations": [
                {"id": "r1", "before": "e1", "after": "ghost", "min_interval": 0}
            ],
        },
        "事件",
    )
    _expect_422(
        {
            "reference": "A", "observations": [],
            "events": [{"id": "e1", "node": "A", "timestamp": 1}],
            "relations": [
                {"id": "r1", "before": "ghost", "after": "e1", "min_interval": 0}
            ],
        }
    )


def test_422_negative_min_interval():
    _expect_422(
        {
            "reference": "A", "observations": [],
            "events": [{"id": "e1", "node": "A", "timestamp": 1}],
            "relations": [
                {"id": "r1", "before": "e1", "after": "e1", "min_interval": -1}
            ],
        }
    )


def test_422_non_finite_values_do_not_500():
    body = (
        '{"reference":"A","observations":[],'
        '"events":[{"id":"e1","node":"A","timestamp":NaN}],"relations":[]}'
    )
    resp = client.post(PATH, content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422

    body = (
        '{"reference":"A","observations":[],"events":[{"id":"e1","node":"A","timestamp":1}],'
        '"relations":[{"id":"r1","before":"e1","after":"e1","min_interval":Infinity}]}'
    )
    resp = client.post(PATH, content=body, headers={"content-type": "application/json"})
    assert resp.status_code == 422


def test_ids_unique_only_within_category():
    # 事件 ID 与观测 ID、关系 ID 允许重名
    payload = {
        "reference": "A",
        "observations": [
            {"id": "same", "source": "A", "target": "B", "min": 0, "max": 10}
        ],
        "events": [{"id": "same", "node": "A", "timestamp": 1}],
        "relations": [
            {"id": "same", "before": "same", "after": "same", "min_interval": 0}
        ],
    }
    resp = client.post(PATH, json=payload)
    assert resp.status_code == 200, resp.text


def test_inherited_observation_validation_still_applies():
    # 静态观测的既有校验（非法区间、重复观测 ID、reference 不在 nodes）仍生效
    _expect_422({
        "reference": "A",
        "observations": [
            {"id": "o1", "source": "A", "target": "B", "min": 9, "max": 1}
        ],
        "events": [], "relations": [],
    })
    _expect_422({
        "reference": "A",
        "observations": [
            {"id": "x", "source": "A", "target": "B", "min": 0, "max": 1},
            {"id": "x", "source": "B", "target": "A", "min": 0, "max": 1},
        ],
        "events": [], "relations": [],
    })
    _expect_422({
        "reference": "Z", "nodes": ["A", "B"],
        "observations": [], "events": [], "relations": [],
    })


def test_422_unknown_field():
    _expect_422({
        "reference": "A", "observations": [],
        "events": [], "relations": [], "bogus": 1,
    })


def test_health_unchanged():
    assert client.get("/health").json() == {"status": "ok"}
