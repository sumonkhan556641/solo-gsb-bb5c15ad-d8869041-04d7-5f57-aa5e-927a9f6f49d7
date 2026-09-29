"""跨节点日志因果一致性（静态观测 + 最小间隔关系）求解器测试。"""

from app.solver import (
    CausalEvent,
    CausalRelation,
    Observation,
    diagnose_causal,
)


def _obs(id_, source, target, lo, hi):
    return Observation(id=id_, source=source, target=target, min=lo, max=hi)


def _ev(id_, node, ts):
    return CausalEvent(id=id_, node=node, timestamp=ts)


def _rel(id_, before, after, gap):
    return CausalRelation(id=id_, before=before, after=after, min_interval=gap)


def test_feasible_returns_offsets_and_real_times():
    # 观测：x_B - x_A ∈ [0, 100]。
    # 关系 r1：e2(B@1060) - e1(A@1000) >= 10 -> x_B - x_A <= 50
    # 关系 r2：e3(B@1120) - e2(B@1060) >= 50 -> 0 <= 10（同节点恒成立）
    result = diagnose_causal(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 100)],
        events=[_ev("e1", "A", 1000), _ev("e2", "B", 1060), _ev("e3", "B", 1120)],
        relations=[_rel("r1", "e1", "e2", 10), _rel("r2", "e2", "e3", 50)],
    )
    assert result["feasible"] is True
    offsets = {o["node"]: o["offset"] for o in result["node_offsets"]}
    assert offsets["A"] == 0.0
    # 见证必须满足观测与全部关系
    assert 0 <= offsets["B"] - offsets["A"] <= 100
    real = {e["event_id"]: e["real_time"] for e in result["event_times"]}
    for eid, node, ts in (
        ("e1", "A", 1000.0),
        ("e2", "B", 1060.0),
        ("e3", "B", 1120.0),
    ):
        assert real[eid] == ts - offsets[node]
    assert real["e2"] - real["e1"] >= 10
    assert real["e3"] - real["e2"] >= 50
    # 节点按字典序、事件按 ID 排序
    assert [o["node"] for o in result["node_offsets"]] == ["A", "B"]
    assert [e["event_id"] for e in result["event_times"]] == ["e1", "e2", "e3"]


def test_event_nodes_unioned_into_nodes():
    # C 仅出现在事件中，仍必须出现在偏移见证中
    result = diagnose_causal(
        nodes=["A"],
        reference="A",
        observations=[],
        events=[_ev("e1", "C", 42)],
        relations=[],
    )
    assert result["feasible"] is True
    assert [o["node"] for o in result["node_offsets"]] == ["A", "C"]
    item = next(e for e in result["event_times"] if e["event_id"] == "e1")
    assert item["real_time"] == 42.0 - next(
        o["offset"] for o in result["node_offsets"] if o["node"] == "C"
    )


def test_infeasible_mixed_observation_relation_cycle():
    # 观测 x_B - x_A ∈ [0, 10]（边 B->A 权 0），
    # 关系要求 e2(B@1050) - e1(A@1000) >= 100（边 A->B 权 50-100=-50），
    # 环权 -50 + 0 = -50。
    result = diagnose_causal(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 10)],
        events=[_ev("e1", "A", 1000), _ev("e2", "B", 1050)],
        relations=[_rel("r1", "e1", "e2", 100)],
    )
    assert result["feasible"] is False
    cycle = result["cycle"]
    assert cycle["accumulated_bound"] == -50.0
    assert cycle["contradiction"] == "0 <= -50.0"
    by_type = {s["type"]: s for s in cycle["steps"]}
    assert set(by_type) == {"observation", "relation"}

    rel_step = by_type["relation"]
    assert rel_step["relation_id"] == "r1"
    assert rel_step["before_event_id"] == "e1"
    assert rel_step["after_event_id"] == "e2"
    assert rel_step["from"] == "A" and rel_step["to"] == "B"
    assert rel_step["weight"] == -50.0
    assert rel_step["min_interval"] == 100.0

    obs_step = by_type["observation"]
    assert obs_step["observation_id"] == "o1"
    assert obs_step["from"] == "B" and obs_step["to"] == "A"
    assert obs_step["constraint"] == "lower"
    assert obs_step["weight"] == 0.0

    # 首尾闭合
    steps = cycle["steps"]
    for i, step in enumerate(steps):
        assert step["from"] == steps[i - 1]["to"]


def test_infeasible_same_node_relation_is_self_cycle():
    # 同节点两事件时间戳差 20，却要求至少间隔 30：自环边权 20-30=-10
    result = diagnose_causal(
        nodes=["A"],
        reference="A",
        observations=[],
        events=[_ev("e1", "A", 1000), _ev("e2", "A", 1020)],
        relations=[_rel("r1", "e1", "e2", 30)],
    )
    assert result["feasible"] is False
    steps = result["cycle"]["steps"]
    assert len(steps) == 1
    assert steps[0]["type"] == "relation"
    assert steps[0]["from"] == "A" and steps[0]["to"] == "A"
    assert steps[0]["weight"] == -10.0
    assert result["cycle"]["accumulated_bound"] == -10.0


def test_pure_observation_contradiction_still_detected():
    result = diagnose_causal(
        nodes=["A", "B"],
        reference="A",
        observations=[
            _obs("o1", "A", "B", 0, 10),
            _obs("o2", "B", "A", 3, 4),
        ],
        events=[],
        relations=[],
    )
    assert result["feasible"] is False
    assert {s["type"] for s in result["cycle"]["steps"]} == {"observation"}
    assert result["cycle"]["accumulated_bound"] == -3.0


def test_cycle_independent_of_submission_order():
    observations = [_obs("o1", "A", "B", 0, 10)]
    events = [_ev("e1", "A", 1000), _ev("e2", "B", 1050)]
    relations = [_rel("r1", "e1", "e2", 100)]
    orders = [
        (observations, events, relations),
        (list(reversed(observations)), list(reversed(events)), list(reversed(relations))),
        (
            list(reversed(observations)),
            [events[1], events[0]],
            list(reversed(relations)),
        ),
    ]
    cycles = [
        diagnose_causal(
            nodes=["A", "B"],
            reference="A",
            observations=o,
            events=e,
            relations=r,
        )["cycle"]
        for o, e, r in orders
    ]
    assert cycles[0] == cycles[1] == cycles[2]


def test_zero_interval_relation_feasible_at_exact_gap():
    # 同节点时间戳差恰为 0、间隔要求 0：零权自环不是负环，可行
    result = diagnose_causal(
        nodes=["A"],
        reference="A",
        observations=[],
        events=[_ev("e1", "A", 1000), _ev("e2", "A", 1000)],
        relations=[_rel("r1", "e1", "e2", 0)],
    )
    assert result["feasible"] is True


def test_empty_inputs_feasible_with_reference_witness():
    result = diagnose_causal(
        nodes=["A", "B"],
        reference="A",
        observations=[],
        events=[],
        relations=[],
    )
    assert result["feasible"] is True
    offsets = {o["node"]: o["offset"] for o in result["node_offsets"]}
    assert offsets == {"A": 0.0, "B": 0.0}
    assert result["event_times"] == []
