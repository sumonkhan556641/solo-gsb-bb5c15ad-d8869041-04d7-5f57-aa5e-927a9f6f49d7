"""跨节点日志因果核验（事件真实时间 + 因果关系差分约束）测试。"""

import itertools

from app.solver import (
    CausalEvent,
    CausalRelation,
    Observation,
    verify_causal,
)


def _obs(id_, source, target, lo, hi):
    return Observation(id=id_, source=source, target=target, min=lo, max=hi)


def _ev(id_, node, ts):
    return CausalEvent(id=id_, node=node, timestamp=ts)


def _rel(id_, before, after, d):
    return CausalRelation(id=id_, before_id=before, after_id=after, min_interval=d)


def _check_witness(result, observations, events, relations):
    """独立核验可行结果的见证：参考偏移为 0、观测区间、关系间隔全部成立。"""
    assert result["feasible"] is True
    offsets = {item["node"]: item["offset"] for item in result["offsets"]}
    assert offsets[result["reference"]] == 0.0

    for o in observations:
        delta = offsets[o.target] - offsets[o.source]
        assert o.min - 1e-9 <= delta <= o.max + 1e-9

    times = {item["event_id"]: item["real_time"] for item in result["event_real_times"]}
    event_by_id = {e.id: e for e in events}
    # 真实时间 = 本地时间戳 − 节点偏移
    for e in events:
        assert times[e.id] == e.timestamp - offsets[e.node]
    for r in relations:
        gap = times[r.after_id] - times[r.before_id]
        assert gap >= r.min_interval - 1e-9
    return offsets, times


def test_causal_feasible_witness_satisfies_everything():
    observations = [_obs("o1", "A", "B", 1, 5), _obs("o2", "B", "C", -2, 2)]
    events = [
        _ev("e1", "A", 1000),
        _ev("e2", "B", 1003),
        _ev("e3", "C", 1004),
    ]
    relations = [
        _rel("r1", "e1", "e2", 0),
        _rel("r2", "e2", "e3", 0),
        _rel("r3", "e1", "e3", 1),
    ]
    result = verify_causal(
        nodes=["A", "B", "C"],
        reference="A",
        observations=observations,
        events=events,
        relations=relations,
    )
    offsets, times = _check_witness(result, observations, events, relations)
    # 节点与事件均按 ID 排序
    assert [o["node"] for o in result["offsets"]] == ["A", "B", "C"]
    assert [t["event_id"] for t in result["event_real_times"]] == ["e1", "e2", "e3"]
    assert times["e1"] == 1000.0  # 参考节点偏移为 0
    # event_real_times 回带节点
    assert {t["node"] for t in result["event_real_times"]} == {"A", "B", "C"}


def test_causal_event_only_nodes_are_covered_by_offsets():
    # 事件节点 D 不在任何观测中：节点宇宙为原诊断节点 ∪ 事件节点，
    # D 仍须出现在偏移见证里，其真实时间 = 时间戳 − 见证偏移。
    result = verify_causal(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 10)],
        events=[_ev("e1", "B", 5), _ev("e2", "D", 42)],
        relations=[],
    )
    offsets, times = _check_witness(
        result, [_obs("o1", "A", "B", 0, 10)],
        [_ev("e1", "B", 5), _ev("e2", "D", 42)], [],
    )
    assert set(offsets) == {"A", "B", "D"}
    assert times["e2"] == 42 - offsets["D"]


def test_causal_empty_input_only_reference():
    result = verify_causal(
        nodes=["A"], reference="A", observations=[], events=[], relations=[]
    )
    assert result["feasible"] is True
    assert result["offsets"] == [{"node": "A", "offset": 0.0}]
    assert result["event_real_times"] == []


def test_causal_relations_only_pin_same_node_clock():
    # 同一节点上的事件：真实时间差恒为本地时间戳差，间隔要求超过时间戳差即矛盾。
    result = verify_causal(
        nodes=["A"],
        reference="A",
        observations=[],
        events=[_ev("e1", "A", 0), _ev("e2", "A", 5)],
        relations=[_rel("r1", "e1", "e2", 6)],
    )
    assert result["feasible"] is False
    cycle = result["cycle"]
    assert cycle["accumulated_bound"] == -1.0
    assert cycle["contradiction"] == "0 <= -1.0"
    (step,) = cycle["steps"]
    assert step["type"] == "relation"
    assert step["from"] == "A" and step["to"] == "A"
    assert step["weight"] == -1.0 and step["min_interval"] == 6.0

    # 间隔 5 恰可满足
    ok = verify_causal(
        nodes=["A"], reference="A", observations=[],
        events=[_ev("e1", "A", 0), _ev("e2", "A", 5)],
        relations=[_rel("r1", "e1", "e2", 5)],
    )
    _check_witness(ok, [], [_ev("e1", "A", 0), _ev("e2", "A", 5)],
                   [_rel("r1", "e1", "e2", 5)])


def test_causal_self_relation_zero_interval_feasible_negative_infeasible():
    events = [_ev("e1", "A", 100)]
    ok = verify_causal(
        nodes=["A"], reference="A", observations=[],
        events=events, relations=[_rel("r1", "e1", "e1", 0)],
    )
    assert ok["feasible"] is True

    bad = verify_causal(
        nodes=["A"], reference="A", observations=[],
        events=events, relations=[_rel("r1", "e1", "e1", 1)],
    )
    assert bad["feasible"] is False
    (step,) = bad["cycle"]["steps"]
    assert step["type"] == "relation"
    assert step["before_event_id"] == "e1" and step["after_event_id"] == "e1"
    assert step["weight"] == -1.0


def test_causal_mixed_negative_cycle_closes_and_carries_both_step_types():
    # 观测：x_B - x_A ∈ [0, 10]。
    # 关系 e1(A,1000) -> e2(B,1005)，间隔 10：边 A->B 权 1005-1000-10 = -5。
    # 环：A->B(-5) + B->A(-min=0) = -5 < 0。
    result = verify_causal(
        nodes=["A", "B"],
        reference="A",
        observations=[_obs("o1", "A", "B", 0, 10)],
        events=[_ev("e1", "A", 1000), _ev("e2", "B", 1005)],
        relations=[_rel("r1", "e1", "e2", 10)],
    )
    assert result["feasible"] is False
    cycle = result["cycle"]
    assert cycle["accumulated_bound"] == -5.0
    types = {s["type"] for s in cycle["steps"]}
    assert types == {"observation", "relation"}

    rel_step = next(s for s in cycle["steps"] if s["type"] == "relation")
    assert rel_step["relation_id"] == "r1"
    assert rel_step["before_event_id"] == "e1"
    assert rel_step["after_event_id"] == "e2"
    assert (rel_step["from"], rel_step["to"]) == ("A", "B")
    assert rel_step["weight"] == -5.0 and rel_step["min_interval"] == 10.0

    obs_step = next(s for s in cycle["steps"] if s["type"] == "observation")
    assert obs_step["observation_id"] == "o1"
    assert obs_step["constraint"] == "lower"
    assert (obs_step["from"], obs_step["to"]) == ("B", "A")

    # 闭合：每步 from 等于上一步 to
    for i, step in enumerate(cycle["steps"]):
        assert step["from"] == cycle["steps"][i - 1]["to"]


def test_causal_result_independent_of_submission_order():
    observations = [_obs("o1", "A", "B", 0, 10), _obs("o2", "B", "C", 0, 10)]
    events = [_ev("e1", "A", 0), _ev("e2", "B", 105), _ev("e3", "C", 200)]
    relations = [
        _rel("r1", "e1", "e2", 100),
        _rel("r2", "e2", "e3", 100),
        _rel("r3", "e3", "e1", 0),
    ]
    results = []
    for ep in itertools.permutations(events):
        for rp in itertools.permutations(relations):
            results.append(
                verify_causal(
                    nodes=["A", "B", "C"], reference="A",
                    observations=observations, events=list(ep), relations=list(rp),
                )
            )
    assert results[0]["feasible"] is False
    assert all(r == results[0] for r in results[1:])


def test_causal_pure_observation_contradiction_still_detected_with_events():
    # 不带任何关系时，原有观测矛盾仍须检出（环上只有 observation 步骤）。
    result = verify_causal(
        nodes=["A", "B"],
        reference="A",
        observations=[
            _obs("o1", "A", "B", 0, 10),
            _obs("o2", "B", "A", 3, 4),
        ],
        events=[_ev("e1", "A", 0)],
        relations=[],
    )
    assert result["feasible"] is False
    assert {s["type"] for s in result["cycle"]["steps"]} == {"observation"}
    assert result["cycle"]["accumulated_bound"] == -3.0


def test_causal_chain_of_relations_consistent_with_offsets():
    # 两节点偏移 x_B ∈ [0,10]；事件链 e1(A,0) -> e2(B,8) -> e3(A,20)，
    # 各间隔 0：边 A->B 权 8、B->A 权 12，环权 8 + 12 = 20（非负，可行）；
    # 但把间隔提到 e1->e2 要求 9：边权 7，环仍非负；提到 21 才矛盾。
    def run(d1, d2):
        return verify_causal(
            nodes=["A", "B"], reference="A",
            observations=[_obs("o1", "A", "B", 0, 10)],
            events=[_ev("e1", "A", 0), _ev("e2", "B", 8), _ev("e3", "A", 20)],
            relations=[_rel("r1", "e1", "e2", d1), _rel("r2", "e2", "e3", d2)],
        )

    result = run(0, 0)
    _check_witness(
        result, [_obs("o1", "A", "B", 0, 10)],
        [_ev("e1", "A", 0), _ev("e2", "B", 8), _ev("e3", "A", 20)],
        [_rel("r1", "e1", "e2", 0), _rel("r2", "e2", "e3", 0)],
    )

    bad = run(21, 0)  # A->B 权 8-21 = -13；B->A 权 12；环 -1
    assert bad["feasible"] is False
    assert bad["cycle"]["accumulated_bound"] == -1.0
