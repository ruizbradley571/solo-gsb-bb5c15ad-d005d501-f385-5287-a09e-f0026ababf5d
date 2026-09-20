"""跨展柜共模突变事件：全局选择算法单测 + 接口集成测试（内存 SQLite）。"""

from __future__ import annotations

import datetime as dt
import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import Base, engine, init_db
from app.incidents import Boundary, compute_incidents
from app.main import app

BASE = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


def _b(sn, minute, mag, delta_t=4.0, delta_h=0.0) -> Boundary:
    return Boundary(
        device_sn=sn,
        ts=BASE + dt.timedelta(minutes=minute),
        magnitude=mag,
        delta_temperature=delta_t,
        delta_humidity=delta_h,
    )


def _reading(sn, ts, t, h):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _iso(minute: float) -> str:
    return (BASE + dt.timedelta(minutes=minute)).isoformat().replace("+00:00", "Z")


def _step_rows(sn, boundary_minute, jump_t, jump_h=0.0, n=6):
    """2n 条读数的阶跃序列：前 n 条 20°C/55%RH，后 n 条叠加跳变。

    分段边界（第二段段首）恰好落在 ``boundary_minute`` 分钟处。
    """
    rows = []
    for i in range(-n, 0):
        rows.append(_reading(sn, _iso(boundary_minute + i), 20.0, 55.0))
    for i in range(0, n):
        rows.append(
            _reading(sn, _iso(boundary_minute + i), 20.0 + jump_t, 55.0 + jump_h)
        )
    return rows


def _ingest(client, rows, **extra):
    payload = {"readings": rows}
    payload.update(extra)
    r = client.post("/api/v1/readings/batch", json=payload)
    assert r.status_code == 200, r.text


# ---------- 纯算法：候选枚举与全局选择 ----------


def test_basic_window_forms_single_event():
    events = compute_incidents(
        [_b("A", 0, 2.0), _b("B", 1, 2.0)],
        tolerance_seconds=120.0,
        min_devices=2,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.device_count == 2
    assert ev.total_magnitude == pytest.approx(4.0)
    assert ev.start_ts == BASE
    assert ev.end_ts == BASE + dt.timedelta(minutes=1)
    assert [b.device_sn for b in ev.boundaries] == ["A", "B"]


def test_span_equal_to_tolerance_is_allowed():
    events = compute_incidents(
        [_b("A", 0, 2.0), _b("B", 2, 2.0)],
        tolerance_seconds=120.0,
        min_devices=2,
    )
    assert len(events) == 1
    # 超出容差 1 秒即不得成候选
    none = compute_incidents(
        [_b("A", 0, 2.0), _b("B", 2, 2.0)],
        tolerance_seconds=119.0,
        min_devices=2,
    )
    assert none == []


def test_single_device_boundaries_never_form_event():
    events = compute_incidents(
        [_b("A", 0, 2.0), _b("A", 1, 3.0)],
        tolerance_seconds=120.0,
        min_devices=2,
    )
    assert events == []


def test_same_device_keeps_max_magnitude_boundary():
    # A 在窗内有两个边界：保留幅度更大的 A@1（3.0）
    events = compute_incidents(
        [_b("A", 0, 1.0), _b("A", 1, 3.0), _b("B", 1, 2.0)],
        tolerance_seconds=120.0,
        min_devices=2,
    )
    assert len(events) == 1
    ev = events[0]
    assert ev.device_count == 2
    assert ev.total_magnitude == pytest.approx(5.0)
    kept = {b.device_sn: b for b in ev.boundaries}
    assert kept["A"].ts == BASE + dt.timedelta(minutes=1)
    assert kept["A"].magnitude == pytest.approx(3.0)


def test_same_device_magnitude_tie_keeps_earlier():
    # A 的两个边界幅度并列：保留更早的 A@0
    events = compute_incidents(
        [_b("A", 0, 2.0), _b("A", 1, 2.0), _b("B", 1, 2.0)],
        tolerance_seconds=120.0,
        min_devices=2,
    )
    assert len(events) == 1
    kept = {b.device_sn: b for b in events[0].boundaries}
    assert kept["A"].ts == BASE


def test_min_devices_filters_candidates():
    boundaries = [_b("A", 0, 2.0), _b("B", 1, 2.0)]
    assert compute_incidents(boundaries, 120.0, 3) == []
    assert len(compute_incidents(boundaries, 120.0, 2)) == 1


def test_global_selection_avoids_greedy_chain_merge():
    # 4 个边界等距 1 分钟，容差 2 分钟：链式归并会串成 [0,3] 的非法大窗，
    # 全局选择应给出互不重叠的 [0,1] 与 [2,3] 两个事件（设备数总和 4 最大）
    boundaries = [_b(sn, m, 2.0) for sn, m in zip("ABCD", (0, 1, 2, 3))]
    events = compute_incidents(boundaries, tolerance_seconds=120.0, min_devices=2)
    assert len(events) == 2
    assert events[0].start_ts == BASE
    assert events[0].end_ts == BASE + dt.timedelta(minutes=1)
    assert events[1].start_ts == BASE + dt.timedelta(minutes=2)
    assert events[1].end_ts == BASE + dt.timedelta(minutes=3)
    assert sum(e.device_count for e in events) == 4


def test_device_total_is_primary_criterion():
    # 互不重叠的三组全部入选（设备数总和随事件数累加）
    boundaries = [
        _b("A", 0, 5.0),
        _b("B", 0, 5.0),
        _b("C", 10, 1.0),
        _b("D", 10, 1.0),
        _b("E", 20, 2.0),
        _b("F", 20, 2.0),
    ]
    events = compute_incidents(boundaries, tolerance_seconds=60.0, min_devices=2)
    assert len(events) == 3
    # 同一容差窗内：高幅度的 2 设备子窗（mag 10）输给 4 设备整窗（mag 12），
    # 尽管单位设备平均幅度更低——设备数总和是第一判据
    boundaries = [
        _b("A", 0, 5.0),
        _b("B", 0, 5.0),
        _b("C", 0, 1.0),
        _b("D", 0, 1.0),
    ]
    events = compute_incidents(boundaries, tolerance_seconds=60.0, min_devices=2)
    assert len(events) == 1
    assert events[0].device_count == 4
    assert events[0].total_magnitude == pytest.approx(12.0)


def test_start_sequence_tiebreak_prefers_earlier():
    # [A,B] 与 [B,C] 重叠且设备数、幅度总和相同：起始更早的 [A,B] 胜出
    boundaries = [_b("A", 0, 2.0), _b("B", 5, 2.0), _b("C", 7, 2.0)]
    events = compute_incidents(boundaries, tolerance_seconds=300.0, min_devices=2)
    assert len(events) == 1
    assert events[0].start_ts == BASE
    assert [b.device_sn for b in events[0].boundaries] == ["A", "B"]


def test_direction_from_scaled_signed_sum():
    assert _b("A", 0, 1.0, delta_t=4.0, delta_h=0.0).direction == "up"
    assert _b("A", 0, 1.0, delta_t=-4.0, delta_h=0.0).direction == "down"
    assert _b("A", 0, 1.0, delta_t=0.0, delta_h=10.0).direction == "up"
    assert _b("A", 0, 1.0, delta_t=0.0, delta_h=-10.0).direction == "down"
    # 温度 +4（+2σ）与湿度 -10（-2σ）恰好抵消：零视为 up
    assert _b("A", 0, 1.0, delta_t=4.0, delta_h=-10.0).direction == "up"


def test_unsorted_input_deterministic():
    boundaries = [_b(sn, m, 2.0) for sn, m in zip("ABCD", (0, 1, 2, 3))]
    ordered = compute_incidents(boundaries, 120.0, 2)
    shuffled = compute_incidents(boundaries[::-1], 120.0, 2)
    assert [(e.start_ts, e.end_ts, e.device_count) for e in ordered] == [
        (e.start_ts, e.end_ts, e.device_count) for e in shuffled
    ]


# ---------- 接口集成 ----------


def test_incidents_basic_cross_case_event(client):
    _ingest(client, _step_rows("CASE-I1", 6, jump_t=4.0))
    _ingest(client, _step_rows("CASE-I2", 7, jump_t=4.0))

    r = client.get("/api/v1/incidents", params={"tolerance_seconds": 120})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["tolerance_seconds"] == 120.0
    assert body["min_devices"] == 2
    assert body["boundary_count"] == 2
    assert body["event_count"] == 1

    ev = body["events"][0]
    assert ev["device_count"] == 2
    assert ev["total_magnitude"] == pytest.approx(4.0)
    assert ev["start_ts"] == _iso(6)
    assert ev["end_ts"] == _iso(7)
    assert [(b["device_sn"], b["direction"]) for b in ev["boundaries"]] == [
        ("CASE-I1", "up"),
        ("CASE-I2", "up"),
    ]
    for b in ev["boundaries"]:
        assert b["magnitude"] == pytest.approx(2.0)
        assert b["delta_temperature"] == pytest.approx(4.0)
        assert b["delta_humidity"] == pytest.approx(0.0)


def test_incidents_default_params_echo(client):
    _ingest(client, _step_rows("CASE-I3", 6, jump_t=4.0))
    _ingest(client, _step_rows("CASE-I4", 7, jump_t=4.0))
    body = client.get("/api/v1/incidents").json()
    assert body["tolerance_seconds"] == 300.0
    assert body["min_devices"] == 2
    assert body["start"] is None and body["end"] is None
    assert body["event_count"] == 1


def test_incidents_tolerance_controls_grouping(client):
    _ingest(client, _step_rows("CASE-I5", 6, jump_t=4.0))
    _ingest(client, _step_rows("CASE-I6", 10, jump_t=4.0))  # 相差 4 分钟
    params = {"tolerance_seconds": 120}
    assert client.get("/api/v1/incidents", params=params).json()["event_count"] == 0
    params = {"tolerance_seconds": 240}
    assert client.get("/api/v1/incidents", params=params).json()["event_count"] == 1


def test_incidents_min_devices_param(client):
    _ingest(client, _step_rows("CASE-I7", 6, jump_t=4.0))
    _ingest(client, _step_rows("CASE-I8", 7, jump_t=4.0))
    r = client.get("/api/v1/incidents", params={"min_devices": 3})
    assert r.status_code == 200
    assert r.json()["event_count"] == 0

    _ingest(client, _step_rows("CASE-I9", 8, jump_t=4.0))
    body = client.get("/api/v1/incidents", params={"min_devices": 3}).json()
    assert body["event_count"] == 1
    assert body["events"][0]["device_count"] == 3
    assert body["events"][0]["total_magnitude"] == pytest.approx(6.0)


def test_incidents_mixed_directions(client):
    _ingest(client, _step_rows("CASE-IU", 6, jump_t=4.0))
    _ingest(client, _step_rows("CASE-IV", 7, jump_t=-4.0))
    body = client.get("/api/v1/incidents", params={"tolerance_seconds": 120}).json()
    assert body["event_count"] == 1
    directions = {b["device_sn"]: b["direction"] for b in body["events"][0]["boundaries"]}
    assert directions == {"CASE-IU": "up", "CASE-IV": "down"}
    deltas = {
        b["device_sn"]: b["delta_temperature"] for b in body["events"][0]["boundaries"]
    }
    assert deltas["CASE-IU"] == pytest.approx(4.0)
    assert deltas["CASE-IV"] == pytest.approx(-4.0)


def test_incidents_no_greedy_chain_merge_via_api(client):
    # 4 个展柜边界间隔 1 分钟、容差 2 分钟：不得链式归并成一个大事件
    for k, sn in enumerate(["CASE-G0", "CASE-G1", "CASE-G2", "CASE-G3"]):
        _ingest(client, _step_rows(sn, 6 + k, jump_t=4.0))
    body = client.get("/api/v1/incidents", params={"tolerance_seconds": 120}).json()
    assert body["event_count"] == 2
    first, second = body["events"]
    assert first["start_ts"] == _iso(6) and first["end_ts"] == _iso(7)
    assert second["start_ts"] == _iso(8) and second["end_ts"] == _iso(9)
    assert first["device_count"] == second["device_count"] == 2
    # 事件按开始时间排序
    starts = [e["start_ts"] for e in body["events"]]
    assert starts == sorted(starts)


def test_incidents_start_end_filter_scopes_boundaries(client):
    _ingest(client, _step_rows("CASE-IF1", 6, jump_t=4.0))
    _ingest(client, _step_rows("CASE-IF2", 7, jump_t=4.0))

    # 窗口只含 CASE-IF2 的边界：distinct device 不足，无事件
    body = client.get(
        "/api/v1/incidents", params={"start": _iso(7)}
    ).json()
    assert body["boundary_count"] == 1
    assert body["event_count"] == 0
    assert body["start"].startswith("2026-09-01T00:07:00")

    # 窗口只含 CASE-IF1 的边界
    body = client.get("/api/v1/incidents", params={"end": _iso(6)}).json()
    assert body["boundary_count"] == 1
    assert body["event_count"] == 0

    # 完整窗口：两个边界都在，成一个事件
    body = client.get(
        "/api/v1/incidents", params={"start": _iso(6), "end": _iso(7)}
    ).json()
    assert body["boundary_count"] == 2
    assert body["event_count"] == 1


def test_incidents_empty_database_returns_empty(client):
    r = client.get("/api/v1/incidents")
    assert r.status_code == 200
    body = r.json()
    assert body["events"] == []
    assert body["event_count"] == 0
    assert body["boundary_count"] == 0


def test_incidents_stable_case_without_boundary(client):
    # 完全平稳的展柜只有基线段，不产生边界
    rows = [
        _reading("CASE-IS", _iso(i), 20.0 + 0.01 * (i % 2), 55.0) for i in range(8)
    ]
    _ingest(client, rows)
    body = client.get("/api/v1/incidents").json()
    assert body["boundary_count"] == 0
    assert body["event_count"] == 0


def test_incidents_uses_on_demand_segmentation_without_snapshot(client):
    # replay=false：无落库快照，仍应基于即时计算的有效分段检出事件
    _ingest(client, _step_rows("CASE-IO1", 6, jump_t=4.0), replay=False)
    _ingest(client, _step_rows("CASE-IO2", 7, jump_t=4.0), replay=False)
    assert client.get("/api/v1/cases").json() == []
    body = client.get("/api/v1/incidents", params={"tolerance_seconds": 120}).json()
    assert body["event_count"] == 1
    assert body["events"][0]["device_count"] == 2


def test_incidents_deterministic(client):
    rows = []
    for k, sn in enumerate(["CASE-ID1", "CASE-ID2", "CASE-ID3"]):
        rows += _step_rows(sn, 6 + k, jump_t=4.0)
    _ingest(client, rows)
    params = {"tolerance_seconds": 180}
    first = client.get("/api/v1/incidents", params=params).json()
    second = client.get("/api/v1/incidents", params=params).json()
    assert first == second
    # 重复上报合并后重查，事件不变
    _ingest(client, rows)
    third = client.get("/api/v1/incidents", params=params).json()
    assert third["events"] == first["events"]


def test_incidents_validation_errors(client):
    _ingest(client, _step_rows("CASE-IE", 6, jump_t=4.0))
    url = "/api/v1/incidents"
    # start 晚于 end
    assert (
        client.get(url, params={"start": _iso(10), "end": _iso(5)}).status_code == 422
    )
    # 非正容差
    assert client.get(url, params={"tolerance_seconds": 0}).status_code == 422
    assert client.get(url, params={"tolerance_seconds": -5}).status_code == 422
    # min_devices 小于 2
    assert client.get(url, params={"min_devices": 1}).status_code == 422
    assert client.get(url, params={"min_devices": 0}).status_code == 422
    # 非法参数类型
    assert client.get(url, params={"tolerance_seconds": "abc"}).status_code == 422
    assert client.get(url, params={"min_devices": "x"}).status_code == 422
    # 合法边界值
    assert (
        client.get(url, params={"tolerance_seconds": 0.5, "min_devices": 2}).status_code
        == 200
    )
