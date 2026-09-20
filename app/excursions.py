"""越界剂量（excursion dose）计算。

把展柜的温湿度读数视为**分段线性**曲线，对超出阈值的部分随时间积分
（剂量，单位 量纲·秒：温度 °C·s、湿度 %RH·s）：

- 阈值交点按相邻读数**线性插值**精确求出，不做阶梯近似；
- 相邻读数间隔**大于** ``max_gap`` 时曲线断开，未知区间不计剂量
  （恰好等于 ``max_gap`` 仍视为连续）；
- 等于阈值不算越界（严格大于上限 / 严格小于下限）；
- 任一维度越界的**连续区间合并为事件**，事件按开始时间稳定排序。

全部计算确定性：相同输入（读数、阈值、max_gap）恒得相同事件序列。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Protocol

TEMPERATURE = "temperature"
HUMIDITY = "humidity"


class _SampleLike(Protocol):
    ts: dt.datetime
    temperature: float
    humidity: float


@dataclass(frozen=True)
class Thresholds:
    """越界阈值（°C / %RH）；至少配置一项（由接口层校验）。"""

    temp_min: float | None = None
    temp_max: float | None = None
    humidity_min: float | None = None
    humidity_max: float | None = None

    def configured(self) -> bool:
        return any(
            v is not None
            for v in (
                self.temp_min,
                self.temp_max,
                self.humidity_min,
                self.humidity_max,
            )
        )


@dataclass
class _Interval:
    """单一维度上的一段连续越界区间（剂量已对时间积分）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    dose: float
    peak_violation: float  # 区间内最大越界幅度（量纲单位）
    peak_value: float  # 峰值越界处的物理量取值（必为某个读数值）
    peak_ts: dt.datetime


@dataclass
class ExcursionEvent:
    """任一维度越界的连续区间（跨维度合并后）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    dimensions: tuple[str, ...]
    duration_seconds: float
    temp_dose: float
    humidity_dose: float
    peak_temperature: float | None
    peak_temperature_ts: dt.datetime | None
    peak_humidity: float | None
    peak_humidity_ts: dt.datetime | None


def _cross_ts(t0: dt.datetime, t1: dt.datetime, frac: float) -> dt.datetime:
    """插值出交点时刻；端点分数直接返回原对象，保证相邻区间精确相接。"""
    if frac <= 0.0:
        return t0
    if frac >= 1.0:
        return t1
    return t0 + dt.timedelta(seconds=frac * (t1 - t0).total_seconds())


def _branch_interval(
    t0: dt.datetime,
    t1: dt.datetime,
    v0: float,
    v1: float,
    e0: float,
    e1: float,
) -> _Interval | None:
    """单采样区间、单方向的越界子区间。

    e(t) 为线性越界函数（上限分支 e = v - hi，下限分支 e = lo - v），
    越界部分为 e > 0 的区间；剂量即该区间上 e 的三角形/梯形面积。
    峰值必落在区间端点（即某个读数）上。
    """
    if e0 <= 0.0 and e1 <= 0.0:
        return None
    span = (t1 - t0).total_seconds()
    f0 = 0.0 if e0 > 0.0 else e0 / (e0 - e1)
    f1 = 1.0 if e1 > 0.0 else e0 / (e0 - e1)
    dose = 0.5 * (max(e0, 0.0) + max(e1, 0.0)) * (f1 - f0) * span
    if e0 >= e1:
        peak_violation, peak_value, peak_ts = e0, v0, t0
    else:
        peak_violation, peak_value, peak_ts = e1, v1, t1
    return _Interval(
        start_ts=_cross_ts(t0, t1, f0),
        end_ts=_cross_ts(t0, t1, f1),
        dose=dose,
        peak_violation=peak_violation,
        peak_value=peak_value,
        peak_ts=peak_ts,
    )


def _dimension_intervals(
    times: list[dt.datetime],
    values: list[float],
    lo: float | None,
    hi: float | None,
    max_gap_seconds: float,
) -> list[_Interval]:
    """单维度（含上、下限两个分支）的越界区间，连续段已合并。"""
    raw: list[_Interval] = []
    for i in range(len(times) - 1):
        span = (times[i + 1] - times[i]).total_seconds()
        if span <= 0.0 or span > max_gap_seconds:
            continue  # 采样间隔超过 max_gap：断开，未知区间不计剂量
        t0, t1 = times[i], times[i + 1]
        v0, v1 = values[i], values[i + 1]
        if hi is not None:
            iv = _branch_interval(t0, t1, v0, v1, v0 - hi, v1 - hi)
            if iv is not None:
                raw.append(iv)
        if lo is not None:
            iv = _branch_interval(t0, t1, v0, v1, lo - v0, lo - v1)
            if iv is not None:
                raw.append(iv)

    raw.sort(key=lambda iv: (iv.start_ts, iv.end_ts))
    merged: list[_Interval] = []
    for iv in raw:
        if merged and iv.start_ts <= merged[-1].end_ts:
            cur = merged[-1]
            cur.end_ts = max(cur.end_ts, iv.end_ts)
            cur.dose += iv.dose
            if iv.peak_violation > cur.peak_violation:  # 并列保留更早出现者
                cur.peak_violation = iv.peak_violation
                cur.peak_value = iv.peak_value
                cur.peak_ts = iv.peak_ts
        else:
            merged.append(iv)
    return merged


def compute_excursions(
    samples: list[_SampleLike],
    thresholds: Thresholds,
    max_gap_seconds: float,
) -> list[ExcursionEvent]:
    """计算越界事件序列（按开始时间稳定排序）。

    samples 无需预先排序；同一展柜的读数应已按 (device_sn, ts) 合并。
    """
    ordered = sorted(samples, key=lambda s: s.ts)
    times = [s.ts for s in ordered]
    temps = [s.temperature for s in ordered]
    hums = [s.humidity for s in ordered]

    dim_intervals = {
        TEMPERATURE: _dimension_intervals(
            times, temps, thresholds.temp_min, thresholds.temp_max, max_gap_seconds
        ),
        HUMIDITY: _dimension_intervals(
            times, hums, thresholds.humidity_min, thresholds.humidity_max, max_gap_seconds
        ),
    }

    # 跨维度合并：任一维度越界的连续区间（取并集）归为同一事件
    tagged = [
        (iv, dim)
        for dim in (TEMPERATURE, HUMIDITY)
        for iv in dim_intervals[dim]
    ]
    tagged.sort(key=lambda x: (x[0].start_ts, x[0].end_ts, x[1]))

    groups: list[dict] = []
    for iv, dim in tagged:
        if groups and iv.start_ts <= groups[-1]["end_ts"]:
            groups[-1]["end_ts"] = max(groups[-1]["end_ts"], iv.end_ts)
        else:
            groups.append(
                {
                    "start_ts": iv.start_ts,
                    "end_ts": iv.end_ts,
                    TEMPERATURE: [],
                    HUMIDITY: [],
                }
            )
        groups[-1][dim].append(iv)

    events: list[ExcursionEvent] = []
    for g in groups:
        t_ivs: list[_Interval] = g[TEMPERATURE]
        h_ivs: list[_Interval] = g[HUMIDITY]
        # max 并列时保留先出现者（列表已按时间升序）
        t_peak = max(t_ivs, key=lambda iv: iv.peak_violation, default=None)
        h_peak = max(h_ivs, key=lambda iv: iv.peak_violation, default=None)
        events.append(
            ExcursionEvent(
                start_ts=g["start_ts"],
                end_ts=g["end_ts"],
                dimensions=tuple(
                    d for d in (TEMPERATURE, HUMIDITY) if g[d]
                ),
                duration_seconds=(g["end_ts"] - g["start_ts"]).total_seconds(),
                temp_dose=sum(iv.dose for iv in t_ivs),
                humidity_dose=sum(iv.dose for iv in h_ivs),
                peak_temperature=t_peak.peak_value if t_peak else None,
                peak_temperature_ts=t_peak.peak_ts if t_peak else None,
                peak_humidity=h_peak.peak_value if h_peak else None,
                peak_humidity_ts=h_peak.peak_ts if h_peak else None,
            )
        )
    return events
