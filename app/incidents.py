"""跨展柜共模突变事件检测。

输入为各展柜**有效分段**的非基线段首（即变点边界，携带跳变方向与幅度），
在 ``[start, end]`` 时间范围内寻找跨展柜的共模突变事件：

1. 边界按 ``(ts, device_sn)`` 升序排列后，**候选事件**为其上的连续窗口，
   窗口内最早与最晚边界之差不得超过 ``tolerance_seconds``；
2. 同一展柜在同一候选中只保留**幅度最大**的边界，并列时取**更早**者；
3. 去重后 distinct device 数少于 ``min_devices`` 的候选无效；
4. 对全部合法连续候选窗做**全局选择**（带权区间调度的精确动态规划，
   而非贪心链式归并）：依次以 **设备数总和** 最大、**幅度总和** 最大、
   **起始时间序列** 字典序最小（越早越优）决胜，选出互不重叠
   （窗口下标区间不相交）的事件集合。

全部规则确定性：相同输入恒得相同事件序列。
"""

from __future__ import annotations

import datetime as dt
from bisect import bisect_left
from dataclasses import dataclass

from .config import settings

_EPS = 1e-12

UP = "up"
DOWN = "down"


@dataclass(frozen=True)
class Boundary:
    """某展柜有效分段中一个非基线段的段首（变点边界）。"""

    device_sn: str
    ts: dt.datetime
    magnitude: float
    delta_temperature: float
    delta_humidity: float

    @property
    def direction(self) -> str:
        """跳变方向：温湿度各自缩放加权后的合成变化符号（零视为 up）。"""
        score = (
            settings.TEMP_WEIGHT * self.delta_temperature / settings.TEMP_SCALE
            + settings.HUMIDITY_WEIGHT * self.delta_humidity / settings.HUMIDITY_SCALE
        )
        return UP if score >= 0.0 else DOWN


@dataclass
class IncidentEvent:
    """一次跨展柜共模突变事件（边界已按 (ts, device_sn) 升序）。"""

    start_ts: dt.datetime  # 事件内最早边界
    end_ts: dt.datetime  # 事件内最晚边界
    device_count: int  # 参与展柜数（distinct device）
    total_magnitude: float  # 各展柜保留边界的幅度之和
    boundaries: tuple[Boundary, ...]


@dataclass
class _Candidate:
    """合法候选窗：有序边界序列上的闭下标区间 [lo, hi] 及其去重后事件。"""

    lo: int
    hi: int
    event: IncidentEvent


def _enumerate_candidates(
    ordered: list[Boundary],
    tolerance_seconds: float,
    min_devices: int,
) -> list[_Candidate]:
    """枚举全部合法连续候选窗。

    窗口跨度（最早与最晚边界之差）不超过 ``tolerance_seconds``；同一展柜
    在窗内只保留幅度最大的边界（并列取更早者——边界已按时间升序处理，
    严格更大才替换，天然保留更早者）；去重后 distinct device 数达到
    ``min_devices`` 才成为候选。
    """
    n = len(ordered)
    candidates: list[_Candidate] = []
    hi_max = -1
    for lo in range(n):
        if hi_max < lo - 1:
            hi_max = lo - 1
        while (
            hi_max + 1 < n
            and (ordered[hi_max + 1].ts - ordered[lo].ts).total_seconds()
            <= tolerance_seconds
        ):
            hi_max += 1

        best: dict[str, Boundary] = {}
        for hi in range(lo, hi_max + 1):
            b = ordered[hi]
            cur = best.get(b.device_sn)
            if cur is None or b.magnitude > cur.magnitude + _EPS:
                best[b.device_sn] = b
            if len(best) < min_devices:
                continue
            kept = tuple(sorted(best.values(), key=lambda x: (x.ts, x.device_sn)))
            candidates.append(
                _Candidate(
                    lo=lo,
                    hi=hi,
                    event=IncidentEvent(
                        start_ts=kept[0].ts,
                        end_ts=kept[-1].ts,
                        device_count=len(kept),
                        total_magnitude=sum(x.magnitude for x in kept),
                        boundaries=kept,
                    ),
                )
            )
    return candidates


# 选择目标：(设备数总和, 幅度总和, 起始时间序列)
_Objective = tuple[int, float, tuple[dt.datetime, ...]]


def _better(a: _Objective, b: _Objective) -> bool:
    """``a`` 是否优于 ``b``：设备数 → 幅度 → 起始时间序列（字典序小者优）。"""
    if a[0] != b[0]:
        return a[0] > b[0]
    if abs(a[1] - b[1]) > _EPS:
        return a[1] > b[1]
    return a[2] < b[2]


def _select_non_overlapping(candidates: list[_Candidate]) -> list[IncidentEvent]:
    """全局选择互不重叠的候选，使目标三元组按字典序最优（精确 DP）。

    候选按窗口结束下标（再按起始下标）排序；``dp[i]`` 为只考虑前 ``i``
    个候选时的最优目标值。转移为经典的“跳过 / 选取”：

        dp[i] = best(dp[i-1], dp[p(i)] + candidate[i])

    其中 ``p(i)`` 为与候选 ``i`` 下标区间不相交（``hi < lo_i``）的候选
    数量，由二分取得。完全相同的目标值时保留“跳过”分支（事件更少），
    保证结果确定。
    """
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda c: (c.hi, c.lo))
    his = [c.hi for c in cands]
    m = len(cands)

    dp: list[_Objective] = [(0, 0.0, ())] * (m + 1)
    parent: list[tuple[int, int | None]] = [(0, None)] * (m + 1)

    for i in range(1, m + 1):
        c = cands[i - 1]
        skip = dp[i - 1]
        p = bisect_left(his, c.lo)  # hi < c.lo 的候选数量，即与 c 不重叠的前缀
        base = dp[p]
        take: _Objective = (
            base[0] + c.event.device_count,
            base[1] + c.event.total_magnitude,
            base[2] + (c.event.start_ts,),
        )
        if _better(take, skip):
            dp[i] = take
            parent[i] = (p, i - 1)
        else:
            dp[i] = skip
            parent[i] = (i - 1, None)

    chosen: list[IncidentEvent] = []
    i = m
    while i > 0:
        prev_i, cand_idx = parent[i]
        if cand_idx is not None:
            chosen.append(cands[cand_idx].event)
        i = prev_i
    chosen.reverse()
    return chosen


def compute_incidents(
    boundaries: list[Boundary],
    tolerance_seconds: float,
    min_devices: int,
) -> list[IncidentEvent]:
    """对边界集合执行候选枚举与全局选择，返回按开始时间排序的事件。

    ``boundaries`` 无需预先排序；应已按调用方的时间范围过滤。
    """
    ordered = sorted(boundaries, key=lambda b: (b.ts, b.device_sn))
    candidates = _enumerate_candidates(ordered, tolerance_seconds, min_devices)
    events = _select_non_overlapping(candidates)
    events.sort(
        key=lambda e: (e.start_ts, e.end_ts, tuple(b.device_sn for b in e.boundaries))
    )
    return events
