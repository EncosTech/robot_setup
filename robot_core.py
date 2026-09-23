"""Hardware-independent parsing and safety checks for the robot setup wizard.

The encos driver may inject warning/error lines into emcli output.  Nothing in
this module trusts line numbers or the last line: it accepts only complete,
strictly shaped business records and ignores everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import math
from pathlib import Path
import re
import subprocess
from typing import Iterable


ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POSITION_RE = re.compile(rf"^position\s+({FLOAT})\s*$")
STRESS_RE = re.compile(
    rf"^\s*(Overview|Adapter\s+.+?|(?:Slave\s+(\d+)\s+)?Bus\s+(\d+)|Motor\s+(\d+)):\s*"
    rf"drop_rate=({FLOAT})%\s+drop=(\d+)/(\d+)\s+actual_fps=({FLOAT})\s*$"
)
FINAL_RE = re.compile(r"^=+\s*FINAL RESULT\s*=+\s*$", re.IGNORECASE)
PLAY_STATUS_RE = re.compile(r"^=+\s*PLAY STRESS STATUS.*=+\s*$", re.IGNORECASE)
PLAY_MOTOR_RE = re.compile(
    r"^Motor\s+(.+):\s+sent=(\d+)\s+received=(\d+)\s+lost=(\d+)\s+loss=" + FLOAT + r"%\s*$"
)
ETHERCAT_CONNECTION_MISSING_RE = re.compile(
    r"\bfailed\s+to\s+initialize\s+ethercat\b", re.IGNORECASE
)
SCAN_HEADER = ("slave", "bus", "id", "eff", "canfd")


class ParseError(ValueError):
    """The command exited but did not emit a complete recognizable report."""


@dataclass(frozen=True)
class Joint:
    label: str
    slave: int
    bus: int
    motor_id: int
    pos: float
    pmin: float | None
    pmax: float | None
    image: str
    text: str
    image2: str = ""
    type: str = "motor"

    @property
    def key(self) -> tuple[int, int, int]:
        return self.slave, self.bus, self.motor_id

    @property
    def requires_limit_check(self) -> bool:
        return self.pmin is not None or self.pmax is not None


def format_joint_button_label(joint: Joint) -> str:
    return f"{joint.label}（ID {joint.motor_id}）"


@dataclass(frozen=True)
class ScanRow:
    # `slave` is the robot logical value used by the CSV (blank emcli cell = 0).
    # `raw_slave` preserves the emcli addressing form so motor commands can
    # omit that component when this is a direct, non-slave EtherCAT bus.
    slave: int
    raw_slave: int | None
    bus: int
    motor_id: int
    eff: int
    canfd: int

    @property
    def key(self) -> tuple[int, int, int]:
        return self.slave, self.bus, self.motor_id


@dataclass
class TopologyReport:
    missing_labels: list[str] = field(default_factory=list)
    extra_ids: list[str] = field(default_factory=list)
    transport_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (self.missing_labels or self.extra_ids or self.transport_errors)

    @property
    def can_proceed(self) -> bool:
        """Extra motors are diagnostic; missing or misconfigured expected motors block progress."""
        return not (self.missing_labels or self.transport_errors)


@dataclass(frozen=True)
class StressStat:
    drop_rate: float
    drop: int
    sent: int
    actual_fps: float


@dataclass
class StressResult:
    measurement_seconds: float | None
    overview: StressStat
    buses: dict[tuple[int | None, int], StressStat]
    motors: dict[tuple[int | None, int, int], StressStat]


@dataclass
class PacketLossDiagnosis:
    over_limit_motors: list[tuple[int | None, int, int]] = field(default_factory=list)
    overall_p90: float = 0.0
    overall_p90_high: bool = False
    abnormal_buses: list[tuple[int | None, int]] = field(default_factory=list)
    abnormal_motors: list[tuple[int | None, int, int]] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        # P90/bus/motor comparisons explain a loss failure; they do not set
        # the pass/fail outcome by themselves.
        return bool(self.over_limit_motors)


@dataclass(frozen=True)
class ImuReport:
    accel: tuple[float, float, float]
    gyro: tuple[float, float, float]
    euler: tuple[float, float, float]


@dataclass(frozen=True)
class ImuConfig:
    label: str
    slave: int
    bus: int
    device_id: int
    pos: tuple[float | None, float | None, float | None]
    pmin: float
    pmax: float
    image: str
    text: str
    image2: str = ""


@dataclass(frozen=True)
class ImuEvaluation:
    average: tuple[float, float, float]
    ranges: tuple[float, float, float]
    mean_ok: bool
    range_ok: bool

    @property
    def passed(self) -> bool:
        return self.mean_ok and self.range_ok


def clean_output(text: str) -> str:
    return ANSI_RE.sub("", text).replace("\r", "")


def is_ethercat_connection_missing(text: str) -> bool:
    """Recognize the driver's no-link initialization error in noisy output."""
    return bool(ETHERCAT_CONNECTION_MISSING_RE.search(clean_output(text)))


def _lines(text: str) -> list[str]:
    return clean_output(text).splitlines()


def _finite(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite number")
    return parsed


def parse_scan_output(text: str) -> list[ScanRow]:
    """Read scan rows after a complete header, ignoring all unknown log lines."""
    after_header = False
    rows: list[ScanRow] = []
    for line in _lines(text):
        fields = line.split()
        if tuple(field.lower() for field in fields) == SCAN_HEADER:
            after_header = True
            continue
        if not after_header or len(fields) not in (4, 5):
            continue
        try:
            # emcli leaves the slave cell blank for a non-slave adapter. The
            # treats that connection as slave 0 as requested by the workflow.
            if len(fields) == 4:
                slave, raw_slave, bus, motor_id, eff, canfd = 0, None, *(int(value) for value in fields)
            else:
                raw_slave, bus, motor_id, eff, canfd = (int(value) for value in fields)
                slave = raw_slave
        except ValueError:
            continue
        if eff not in (0, 1) or canfd not in (0, 1):
            continue
        rows.append(ScanRow(slave, raw_slave, bus, motor_id, eff, canfd))
    if not after_header:
        raise ParseError("未找到完整 scan 表头（slave bus id eff canfd）")
    if not rows:
        raise ParseError("scan 输出中未找到任何完整电机记录")
    if len({row.key for row in rows}) != len(rows):
        raise ParseError("scan 输出包含重复的电机地址")
    return rows


def format_scan_choice_labels(rows: Iterable[ScanRow]) -> list[str]:
    """Format ID-choice buttons without hiding an ambiguous bus/slave address.

    The ID remains the final component.  Its bus and slave prefix is shown
    only when that component varies among the scan candidates.
    """
    candidates = list(rows)
    has_multiple_slaves = len({row.slave for row in candidates}) > 1
    has_multiple_buses = len({row.bus for row in candidates}) > 1
    labels: list[str] = []
    for row in candidates:
        components: list[str] = []
        if has_multiple_slaves:
            components.append(str(row.slave))
        if has_multiple_buses:
            components.append(str(row.bus))
        components.append(str(row.motor_id))
        labels.append(":".join(components))
    return labels


def parse_interface_output(text: str, available_interfaces: set[str] | None = None) -> list[str]:
    """Read `emcli scan Ethercat` interface names while discarding driver logs."""
    interface_re = re.compile(r"^[A-Za-z0-9_.-]+$")
    excluded = {"error", "warning", "warn", "info", "debug"}
    names = []
    for line in _lines(text):
        candidate = line.strip()
        if (interface_re.fullmatch(candidate) and candidate.lower() not in excluded
                and (available_interfaces is None or candidate in available_interfaces)):
            names.append(candidate)
    return list(dict.fromkeys(names))


def parse_position_output(text: str) -> float:
    values: list[float] = []
    for line in _lines(text):
        match = POSITION_RE.fullmatch(line.strip())
        if match:
            try:
                values.append(_finite(match.group(1)))
            except ValueError:
                pass
    if not values:
        raise ParseError("未找到完整的 position 回读记录")
    return values[-1]


def _parse_stat(match: re.Match[str]) -> StressStat:
    rate_percent, drop, sent, fps = match.group(5), match.group(6), match.group(7), match.group(8)
    rate, fps_value = _finite(rate_percent), _finite(fps)
    drop_value, sent_value = int(drop), int(sent)
    if rate < 0 or drop_value < 0 or sent_value < 0 or fps_value < 0:
        raise ValueError("negative stress value")
    # emcli formats drop_rate to two decimal percentage places.  Thresholds
    # here are below that resolution, so use its exact counters instead.
    exact_rate = (drop_value / sent_value) if sent_value else 0.0
    return StressStat(exact_rate, drop_value, sent_value, fps_value)


def parse_stress_final_result(text: str) -> StressResult:
    """Parse only the final full report; preceding live snapshots never count."""
    lines = _lines(text)
    marker_indexes = [index for index, line in enumerate(lines) if FINAL_RE.fullmatch(line.strip())]
    if not marker_indexes:
        raise ParseError("未找到 FINAL RESULT 压测结果块")

    overview: StressStat | None = None
    measurement_seconds: float | None = None
    buses: dict[tuple[int | None, int], StressStat] = {}
    motors: dict[tuple[int | None, int, int], StressStat] = {}
    current_bus: tuple[int | None, int] | None = None
    for line in lines[marker_indexes[-1] + 1 :]:
        stripped = line.strip()
        if stripped.startswith("Measurement seconds:"):
            try:
                measurement_seconds = _finite(stripped.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                continue
            continue
        match = STRESS_RE.fullmatch(line)
        if not match:
            continue
        try:
            stat = _parse_stat(match)
        except ValueError:
            continue
        label, slave, bus, motor = match.group(1), match.group(2), match.group(3), match.group(4)
        if label == "Overview":
            overview = stat
        elif label.startswith("Adapter "):
            current_bus = None
        elif "Bus " in label:
            current_bus = (int(slave) if slave is not None else 0, int(bus))
            buses[current_bus] = stat
        elif label.startswith("Motor ") and current_bus is not None:
            motors[(current_bus[0], current_bus[1], int(motor))] = stat
    if overview is None or not buses or not motors:
        raise ParseError("FINAL RESULT 不完整：缺少 Overview、Bus 或 Motor 统计")
    return StressResult(measurement_seconds, overview, buses, motors)


def _play_motor_key(name: str) -> tuple[int, int, int]:
    """Decode the emcli trajectory motor target without treating adapter id ':' specially."""
    parts = name.split(":")
    if len(parts) == 4:
        _, _, bus, motor = parts
        return 0, int(bus), int(motor)
    if len(parts) == 5:
        _, _, slave, bus, motor = parts
        return int(slave), int(bus), int(motor)
    raise ValueError("trajectory motor target must have 4 or 5 colon-separated parts")


def parse_play_stress_result(text: str) -> StressResult:
    """Parse play's final post-return report (the last complete status block).

    `play --stress` has no FINAL RESULT banner. After Ctrl-C it returns to its
    initial pose for two seconds, drains callbacks, prints one final status
    block, and exits. Since the process is fully reaped before this function is
    called, the last complete block is that post-drain final report.
    """
    lines = _lines(text)
    markers = [index for index, line in enumerate(lines) if PLAY_STATUS_RE.fullmatch(line.strip())]
    if not markers:
        raise ParseError("未找到 PLAY STRESS STATUS 结果块")
    measurement_seconds: float | None = None
    motors: dict[tuple[int | None, int, int], StressStat] = {}
    for line in lines[markers[-1] + 1 :]:
        stripped = line.strip()
        if stripped.startswith("Measurement seconds:"):
            try:
                measurement_seconds = _finite(stripped.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                continue
            continue
        match = PLAY_MOTOR_RE.fullmatch(stripped)
        if not match:
            continue
        try:
            key = _play_motor_key(match.group(1))
            sent, received, lost = (int(match.group(index)) for index in (2, 3, 4))
        except ValueError:
            continue
        if sent < 0 or received < 0 or lost < 0:
            continue
        motors[key] = StressStat((lost / sent) if sent else 0.0, lost, sent,
                                 (received / measurement_seconds) if measurement_seconds else 0.0)
    if measurement_seconds is None or not motors:
        raise ParseError("最后一个 PLAY STRESS STATUS 不完整")
    buses: dict[tuple[int | None, int], StressStat] = {}
    for bus_key in {key[:2] for key in motors}:
        group = [stat for key, stat in motors.items() if key[:2] == bus_key]
        sent, received = sum(stat.sent for stat in group), sum(stat.sent - stat.drop for stat in group)
        buses[bus_key] = StressStat((sent - received) / sent if sent else 0.0, sent - received, sent,
                                    received / measurement_seconds)
    sent, received = sum(stat.sent for stat in motors.values()), sum(stat.sent - stat.drop for stat in motors.values())
    overview = StressStat((sent - received) / sent if sent else 0.0, sent - received, sent,
                          received / measurement_seconds)
    return StressResult(measurement_seconds, overview, buses, motors)


def parse_imu_reports(text: str) -> list[ImuReport]:
    """Return every complete four-line IMU report, with logs allowed between rows."""
    pending: dict[str, tuple[float, float, float]] = {}
    reports: list[ImuReport] = []
    row_re = re.compile(rf"^(accel|gyro|euler)\s+({FLOAT})\s+({FLOAT})\s+({FLOAT})\s*$")
    for raw_line in _lines(text):
        line = raw_line.strip()
        if line.split()[:1] == ["group"]:
            pending = {}
            continue
        match = row_re.fullmatch(line)
        if not match:
            continue
        try:
            pending[match.group(1)] = tuple(_finite(match.group(index)) for index in (2, 3, 4))  # type: ignore[assignment]
        except ValueError:
            continue
        if all(key in pending for key in ("accel", "gyro", "euler")):
            reports.append(ImuReport(pending["accel"], pending["gyro"], pending["euler"]))
            pending = {}
    return reports


def read_hardware_csv(path: Path) -> tuple[list[Joint], list[ImuConfig]]:
    required = ("type", "label", "slave", "bus", "id", "pos", "pmin", "pmax", "image", "image2", "text")
    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or tuple(reader.fieldnames) != required:
            raise ValueError("关节 CSV 表头必须为: " + ",".join(required))
        joints, imus = [], []
        for row in reader:
            if row["type"] == "motor":
                joints.append(Joint(row["label"], int(row["slave"]), int(row["bus"]), int(row["id"]), float(row["pos"]),
                                    _optional_float(row["pmin"]), _optional_float(row["pmax"]),
                                    row["image"], row["text"], row["image2"], "motor"))
            elif row["type"] == "imu":
                raw_pos = row["pos"].strip()
                if not (raw_pos.startswith("[") and raw_pos.endswith("]")):
                    raise ValueError(f"IMU {row['label']} 的 pos 必须是长度为 3 的数组")
                pos = [part.strip() for part in raw_pos[1:-1].split(",")]
                if len(pos) != 3:
                    raise ValueError(f"IMU {row['label']} 的 pos 必须是长度为 3 的数组")
                imus.append(ImuConfig(row["label"], int(row["slave"]), int(row["bus"]), int(row["id"]),
                                      tuple(_optional_float(value) for value in pos), float(row["pmin"]), float(row["pmax"]),
                                      row["image"], row["text"], row["image2"]))
            else:
                raise ValueError(f"不支持的 type：{row['type']}")
    if len({joint.key for joint in joints}) != len(joints) or len({joint.label for joint in joints}) != len(joints):
        raise ValueError("电机关节的 label 和 slave/bus/id 必须各自唯一")
    return joints, imus


def _optional_float(value: str) -> float | None:
    return None if value.strip() == "~" else _finite(value.strip())


def read_joints(path: Path) -> list[Joint]:
    return read_hardware_csv(path)[0]


def evaluate_imu_reports(reports: list[ImuReport], config: ImuConfig) -> ImuEvaluation:
    if not reports:
        raise ValueError("没有完整的 IMU 数据")
    values = list(zip(*(report.euler for report in reports)))
    averages = tuple(sum(axis) / len(axis) for axis in values)
    ranges = tuple(max(axis) - min(axis) for axis in values)
    active_axes = [index for index, target in enumerate(config.pos) if target is not None]
    mean_ok = all(
        config.pmin <= averages[index] - config.pos[index] <= config.pmax  # type: ignore[operator]
        for index in active_axes
    )
    range_ok = not active_axes or any(
        ranges[index] > config.pmax - config.pmin for index in active_axes
    )
    return ImuEvaluation(averages, ranges, mean_ok, range_ok)


def compare_topology(rows: Iterable[ScanRow], joints: Iterable[Joint]) -> TopologyReport:
    scanned = {row.key: row for row in rows}
    expected = {joint.key: joint for joint in joints}
    report = TopologyReport()
    report.missing_labels = [joint.label for key, joint in expected.items() if key not in scanned]
    report.extra_ids = [f"slave={key[0]} bus={key[1]} id={key[2]}" for key in scanned if key not in expected]
    for key, row in scanned.items():
        if row.eff != 0:
            report.transport_errors.append(f"slave={key[0]} bus={key[1]} id={key[2]} 的 eff={row.eff}，应为 0")
        if row.canfd != 1:
            report.transport_errors.append(f"slave={key[0]} bus={key[1]} id={key[2]} 的 canfd={row.canfd}，应为 1")
    return report


# ---------------------------------------------------------------------------
# play 落盘日志（--log 输出）分析：丢包、跟踪精度、温度
# ---------------------------------------------------------------------------

# emcli play --log 每台电机会生成 <名称>_command.csv 与 <名称>_status.csv。
# 名称以三个数字结尾：<bus>_<id>_<id-1>（如 0_2_1），压缩存档时额外带 .zstd。
PLAY_LOG_FILE_RE = re.compile(r"^(.*)_(\d+)_(\d+)_(\d+)_(command|status)\.csv(?:\.zstd)?$")
PLAY_LOG_KINDS = {"command", "status"}
# 跟踪精度判据：平均误差上限（rad）与“连续超差上限时间”（ms）。
# 正常动态滞后允许瞬时/平均误差接近上限；持续无法收敛（腰右型故障）由
# 连续超差时长过滤：20A 实测其余关节最长连续超差 ≤1174 ms，故障腰右 3529 ms。
PLAY_LOG_MEAN_ERROR_LIMIT_RAD = 0.15
PLAY_LOG_TRACKING_LIMIT_RAD = 0.15
PLAY_LOG_CONTINUOUS_LIMIT_MS = 1500
PLAY_LOG_TEMPERATURE_WARN_C = 45.0


@dataclass(frozen=True)
class MotorLogStat:
    key: tuple[int, int, int]
    label: str
    sent: int
    received: int
    missing: int
    drop_rate: float
    mean_error_rad: float
    max_error_rad: float
    longest_over_limit_ms: int
    max_motor_temp_c: float
    max_mos_temp_c: float


@dataclass
class PlayLogResult:
    measurement_seconds: float
    motors: dict[tuple[int, int, int], MotorLogStat]
    unmatched_files: list[str] = field(default_factory=list)


def _read_log_text(path: Path) -> str:
    """Read a driver log CSV, transparently decompressing a .zstd archive."""
    if path.name.endswith(".zstd"):
        try:
            completed = subprocess.run(["zstd", "-d", "-c", str(path)], check=True,
                                       capture_output=True, timeout=60)
        except FileNotFoundError as exc:
            raise ValueError(f"读取压缩日志需要 zstd 命令：{path}") from exc
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"解压日志失败：{path}") from exc
        return completed.stdout.decode("utf-8", errors="replace")
    return path.read_text(encoding="utf-8")


def _parse_log_csv_rows(path: Path, columns: tuple[int, ...]) -> list[tuple]:
    """Parse numeric rows from a driver log CSV, skipping its header line."""
    rows: list[tuple] = []
    text = _read_log_text(path)
    reader = csv.reader(text.splitlines())
    header = next(reader, None)
    if header is None or not any(column.strip() == "timestamp_ns" for column in header):
        raise ValueError(f"日志缺少表头：{path}")
    for raw in reader:
        if not raw or len(raw) <= max(columns):
            continue
        try:
            parsed = tuple(float(raw[index]) for index in columns)
        except ValueError:
            continue
        if not all(math.isfinite(value) for value in parsed):
            raise ValueError(f"日志包含非有限数值：{path}")
        rows.append(parsed)
    return rows


def _motor_log_key_candidates(
    a: int,
    b: int,
    c: int,
    expected: dict[tuple[int, int, int], Joint],
) -> list[tuple[int, int, int]]:
    """Resolve a log filename to full (slave, bus, id) candidates.

    当前驱动命名：<bus>_<id>_<id-1>（H130/H180，slave 均为 0）。 为兼容其他
    命名（如 slave>0 时 <slave>_<bus>_<id>），优先匹配完整地址，再尝试旧格式。
    """
    exact = (a, b, c)
    if exact in expected:
        return [exact]
    legacy = (0, a, b)
    return [legacy] if legacy in expected else []


def analyze_play_log_directory(directory: Path, joints: Iterable[Joint]) -> PlayLogResult:
    """Analyze emcli play --log output for loss, tracking error and temperature.

    丢包以与 emcli 一致的口径统计（lost = command 行数 - status 行数）。
    跟踪误差取 command/status 位置列（弧度）按时间对齐后的全程平均、
    峰值与“连续超过 0.15 rad”的最长时长。温度取 status 中电机/MOS 最高值。
    """
    expected: dict[tuple[int, int, int], Joint] = {joint.key: joint for joint in joints}
    files: dict[tuple[int, int, int], dict[str, Path]] = {}
    unmatched: set[str] = set()
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = PLAY_LOG_FILE_RE.match(path.name)
        if not match:
            continue
        candidates = _motor_log_key_candidates(
            int(match.group(2)), int(match.group(3)), int(match.group(4)), expected,
        )
        if not candidates:
            unmatched.add(path.name)
            continue
        if len(candidates) > 1:
            labels = "、".join(expected[key].label for key in candidates)
            raise ValueError(f"日志文件对应到多个关节（{labels}）：{path.name}")
        key = candidates[0]
        kind = match.group(5)
        if kind in files.setdefault(key, {}):
            raise ValueError(f"关节 {expected[key].label} 存在重复的 {kind} 日志")
        files[key][kind] = path
    if unmatched:
        raise ValueError("无法把日志文件对应到任何关节：\n" + "\n".join(sorted(unmatched)))

    missing_labels = [
        joint.label for key, joint in expected.items()
        if key not in files or set(files[key]) != PLAY_LOG_KINDS
    ]
    if missing_labels:
        raise ValueError("缺少日志文件（电机可能未工作或未响应）：\n" + "、".join(missing_labels))

    motors: dict[tuple[int, int, int], MotorLogStat] = {}
    duration = 0.0
    for key, joint in sorted(expected.items()):
        command_rows = _parse_log_csv_rows(files[key]["command"], (0, 4))  # ts, position
        status_rows = _parse_log_csv_rows(files[key]["status"], (0, 2, 3, 4, 5, 6))
        # status: ts, position, speed, current, motor_temp, mos_temp
        if not command_rows or not status_rows:
            raise ValueError(f"{joint.label} 的日志为空，无法分析。")
        status_ts = [row[0] for row in status_rows]
        if any(current < previous for previous, current in zip(status_ts, status_ts[1:])):
            raise ValueError(f"{joint.label} 的日志时间戳不是单调递增。")
        duration = max(duration, (status_ts[-1] - status_ts[0]) / 1e9)
        # 与 emcli play 的统计口径一致：lost = command 行数 - status 行数。
        sent = len(command_rows)
        received = len(status_rows)
        missing = max(0, sent - received)
        drop_rate = missing / sent if sent else 0.0

        # index-align command and status (both ~1 kHz on the same grid).
        error_sum = 0.0
        max_error = 0.0
        streak_start_ns: float | None = None
        longest_over_limit_ms = 0
        cmd_index = 0
        for status_row in status_rows:
            status_ts_value = status_row[0]
            while (cmd_index + 1 < len(command_rows)
                   and abs(command_rows[cmd_index + 1][0] - status_ts_value)
                   < abs(command_rows[cmd_index][0] - status_ts_value)):
                cmd_index += 1
            error = abs(command_rows[cmd_index][1] - status_row[1])
            error_sum += error
            if error > max_error:
                max_error = error
            if error > PLAY_LOG_TRACKING_LIMIT_RAD:
                if streak_start_ns is None:
                    streak_start_ns = status_ts_value
                current_streak_ms = int(round((status_ts_value - streak_start_ns) / 1e6))
                longest_over_limit_ms = max(longest_over_limit_ms, current_streak_ms)
            else:
                streak_start_ns = None
        mean_error = error_sum / len(status_rows)
        max_motor_temp = max(row[4] for row in status_rows)
        max_mos_temp = max(row[5] for row in status_rows)
        motors[key] = MotorLogStat(
            key=key, label=joint.label, sent=sent, received=received,
            missing=missing, drop_rate=drop_rate, mean_error_rad=mean_error,
            max_error_rad=max_error, longest_over_limit_ms=longest_over_limit_ms,
            max_motor_temp_c=max_motor_temp, max_mos_temp_c=max_mos_temp)
    return PlayLogResult(measurement_seconds=duration, motors=motors, unmatched_files=sorted(unmatched))


def percentile_90(values: Iterable[float]) -> float:
    """Linear-interpolated p90, useful even for small per-bus samples."""
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = 0.9 * (len(sorted_values) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def angular_distance_degrees(left: float, right: float) -> float:
    """Shortest absolute angle difference, including the ±180° seam."""
    return abs((left - right + 180.0) % 360.0 - 180.0)


def analyze_packet_loss(result: StressResult, *, max_drop_rate: float, overall_p90_limit: float,
                        relative_p90_factor: float) -> PacketLossDiagnosis:
    diagnosis = PacketLossDiagnosis()
    diagnosis.over_limit_motors = [key for key, stat in result.motors.items() if stat.drop_rate > max_drop_rate]
    diagnosis.overall_p90 = percentile_90(stat.drop_rate for stat in result.motors.values())
    diagnosis.overall_p90_high = diagnosis.overall_p90 > overall_p90_limit
    bus_motors: dict[tuple[int | None, int], list[tuple[tuple[int | None, int, int], StressStat]]] = {}
    for key, stat in result.motors.items():
        bus_motors.setdefault(key[:2], []).append((key, stat))
    for bus_key, motor_stats in bus_motors.items():
        bus_p90 = percentile_90(stat.drop_rate for _, stat in motor_stats)
        if bus_p90 > diagnosis.overall_p90 * relative_p90_factor:
            diagnosis.abnormal_buses.append(bus_key)
        for motor_key, stat in motor_stats:
            if stat.drop_rate > bus_p90 * relative_p90_factor:
                diagnosis.abnormal_motors.append(motor_key)
    return diagnosis
