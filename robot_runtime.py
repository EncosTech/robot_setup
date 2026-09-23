"""Standard-library process and configuration support for the robot NiceGUI app."""

from __future__ import annotations

import asyncio
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import zipfile

from robot_core import ImuReport, angular_distance_degrees, parse_imu_reports


MIN_EMCLI_VERSION = (1, 9, 9)
VERSION_RE = re.compile(r"^emcli\s+(\d+)\.(\d+)\.(\d+)(?:[-+][^\s]+)?\s*$", re.MULTILINE)
DEB_MEMBER_RE = re.compile(r"^(libencosdriver|emcli)_([^_]+)_([^_]+)_([^_]+)\.deb$", re.IGNORECASE)
SYSTEM_DEFAULTS: dict[str, float | bool | str] = {
    # Loss/position thresholds.
    "max_drop_rate": 0.0005,
    "overall_p90_limit": 0.00075,
    "relative_p90_factor": 1.1,
    "limit_tolerance_deg": 4.0,
    "limit_confirmations": 3.0,
    "position_tolerance_deg": 0.1,
    "imu_move_deg": 10.0,
    # Overall robot-configuration steps.
    "enable_connect": True,
    "enable_scan": True,
    "enable_stress": True,
    "enable_calibration": True,
    "enable_imu": True,
    "enable_motion": True,
}


class CommandError(RuntimeError):
    def __init__(self, message: str, result: "ProcessResult | None" = None, raw_output: str = ""):
        super().__init__(message)
        self.result = result
        self.raw_output = raw_output


@dataclass
class ProcessResult:
    args: list[str]
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float

    @property
    def combined_output(self) -> str:
        # The driver can emit logs to either stream. Parsers deliberately treat
        # the concatenation as noisy input rather than assuming a line order.
        return self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr


@dataclass(frozen=True)
class PackageArchiveCheck:
    usable: bool
    message: str
    driver_member: str | None = None
    cli_member: str | None = None


def _package_version(value: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    return tuple(int(match.group(index)) for index in (1, 2, 3)) if match else None


def inspect_package_archive(archive: Path, codename: str, architecture: str) -> PackageArchiveCheck:
    """Validate and select DEBs compatible with a specific Ubuntu platform."""
    if archive.suffix.lower() != ".zip" or not archive.is_file() or not zipfile.is_zipfile(archive):
        return PackageArchiveCheck(False, "安装包不是可用的 ZIP 文件")
    drivers: list[tuple[tuple[int, int, int], str]] = []
    clis: list[tuple[tuple[int, int, int], str]] = []
    old_cli_found = False
    try:
        with zipfile.ZipFile(archive) as package_zip:
            for member in package_zip.infolist():
                parts = Path(member.filename).parts
                if Path(member.filename).is_absolute() or ".." in parts:
                    return PackageArchiveCheck(False, "安装包 ZIP 包含不安全路径")
                match = DEB_MEMBER_RE.fullmatch(Path(member.filename).name)
                if not match:
                    continue
                package, version_text, member_codename, member_architecture = match.groups()
                if member_codename != codename or member_architecture != architecture:
                    continue
                version = _package_version(version_text)
                if version is None:
                    continue
                if package.lower() == "libencosdriver":
                    drivers.append((version, member.filename))
                elif version >= MIN_EMCLI_VERSION:
                    clis.append((version, member.filename))
                else:
                    old_cli_found = True
    except (OSError, zipfile.BadZipFile):
        return PackageArchiveCheck(False, "无法读取安装包 ZIP 文件")
    if not drivers:
        return PackageArchiveCheck(False, f"未找到适用于 {codename}/{architecture} 的 libencosdriver 安装包")
    if not clis:
        message = f"未找到适用于 {codename}/{architecture} 且版本不低于 1.9.9 的 emcli 安装包"
        if old_cli_found:
            message = f"ZIP 中适用于 {codename}/{architecture} 的 emcli 版本低于最低版本 1.9.9"
        return PackageArchiveCheck(False, message)
    driver = max(drivers, key=lambda item: item[0])[1]
    cli = max(clis, key=lambda item: item[0])[1]
    return PackageArchiveCheck(True, "安装包可用", driver, cli)


def package_archive_check_for_current_platform(archive: Path) -> PackageArchiveCheck:
    """Inspect an archive against the release codename and architecture in use."""
    try:
        codename_result = subprocess.run(["lsb_release", "-cs"], capture_output=True, text=True, timeout=5, check=False)
        architecture_result = subprocess.run(["dpkg", "--print-architecture"], capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return PackageArchiveCheck(False, "无法识别当前系统的发行版代号或架构")
    codename = codename_result.stdout.strip()
    architecture = architecture_result.stdout.strip()
    if codename_result.returncode != 0 or architecture_result.returncode != 0 or not codename or not architecture:
        return PackageArchiveCheck(False, "无法识别当前系统的发行版代号或架构")
    return inspect_package_archive(archive, codename, architecture)


def runtime_package_archive(runtime_root: Path) -> Path:
    """Return the only package archive location accepted by the application."""
    return runtime_root / "packages.zip"


def default_config() -> dict:
    return {
        "robot_type": "",
        "plugin": "Ethercat",
        "interface": "",
    }


def load_config(path: Path) -> dict:
    config = default_config()
    if not path.exists():
        return config
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#") or ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        key, value = key.strip(), value.strip()
        if key not in config:
            continue
        try:
            if value in ("true", "false"):
                config[key] = value == "true"
            elif value.startswith(("[", "{")):
                config[key] = json.loads(value)
            elif isinstance(config[key], int):
                config[key] = int(value)
            else:
                config[key] = json.loads(value) if value.startswith('"') else value
        except (ValueError, SyntaxError):
            continue
    return config


def save_config(path: Path, config: dict) -> None:
    content = "# Robot setup wizard state. This file uses a deliberately small YAML subset.\n"
    for key in ("robot_type", "plugin", "interface"):
        value = config[key]
        if isinstance(value, str):
            rendered = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        content += f"{key}: {rendered}\n"
    path.write_text(content, encoding="utf-8")


def config_directory_sha1(directory: Path) -> str:
    """Return a deterministic digest of every file in a robot configuration directory."""
    if not directory.is_dir():
        raise ValueError(f"机器人配置目录不存在：{directory}")
    digest = hashlib.sha1()
    for path in sorted((item for item in directory.rglob("*") if item.is_file()), key=lambda item: item.relative_to(directory).as_posix()):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def load_progress(path: Path) -> dict | None:
    """Load a locally stored, user-resumable wizard state if it is valid JSON."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def save_progress(path: Path, progress: dict) -> None:
    """Atomically replace the progress file so interruption cannot leave partial JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(progress, temporary, ensure_ascii=False, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def remove_progress(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def load_system_settings(path: Path) -> dict[str, float | bool | str]:
    """Read flat system.yaml settings, retaining safe defaults for bad values."""
    settings = SYSTEM_DEFAULTS.copy()
    if not path.exists():
        return settings
    for raw in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw or raw.lstrip().startswith("#"):
            continue
        key, value = (part.strip() for part in raw.split(":", 1))
        default = settings.get(key)
        if default is None:
            continue
        if isinstance(default, bool):
            if value.lower() in ("true", "false"):
                settings[key] = value.lower() == "true"
            continue
        if isinstance(default, str):
            settings[key] = value.strip().strip('"')
            continue
        try:
            settings[key] = float(value)
        except ValueError:
            continue
    return settings


def parse_interface_link_state(output: str) -> bool:
    """Return whether `ip -j link` reports an administratively and physically up interface."""
    try:
        rows = json.loads(output)
    except json.JSONDecodeError:
        return False
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        return False
    row = rows[0]
    flags = row.get("flags")
    return row.get("operstate") == "UP" and isinstance(flags, list) and {"UP", "LOWER_UP"}.issubset(flags)


async def interface_link_is_up(interface: str) -> bool:
    """Query Linux link state without involving an emcli process."""
    try:
        result = await run_command(["ip", "-j", "link", "show", "dev", interface], timeout_seconds=2)
    except (CommandError, OSError):
        return False
    return result.returncode == 0 and parse_interface_link_state(result.stdout)


def adapter_target(config: dict) -> str:
    interface = str(config.get("interface", "")).strip()
    if not interface:
        raise ValueError("尚未检测到 EtherCAT 网卡")
    return f"{config.get('plugin', 'Ethercat')}:{interface}"


def _raw_slave(config: dict, logical_slave: int, bus: int) -> int | None:
    return None if logical_slave == 0 else logical_slave


def motor_target(config: dict, slave: int, bus: int, motor_id: int) -> str:
    raw_slave = _raw_slave(config, slave, bus)
    if raw_slave is None:
        return f"{adapter_target(config)}:{bus}:{motor_id}"
    return f"{adapter_target(config)}:{raw_slave}:{bus}:{motor_id}"


def bus_target(config: dict, slave: int, bus: int) -> str:
    raw_slave = _raw_slave(config, slave, bus)
    return f"{adapter_target(config)}:{bus}" if raw_slave is None else f"{adapter_target(config)}:{raw_slave}:{bus}"


def raw_motor_target(config: dict, raw_slave: int | None, bus: int, motor_id: int) -> str:
    return f"{adapter_target(config)}:{bus}:{motor_id}" if raw_slave is None else f"{adapter_target(config)}:{raw_slave}:{bus}:{motor_id}"


def imu_target(config: dict, slave: int, bus: int, index: int) -> str:
    raw_slave = _raw_slave(config, slave, bus)
    if raw_slave is None:
        return f"{adapter_target(config)}:{bus}:{index}"
    return f"{adapter_target(config)}:{raw_slave}:{bus}:{index}"


async def _finish_after_signal(process: asyncio.subprocess.Process, communicate_task: asyncio.Task,
                               grace_seconds: float) -> tuple[bytes, bytes]:
    """Allow emcli's normal Ctrl-C cleanup before escalating without cancelling pipe readers."""
    try:
        return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=grace_seconds)
    except TimeoutError:
        _stop_process_group(process, signal.SIGTERM)
    try:
        return await asyncio.wait_for(asyncio.shield(communicate_task), timeout=3)
    except TimeoutError:
        _stop_process_group(process, signal.SIGKILL)
        return await communicate_task


async def run_command(args: list[str], *, timeout_seconds: float | None = None, cwd: str | None = None, on_started=None) -> ProcessResult:
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
            cwd=cwd,
        )
    except FileNotFoundError as exc:
        raise CommandError(f"找不到命令：{args[0]}") from exc
    try:
        if on_started:
            on_started(process)
        communicate_task = asyncio.create_task(process.communicate())
        stdout_bytes, stderr_bytes = await asyncio.wait_for(asyncio.shield(communicate_task), timeout=timeout_seconds)
    except TimeoutError:
        _stop_process_group(process, signal.SIGINT)
        stdout_bytes, stderr_bytes = await _finish_after_signal(process, communicate_task, 5)
        result = ProcessResult(args, process.returncode or -1, stdout_bytes.decode(errors="replace"),
                               stderr_bytes.decode(errors="replace"), time.monotonic() - started)
        raise CommandError(f"命令超时：{' '.join(args)}", result)
    return ProcessResult(args, process.returncode, stdout_bytes.decode(errors="replace"), stderr_bytes.decode(errors="replace"),
                         time.monotonic() - started)


def _stop_process_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def interrupt_process(process: asyncio.subprocess.Process) -> None:
    """Send Ctrl-C to a running command without waiting for it to exit."""
    _stop_process_group(process, signal.SIGINT)


async def run_timed_command(args: list[str], seconds: float, on_tick=None, on_started=None,
                            *, cwd: str | None = None, stop_on_timeout: bool = True, on_timeout=None) -> ProcessResult:
    """Run a Ctrl-C controlled emcli job and always reap its process group."""
    started = time.monotonic()
    try:
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True, cwd=cwd
        )
    except FileNotFoundError as exc:
        raise CommandError(f"找不到命令：{args[0]}") from exc
    communicate_task = asyncio.create_task(process.communicate())
    try:
        if on_started:
            on_started(process)
        timeout_notified = False
        while not communicate_task.done():
            elapsed_seconds = time.monotonic() - started
            if elapsed_seconds < seconds:
                if on_tick:
                    on_tick(max(0, math_ceil(seconds - elapsed_seconds)))
            elif not stop_on_timeout:
                if not timeout_notified and on_timeout:
                    on_timeout()
                timeout_notified = True
            else:
                break
            await asyncio.sleep(0.2)
        if stop_on_timeout and process.returncode is None:
            _stop_process_group(process, signal.SIGINT)
        if stop_on_timeout:
            stdout_bytes, stderr_bytes = await _finish_after_signal(process, communicate_task, 8)
        else:
            stdout_bytes, stderr_bytes = await communicate_task
    finally:
        if process.returncode is None:
            _stop_process_group(process, signal.SIGTERM)
    result = ProcessResult(args, process.returncode or 0, stdout_bytes.decode(errors="replace"),
                           stderr_bytes.decode(errors="replace"), time.monotonic() - started)
    if result.returncode != 0:
        raise CommandError(f"命令异常退出（{result.returncode}）：{' '.join(args)}", result)
    return result


def math_ceil(value: float) -> int:
    return max(0, int(value) if value.is_integer() else int(value) + 1)


async def check_emcli(on_result=None) -> tuple[bool, str]:
    try:
        result = await run_command(["emcli", "--version"], timeout_seconds=5)
    except CommandError as exc:
        return False, str(exc)
    if on_result:
        on_result(result)
    match = VERSION_RE.search(result.combined_output)
    if result.returncode != 0 or not match:
        return False, "无法从 emcli --version 识别 emcli 版本"
    version = tuple(int(match.group(index)) for index in (1, 2, 3))
    if version < MIN_EMCLI_VERSION:
        return False, f"emcli {'.'.join(map(str, version))} 低于最低版本 1.9.9"
    return True, f"emcli {'.'.join(map(str, version))} 已就绪"


async def install_package_archive(archive: Path) -> list[ProcessResult]:
    """Install driver and CLI from the validated runtime ZIP using apt and pkexec."""
    check = package_archive_check_for_current_platform(archive)
    if not check.usable:
        raise ValueError(check.message)
    assert check.driver_member is not None and check.cli_member is not None
    unpack_dir = Path(tempfile.mkdtemp(prefix="robot-debs-"))
    try:
        with zipfile.ZipFile(archive) as package_zip:
            package_zip.extract(check.driver_member, unpack_dir)
            package_zip.extract(check.cli_member, unpack_dir)
        driver = unpack_dir / check.driver_member
        cli = unpack_dir / check.cli_member
        result = await run_command(["pkexec", "apt", "install", "-y", str(driver), str(cli)], timeout_seconds=180)
        if result.returncode != 0:
            raise CommandError("安装 libencosdriver 与 emcli 失败", result)
        return [result]
    finally:
        shutil.rmtree(unpack_dir, ignore_errors=True)


def prepare_play_file(source: Path, destination: Path, plugin: str, interface: str) -> Path:
    """Materialize the template with the configured plugin and interface."""
    if not source.exists():
        raise ValueError("缺少 play.csv")
    content = source.read_text(encoding="utf-8")
    target_prefix = f"{plugin}:{interface}:"
    # Support the previous Ethercat-only template while shipping the generic
    # plugin placeholder for all new play.csv files.
    content = content.replace("plugin-replace-me:enx-replace-me:", target_prefix)
    content = content.replace("Ethercat:enx-replace-me:", target_prefix)
    destination.write_text(content, encoding="utf-8")
    return destination


def play_duration_seconds(path: Path) -> float:
    """Return the largest timestamp in a play.csv trajectory."""
    if not path.exists():
        raise ValueError("缺少 play.csv")
    duration: float | None = None
    header_found = False
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.reader(file):
            if not row or not row[0].strip() or row[0].lstrip().startswith("#"):
                continue
            if not header_found:
                if row[0].strip().lower() != "time":
                    raise ValueError("play.csv 缺少 time 表头")
                header_found = True
                continue
            try:
                timestamp = float(row[0])
            except ValueError as exc:
                raise ValueError("play.csv 包含无效时间") from exc
            if not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError("play.csv 包含无效时间")
            duration = timestamp if duration is None else max(duration, timestamp)
    if duration is None:
        raise ValueError("play.csv 不包含轨迹数据")
    return duration


async def collect_imu_for_seconds(args: list[str], *, seconds: float, connection_timeout: float, on_report) -> list[ImuReport]:
    """Collect noisy IMU samples for an exact observation window."""
    try:
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True
        )
    except FileNotFoundError as exc:
        raise CommandError(f"找不到命令：{args[0]}") from exc
    chunks: list[str] = []

    async def consume(stream):
        while True:
            block = await stream.read(4096)
            if not block:
                break
            chunks.append(block.decode(errors="replace"))

    consumers = [asyncio.create_task(consume(process.stdout)), asyncio.create_task(consume(process.stderr))]
    started = time.monotonic()
    delivered = 0
    try:
        while process.returncode is None and time.monotonic() - started < seconds:
            reports = parse_imu_reports("".join(chunks))
            if reports:
                if len(reports) > delivered:
                    delivered = len(reports)
                    on_report(reports)
            elif time.monotonic() - started > connection_timeout:
                raise CommandError("2 秒内未获得完整 IMU 数据，请检查 IMU 连接", raw_output="".join(chunks))
            await asyncio.sleep(0.1)
        reports = parse_imu_reports("".join(chunks))
        if reports:
            return reports
        await asyncio.gather(*consumers)
        raise CommandError("IMU 命令提前退出：\n" + "".join(chunks)[-2000:], raw_output="".join(chunks))
    finally:
        _stop_process_group(process, signal.SIGINT)
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            _stop_process_group(process, signal.SIGTERM)
            await process.wait()
        await asyncio.gather(*consumers, return_exceptions=True)
