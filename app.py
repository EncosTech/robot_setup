"""Local NiceGUI wizard for robot commissioning.

Run with:  uv run python app.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
import shutil
import socket
import tempfile

from nicegui import Client, app, ui

from robot_core import (
    ImuConfig,
    Joint,
    ParseError,
    PLAY_LOG_CONTINUOUS_LIMIT_MS,
    PLAY_LOG_MEAN_ERROR_LIMIT_RAD,
    PLAY_LOG_TEMPERATURE_WARN_C,
    PLAY_LOG_TRACKING_LIMIT_RAD,
    PlayLogResult,
    StressResult,
    StressStat,
    analyze_packet_loss,
    analyze_play_log_directory,
    compare_topology,
    evaluate_imu_reports,
    format_joint_button_label,
    format_scan_choice_labels,
    is_ethercat_connection_missing,
    parse_interface_output,
    parse_position_output,
    parse_scan_output,
    parse_stress_final_result,
    read_hardware_csv,
)
from robot_runtime import (
    CommandError,
    adapter_target,
    check_emcli,
    config_directory_sha1,
    interface_link_is_up,
    imu_target,
    interrupt_process,
    install_package_archive,
    load_config,
    load_progress,
    load_system_settings,
    package_archive_check_for_current_platform,
    collect_imu_for_seconds,
    motor_target,
    play_duration_seconds,
    prepare_play_file,
    raw_motor_target,
    remove_progress,
    run_command,
    run_timed_command,
    runtime_package_archive,
    save_config,
    save_progress,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "robot_config.yaml"
SYSTEM_PATH = ROOT / "system.yaml"
PROGRESS_PATH = ROOT / "robot_progress.json"
ROBOT_CONFIG_ROOT = ROOT / "config"
FAVICON_PATH = ROOT / "assets" / "robot_setup.svg"
CONNECT_IMAGE_PATH = ROOT / "assets" / "connect_robot.png"
MAX_LOG_ENTRIES = 10
APP_VERSION = "0.0.10"
CONFIGURATION_STEPS = (
    ("connect", "1 连接机器人"),
    ("scan", "2 扫描"),
    ("stress", "3 压测"),
    ("calibration", "4 零点"),
    ("imu", "5 IMU"),
    ("motion", "6 运动测试"),
)


class RobotWizard:
    def __init__(self) -> None:
        self.config = load_config(CONFIG_PATH)
        self.system = load_system_settings(SYSTEM_PATH)
        self.stage = "home"
        self.critical_error = ""
        self.before_interfaces: set[str] = set()
        self.error = ""
        self.notice = ""
        self.raw_output = ""
        self.package_install_ready = False
        self.completed_stage: str | None = None
        self.motor_loss_summaries: dict[str, list[tuple[str, float]]] = {}
        self.motion_log_results: dict[str, PlayLogResult] = {}
        self.connection_monitoring = False
        self.connection_monitor_stop = False
        self.connection_status_label: ui.label | None = None
        self.id_status = ""
        self.id_status_label: ui.label | None = None
        # Keep the choice as state and render the dialog during refresh.  A
        # dialog appended directly from an async action is otherwise removed
        # by _action's final refresh before the browser can see it.
        self.pending_id_choice = None
        self.log_entries: list[str] = []
        self._logged_stage = ""
        self.busy = False
        self.content: ui.column | None = None
        self.calibration_index = 0
        self.calibration_phase = "pose"
        self.limit_monitor_stop = False
        self.limit_low_hits = 0
        self.limit_high_hits = 0
        self.limit_low_ok = False
        self.limit_high_ok = False
        self.position_label: ui.label | None = None
        self.low_limit_label: ui.label | None = None
        self.high_limit_label: ui.label | None = None
        self.imu_label: ui.label | None = None
        self.countdown_label: ui.label | None = None
        self.test_stop_label: ui.label | None = None
        self.active_test_process: asyncio.subprocess.Process | None = None
        self.active_command_process: asyncio.subprocess.Process | None = None
        self.test_aborted = False
        self.network_monitoring = False
        self.network_fault = False
        self.network_fault_dialog = None
        self.admin_dialog_open = False
        self.joints: list[Joint] = []
        self.imus: list[ImuConfig] = []
        self.imu_index = 0
        self.imu_average_label: ui.label | None = None
        self.imu_range_label: ui.label | None = None
        self._load_selected_robot_files()
        self.pending_progress_restore = self._load_recoverable_progress()

    @staticmethod
    def _robot_types() -> list[str]:
        if not ROBOT_CONFIG_ROOT.is_dir():
            return []
        return sorted(
            directory.name
            for directory in ROBOT_CONFIG_ROOT.iterdir()
            if directory.is_dir()
        )

    def _robot_directory(self) -> Path | None:
        robot_type = str(self.config.get("robot_type", "")).strip()
        if not robot_type or robot_type not in self._robot_types():
            return None
        return ROBOT_CONFIG_ROOT / robot_type

    def _progress_payload(self) -> dict | None:
        directory = self._robot_directory()
        if directory is None or self.stage not in {stage for stage, _ in CONFIGURATION_STEPS}:
            return None
        return {
            "robot_type": self.config.get("robot_type", ""),
            "config_sha1": config_directory_sha1(directory),
            "stage": self.stage,
            "completed_stage": self.completed_stage,
            "calibration_index": self.calibration_index,
            "calibration_phase": self.calibration_phase,
            "imu_index": self.imu_index,
        }

    def _save_progress(self) -> None:
        try:
            payload = self._progress_payload()
            if payload is not None:
                save_progress(PROGRESS_PATH, payload)
        except (OSError, ValueError) as exc:
            self._log(f"写入恢复进度失败：{exc}")

    def _load_recoverable_progress(self) -> dict | None:
        progress = load_progress(PROGRESS_PATH)
        directory = self._robot_directory()
        valid_stages = {stage for stage, _ in CONFIGURATION_STEPS}
        if (
            not progress
            or directory is None
            or progress.get("stage") not in valid_stages
            or progress.get("robot_type") != self.config.get("robot_type")
        ):
            return None
        try:
            is_current = progress.get("config_sha1") == config_directory_sha1(directory)
        except (OSError, ValueError):
            is_current = False
        if is_current:
            return progress
        remove_progress(PROGRESS_PATH)
        return None

    def _render_progress_restore_dialog(self) -> None:
        dialog = ui.dialog().props("persistent")
        with dialog, ui.card():
            ui.label("发现可恢复的上次配置进度").classes("text-lg font-bold")
            ui.label("当前机器人配置文件未发生变化，是否恢复上次进度？")
            with ui.row().classes("gap-3"):
                ui.button("恢复上次进度", on_click=lambda: self._restore_progress(dialog), color="primary")
                ui.button("放弃并重新开始", on_click=lambda: self._discard_progress(dialog), color="secondary")
        dialog.open()

    def _restore_progress(self, dialog) -> None:
        progress = self.pending_progress_restore
        if progress is None:
            dialog.close()
            return
        dialog.close()
        self.pending_progress_restore = None
        self.stage = progress["stage"]
        self.completed_stage = progress.get("completed_stage") if progress.get("completed_stage") == self.stage else None
        self.calibration_index = max(0, min(int(progress.get("calibration_index", 0)), len(self.joints)))
        self.calibration_phase = progress.get("calibration_phase") if progress.get("calibration_phase") in {"pose", "limits"} else "pose"
        self.imu_index = max(0, min(int(progress.get("imu_index", 0)), len(self.imus)))
        self.notice = "已恢复上次配置进度。"
        self.refresh()

    def _discard_progress(self, dialog) -> None:
        dialog.close()
        self.pending_progress_restore = None
        remove_progress(PROGRESS_PATH)
        self.refresh()

    def _on_key(self, event) -> None:
        if (event.action.keydown and not event.action.repeat and event.modifiers.ctrl and event.key == "b"
                and not self.network_fault and not self.admin_dialog_open):
            self._open_admin_jump()

    def _open_admin_jump(self) -> None:
        """直接打开进度跳转，无需密码验证。"""
        if self.admin_dialog_open:
            return
        self._show_admin_jump_dialog()

    def _show_admin_jump_dialog(self) -> None:
        self.admin_dialog_open = True
        dialog = ui.dialog().props("persistent")
        stage_options = {stage: label for stage, label in CONFIGURATION_STEPS if self._step_enabled(stage)}
        state: dict[str, object] = {"stage": self.stage if self.stage in stage_options else next(iter(stage_options), "connect")}
        with dialog, ui.card().classes("w-[520px] max-w-full"):
            ui.label("进度跳转").classes("text-lg font-bold")
            stage_select = ui.select(stage_options, label="大步骤", value=state["stage"])
            details = ui.column().classes("w-full gap-3")
            def render_details() -> None:
                details.clear()
                stage = str(state["stage"])
                with details:
                    if stage == "calibration":
                        state["calibration_index"] = min(int(state.get("calibration_index", 0)), max(0, len(self.joints) - 1))
                        joint_options = {index: joint.label for index, joint in enumerate(self.joints)}
                        joint_select = ui.select(joint_options, label="关节", value=state["calibration_index"])
                        def select_joint(event) -> None:
                            state["calibration_index"] = int(event.value)
                            render_details()
                        joint_select.on_value_change(select_joint)
                        joint = self.joints[int(state["calibration_index"])] if self.joints else None
                        phase_options = {"pose": "零位标定"}
                        if joint and joint.requires_limit_check:
                            phase_options["limits"] = "限位检测"
                        state["calibration_phase"] = state.get("calibration_phase", "pose") if state.get("calibration_phase") in phase_options else "pose"
                        phase_select = ui.select(phase_options, label="子步骤", value=state["calibration_phase"])
                        phase_select.on_value_change(lambda event: state.__setitem__("calibration_phase", event.value))
                    elif stage == "imu":
                        state["imu_index"] = min(int(state.get("imu_index", 0)), max(0, len(self.imus) - 1))
                        imu_options = {index: imu.label for index, imu in enumerate(self.imus)}
                        imu_select = ui.select(imu_options, label="IMU", value=state["imu_index"])
                        imu_select.on_value_change(lambda event: state.__setitem__("imu_index", int(event.value)))
            def select_stage(event) -> None:
                state["stage"] = event.value
                render_details()
            stage_select.on_value_change(select_stage)
            render_details()
            ui.button("跳转", on_click=lambda: self._apply_admin_jump(dialog, state), color="primary")
            ui.button("取消", on_click=lambda: (dialog.close(), setattr(self, "admin_dialog_open", False)),
                      color="secondary").props("flat")
        dialog.open()

    def _apply_admin_jump(self, dialog, state: dict[str, object]) -> None:
        self.limit_monitor_stop = True
        self.connection_monitor_stop = True
        self._clear_page_feedback()
        self.stage = str(state["stage"])
        if self.stage == "calibration":
            self.calibration_index = int(state.get("calibration_index", 0))
            self.calibration_phase = str(state.get("calibration_phase", "pose"))
            self.limit_low_hits = self.limit_high_hits = 0
            self.limit_low_ok = self.limit_high_ok = False
        elif self.stage == "imu":
            self.imu_index = int(state.get("imu_index", 0))
        self._save_progress()
        dialog.close()
        self.admin_dialog_open = False
        self.refresh()

    @property
    def joints_path(self) -> Path | None:
        directory = self._robot_directory()
        return directory / "robot_joints.csv" if directory else None

    @property
    def play_path(self) -> Path | None:
        directory = self._robot_directory()
        return directory / "play.csv" if directory else None

    def _load_selected_robot_files(self) -> None:
        self.joints = []
        self.imus = []
        self.critical_error = ""
        directory = self._robot_directory()
        if directory is None:
            return
        required_files = {
            "robot_joints.csv": directory / "robot_joints.csv",
            "play.csv": directory / "play.csv",
        }
        missing_files = [
            filename for filename, path in required_files.items()
            if not path.is_file()
        ]
        if missing_files:
            self.critical_error = (
                f"缺少关键文件：{'、'.join(missing_files)}。"
                f"请确认文件位于机器人配置目录：{directory}。"
            )
            self.error = self.critical_error
            return
        try:
            self.joints, self.imus = read_hardware_csv(required_files["robot_joints.csv"])
        except (OSError, ValueError) as exc:
            self.critical_error = "无法读取关键文件 robot_joints.csv。请检查所选机器人配置。"
            self.error = self.critical_error

    def refresh(self) -> None:
        assert self.content is not None
        self.content.clear()
        with self.content:
            if self.critical_error:
                with ui.card().classes("w-full bg-red-1 text-red-10"):
                    ui.label("无法启动机器人配置程序").classes("text-xl font-bold")
                    ui.label(self.critical_error)
                return
            if self.stage != self._logged_stage:
                self._log(f"进入步骤：{self.stage}")
                self._logged_stage = self.stage
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"机器人配置向导 v{APP_VERSION}").classes("text-2xl font-bold")
                with ui.row().classes("gap-2"):
                    ui.button("查看日志", on_click=self._show_logs, color="secondary").props("outline")
                    ui.button("跳转步骤", on_click=self._open_admin_jump, color="secondary").props("outline")
                    ui.button("下载日志", on_click=self._show_download, color="secondary").props("outline")
            if self.stage != "home":
                ui.button("返回功能选择", on_click=self._go_home, color="secondary").props("flat")
            if self.stage in {"connect", "scan", "stress", "calibration", "imu", "motion", "done"}:
                self._stepper()
            if self.error:
                with ui.card().classes("w-full bg-red-1 text-red-10"):
                    ui.label(self.error).classes("whitespace-pre-line")
            if self.notice:
                with ui.card().classes("w-full bg-green-1 text-green-10"):
                    ui.label(self.notice)
            if self.stage == "home":
                self._render_home()
            elif self.stage == "preflight":
                self._render_preflight()
            elif self.stage == "robot_type":
                self._render_robot_type()
            elif self.stage == "unplug":
                self._render_unplug()
            elif self.stage == "plug":
                self._render_plug()
            elif self.stage == "connect":
                self._render_connect()
            elif self.stage == "scan":
                self._render_scan()
            elif self.stage == "stress":
                self._render_stress()
            elif self.stage == "calibration":
                self._render_calibration()
            elif self.stage == "imu":
                self._render_imu()
            elif self.stage == "motion":
                self._render_motion()
            elif self.stage == "ids":
                self._render_ids()
            else:
                self._render_done()
            if self._network_guard_active():
                ui.timer(0.1, self._start_network_monitor, once=True)
            if self.network_fault:
                self._render_network_fault_dialog()
            if self.pending_progress_restore is not None:
                self._render_progress_restore_dialog()
    def _log(self, message: str, output: str = "") -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{stamp}] {message}"
        if output:
            entry += "\n--- emcli 原样返回 ---\n" + output + "\n--- 返回结束 ---"
        self.log_entries.append(entry)
        del self.log_entries[:-MAX_LOG_ENTRIES]

    def _log_text(self) -> str:
        return "\n\n".join(self.log_entries) or "尚无日志。\n"

    def _show_logs(self) -> None:
        dialog = ui.dialog()
        with dialog, ui.card().classes("w-[900px] max-w-full"):
            ui.label("运行日志").classes("text-lg font-bold")
            ui.code(self._log_text(), language="text").classes("w-full max-h-96 overflow-auto")
            ui.button("关闭", on_click=dialog.close)
        dialog.open()

    def _show_download(self) -> None:
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label("下载日志").classes("text-lg font-bold")
            ui.label("选择浏览器保存位置后，将保存为 .log 文件。")
            filename = ui.input("文件名", value=f"robot-{datetime.now():%Y%m%d-%H%M%S}.log")
            def download() -> None:
                name = filename.value if str(filename.value).endswith(".log") else f"{filename.value}.log"
                ui.download(self._log_text().encode("utf-8"), filename=name)
                dialog.close()
            ui.button("选择位置并下载", on_click=download, color="primary")
        dialog.open()

    def _show_export_dialog(self) -> None:
        """导出当前机器人的 play.csv（先替换插件/网卡占位文本）到所选目录。"""
        source = self.play_path
        if source is None or not source.is_file():
            ui.notify("未找到可导出的 play.csv，请先完成机器人类型选择。", type="warning")
            return
        interface = str(self.config.get("interface", "")).strip()
        if not interface:
            ui.notify("未配置网卡 interface，无法导出 play.csv。", type="warning")
            return
        plugin = str(self.config.get("plugin", "Ethercat")).strip() or "Ethercat"
        dialog = ui.dialog()
        with dialog, ui.card().classes("w-[560px] max-w-full"):
            ui.label("运动数据导出").classes("text-lg font-bold")
            ui.label(f"来源模板：{source}").classes("text-xs text-grey")
            ui.label("导出时自动将文件中的插件/网卡占位文本替换为当前配置（两个占位文本）。").classes("text-sm text-grey")
            with ui.row().classes("w-full items-center gap-2"):
                directory_input = ui.input("导出目录", value=str(Path.home())).classes("flex-1")
                ui.button("进入", on_click=lambda: refresh_listing(), color="secondary").props("flat")
            listing = ui.column().classes("w-full gap-1 max-h-72 overflow-auto")
            filename_input = ui.input("文件名", value="play.csv").classes("w-full")
            def refresh_listing() -> None:
                listing.clear()
                directory = Path(str(directory_input.value)).expanduser()
                with listing:
                    if not directory.is_dir():
                        ui.label(f"目录不存在：{directory}").classes("text-negative")
                        return
                    ui.button("⬆ 上级目录",
                              on_click=lambda: enter_directory(directory.parent), color="secondary").props("flat dense")
                    for child in sorted((p for p in directory.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
                        ui.button(child.name,
                                  on_click=lambda path=child: enter_directory(path),
                                  color="secondary").props("flat dense")
                    ui.label(str(directory)).classes("text-xs text-grey")
            def enter_directory(directory: Path) -> None:
                directory_input.value = str(directory)
                refresh_listing()
            refresh_listing()
            with ui.row().classes("gap-3"):
                def export_now() -> None:
                    directory = Path(str(directory_input.value)).expanduser()
                    if not directory.is_dir():
                        ui.notify(f"目录不存在：{directory}", type="negative")
                        return
                    filename = str(filename_input.value or "play.csv").strip()
                    if not filename.lower().endswith(".csv"):
                        filename += ".csv"
                    destination = directory / filename
                    try:
                        prepare_play_file(source, destination, plugin, interface)
                    except (OSError, ValueError) as exc:
                        ui.notify(f"导出失败：{exc}", type="negative")
                        return
                    ui.notify(f"已导出运动数据：{destination}")
                    dialog.close()
                ui.button("导出", on_click=export_now, color="primary")
                ui.button("取消", on_click=dialog.close, color="secondary").props("flat")
        dialog.open()

    async def _command(self, args: list[str], **kwargs):
        self._log("执行命令：" + " ".join(args))
        def started(process: asyncio.subprocess.Process) -> None:
            self.active_command_process = process
        try:
            result = await run_command(args, on_started=started, **kwargs)
        except CommandError as exc:
            result = getattr(exc, "result", None)
            self._log("命令异常：" + " ".join(args), result.combined_output if result else getattr(exc, "raw_output", ""))
            raise
        finally:
            self.active_command_process = None
        self._log("命令完成：" + " ".join(args), result.combined_output)
        return result

    async def _timed_command(self, args: list[str], seconds: float, **kwargs):
        self._log(f"执行命令（计划 {seconds} 秒）：" + " ".join(args))
        callback = kwargs.pop("on_started", None)
        def started(process: asyncio.subprocess.Process) -> None:
            self.active_command_process = process
            if callback:
                callback(process)
        try:
            result = await run_timed_command(args, seconds, on_started=started, **kwargs)
        except CommandError as exc:
            result = getattr(exc, "result", None)
            self._log("命令异常：" + " ".join(args), result.combined_output if result else "")
            raise
        finally:
            self.active_command_process = None
        self._log("命令完成：" + " ".join(args), result.combined_output)
        return result

    def _step_enabled(self, stage: str) -> bool:
        if not self.system[f"enable_{stage}"]:
            return False
        if stage in {"scan", "stress", "calibration", "motion"}:
            return bool(self.joints)
        if stage == "imu":
            return bool(self.imus)
        return True

    def _network_guard_active(self) -> bool:
        return self.stage in {"scan", "stress", "calibration", "imu", "motion"}

    def _start_network_monitor(self) -> None:
        if self.network_monitoring or not self._network_guard_active():
            return
        self.network_monitoring = True
        asyncio.create_task(self._monitor_network())

    async def _monitor_network(self) -> None:
        try:
            while self._network_guard_active():
                interface = str(self.config.get("interface", "")).strip()
                healthy = bool(interface) and await interface_link_is_up(interface)
                if healthy:
                    self._recover_network_if_needed()
                else:
                    self._handle_network_failure()
                await asyncio.sleep(1)
        finally:
            self.network_monitoring = False

    def _handle_network_failure(self) -> None:
        if self.network_fault:
            return
        self.network_fault = True
        self.limit_monitor_stop = True
        process = self.active_command_process
        if process is not None and process.returncode is None:
            self.test_aborted = self.active_test_process is process
            interrupt_process(process)
            self._log("检测到网卡链路异常：已向当前 emcli 发送 Ctrl+C")
        self._clear_page_feedback()
        self.busy = False
        if self.stage == "calibration":
            self.calibration_phase = "pose"
            self.limit_low_hits = self.limit_high_hits = 0
            self.limit_low_ok = self.limit_high_ok = False
        self._save_progress()
        self.refresh()

    def _recover_network_if_needed(self) -> None:
        if not self.network_fault:
            return
        self.network_fault = False
        if self.network_fault_dialog is not None:
            self.network_fault_dialog.close()
            self.network_fault_dialog = None
        self.notice = "网卡链路已恢复，请重新开始当前步骤。"
        self.refresh()

    def _render_network_fault_dialog(self) -> None:
        if self.network_fault_dialog is not None:
            return
        dialog = ui.dialog().props("persistent")
        with dialog, ui.card().classes("w-[520px] max-w-full"):
            ui.label("机器人网卡连接异常").classes("text-xl font-bold text-negative")
            ui.label("请检查机器人网卡是否启用，以及 EtherCAT 网线与机器人电源连接。链路恢复后将自动继续。")
        self.network_fault_dialog = dialog
        dialog.open()

    def _first_configuration_stage(self) -> str:
        return next((stage for stage, _ in CONFIGURATION_STEPS if self._step_enabled(stage)), "done")

    def _advance_from(self, stage: str) -> None:
        stages = [name for name, _ in CONFIGURATION_STEPS]
        start = stages.index(stage) + 1
        self.stage = next((name for name in stages[start:] if self._step_enabled(name)), "done")
        if self.stage == "done":
            remove_progress(PROGRESS_PATH)
        else:
            self._save_progress()

    def _complete_stage(self, stage: str, message: str) -> None:
        self.completed_stage = stage
        self.notice = f"{message}请确认后进入下一步。"
        stages = [name for name, _ in CONFIGURATION_STEPS]
        has_next_stage = any(self._step_enabled(name) for name in stages[stages.index(stage) + 1:])
        if has_next_stage:
            self._save_progress()
        else:
            remove_progress(PROGRESS_PATH)

    def _motor_loss_summary(self, stress) -> list[tuple[str, float]]:
        labels = {joint.key: joint.label for joint in self.joints}
        rows = []
        for (slave, bus, motor_id), stat in sorted(
            stress.motors.items(), key=lambda item: ((0 if item[0][0] is None else item[0][0]), item[0][1], item[0][2])
        ):
            key = (0 if slave is None else slave, bus, motor_id)
            label = labels.get(key, f"关节 {bus}:{motor_id}")
            rows.append((label, stat.drop_rate))
        return rows

    def _render_next_step_button(self, stage: str) -> bool:
        if self.completed_stage != stage:
            return False
        self._render_motor_loss_summary(stage)
        ui.label("本步骤已完成。")
        self._buttons(("进入下一步", lambda: self._continue_from(stage), "primary"))
        return True

    def _render_motor_loss_summary(self, stage: str) -> None:
        report = self.motion_log_results.get(stage)
        if report:
            ui.label("各关节检测结果（来自落盘日志）").classes("text-lg font-bold")
            with ui.card().classes("w-full"):
                for joint in self.joints:
                    stat = report.motors.get(joint.key)
                    if stat is None:
                        continue
                    ui.label(f"{stat.label}：丢包率 {stat.drop_rate:.4%}，平均误差 "
                             f"{stat.mean_error_rad:.4f} rad，最大误差 {stat.max_error_rad:.4f} rad，"
                             f"连续超差最长 {stat.longest_over_limit_ms} ms，"
                             f"最高电机温度 {stat.max_motor_temp_c:.0f}°C")
            return
        summary = self.motor_loss_summaries.get(stage)
        if summary:
            ui.label("各关节丢包率").classes("text-lg font-bold")
            with ui.card().classes("w-full"):
                for label, drop_rate in summary:
                    ui.label(f"{label}：{drop_rate:.4%}")

    def _continue_from(self, stage: str) -> None:
        if self.busy or self.completed_stage != stage:
            return
        self._clear_page_feedback()
        self._advance_from(stage)
        self.refresh()

    def _clear_page_feedback(self) -> None:
        """Clear transient status whenever navigation leaves the current page."""
        self.error = ""
        self.notice = ""
        self.raw_output = ""
        self.completed_stage = None
        self.motor_loss_summaries.clear()
        self.motion_log_results.clear()
        self.package_install_ready = False
        self.id_status = ""
        self.pending_id_choice = None
        self.connection_status_label = None
        self.id_status_label = None
        self.position_label = None
        self.low_limit_label = None
        self.high_limit_label = None
        self.imu_label = None
        self.imu_average_label = None
        self.imu_range_label = None
        self.countdown_label = None
        self.test_stop_label = None

    def _stepper(self) -> None:
        names = [(stage, label) for stage, label in CONFIGURATION_STEPS if self._step_enabled(stage)]
        names.append(("done", "完成"))
        current = next((index for index, (key, _) in enumerate(names) if key == self.stage), 0)
        with ui.row().classes("items-center gap-2"):
            for index, (_, label) in enumerate(names):
                ui.badge(label, color="primary" if index == current else "grey")

    def _go_home(self) -> None:
        if not self.busy:
            self.connection_monitor_stop = True
            self._clear_page_feedback()
            self.stage = "home"
            self.refresh()

    def _render_home(self) -> None:
        ui.label("请选择要进入的界面。").classes("text-lg")
        configured = bool(self.config.get("interface"))
        items = [("安装与初始化", self._enter_install, "primary")]
        if configured:
            items.extend((("电机 ID 设置", self._enter_ids, "primary"),
                          ("机器人整体配置", self._enter_robot, "primary")))
            if self.play_path and self.play_path.exists():
                items.append(("运动数据导出", self._show_export_dialog, "primary"))
        self._buttons(*items)

    def _enter_install(self) -> None:
        self._clear_page_feedback()
        self.stage = "robot_type"
        self.refresh()

    def _enter_ids(self) -> None:
        self._clear_page_feedback()
        self.stage = "ids" if self.config.get("interface") else "preflight"
        self.refresh()

    def _enter_robot(self) -> None:
        self._clear_page_feedback()
        self.stage = self._first_configuration_stage() if self.config.get("interface") else "preflight"
        self._save_progress()
        self.refresh()

    def _action(self, operation):
        async def wrapped():
            if self.busy:
                return
            self.busy = True
            self.error = ""
            self.notice = ""
            self.raw_output = ""
            self.refresh()
            try:
                await operation()
            except (CommandError, ParseError, ValueError, OSError) as exc:
                if self.network_fault:
                    return
                self.error = str(exc)
                result = getattr(exc, "result", None)
                if result:
                    self.raw_output = result.combined_output
                elif getattr(exc, "raw_output", ""):
                    self.raw_output = exc.raw_output
            except Exception as exc:  # keep unexpected driver/UI failures visible, never silently pass
                self.error = f"未预期错误：{type(exc).__name__}: {exc}"
            finally:
                self.busy = False
                self.refresh()
        return wrapped

    def _buttons(self, *items: tuple[str, object, str]) -> None:
        with ui.row().classes("gap-3"):
            for label, callback, color in items:
                ui.button(label, on_click=callback, color=color).props("disable" if self.busy else "")

    def _render_preflight(self) -> None:
        ui.label("首次使用：检查 emcli 是否存在且版本不低于 1.9.9。").classes("text-lg")
        self._buttons(("检查 emcli", self._action(self._check_preflight), "primary"))
        if self.package_install_ready:
            self._buttons(("安装 packages.zip", self._action(self._install_runtime_packages), "primary"))

    def _render_robot_type(self) -> None:
        ui.label("请选择机器人类别。该选择会保存到本机 robot_config.yaml。 ").classes("text-lg")
        robot_types = self._robot_types()
        if not robot_types:
            self.error = "未找到任何机器人配置目录。请检查 config/ 目录。"
            return
        self._buttons(*[
            (robot_type, lambda value=robot_type: self._select_robot_type(value), "primary")
            for robot_type in robot_types
        ])

    def _select_robot_type(self, robot_type: str) -> None:
        if robot_type not in self._robot_types():
            self.error = "所选机器人类别不存在，请重新选择。"
            self.refresh()
            return
        self._clear_page_feedback()
        remove_progress(PROGRESS_PATH)
        self.config["robot_type"] = robot_type
        save_config(CONFIG_PATH, self.config)
        self.error = ""
        self.notice = f"已选择机器人类别：{robot_type}"
        self._load_selected_robot_files()
        self.stage = "preflight"
        self.refresh()

    async def _check_preflight(self) -> None:
        self.package_install_ready = False
        self._log("执行命令：emcli --version")
        ok, message = await check_emcli(lambda result: self._log("命令完成：emcli --version", result.combined_output))
        if not ok:
            archive = runtime_package_archive(ROOT)
            check = package_archive_check_for_current_platform(archive)
            if not check.usable:
                self.error = f"{message}\n运行目录中的 packages.zip 不可用：{check.message}"
                return
            self.package_install_ready = True
            self.error = message
            return
        self.stage = "unplug"
        self.notice = message

    def _render_unplug(self) -> None:
        ui.label("请拔掉连接机器人 EtherCAT 网口的网线，确认机器人不与本机相连后点击扫描。").classes("text-lg")
        self._buttons(("已拔线，扫描网卡", self._action(self._scan_without_robot), "primary"))

    async def _install_runtime_packages(self) -> None:
        archive = runtime_package_archive(ROOT)
        check = package_archive_check_for_current_platform(archive)
        if not check.usable:
            self.package_install_ready = False
            raise ValueError(f"运行目录中的 packages.zip 不可用：{check.message}")
        results = await install_package_archive(archive)
        self.raw_output = "\n".join(result.combined_output for result in results)
        self._log("执行命令：emcli --version")
        ok, message = await check_emcli(lambda result: self._log("命令完成：emcli --version", result.combined_output))
        if not ok:
            raise ValueError(message)
        self.package_install_ready = False
        self.stage = "unplug"
        self.notice = message

    async def _scan_without_robot(self) -> None:
        result = await self._command(["emcli", "scan", "Ethercat"], timeout_seconds=10)
        self.raw_output = result.combined_output
        if result.returncode != 0:
            raise CommandError("拔线扫描失败", result)
        self.before_interfaces = set(parse_interface_output(result.combined_output, self._local_interfaces()))
        self.stage = "plug"

    def _render_plug(self) -> None:
        ui.label("现在插入 EtherCAT 网线并开启机器人电源，待系统稳定后点击扫描。").classes("text-lg")
        self._buttons(("已接线开机，扫描网卡", self._action(self._scan_with_robot), "primary"),
                      ("返回拔线步骤", self._back_to_unplug, "secondary"))

    def _back_to_unplug(self) -> None:
        self._clear_page_feedback()
        self.before_interfaces.clear()
        self.stage = "unplug"
        self.refresh()

    async def _scan_with_robot(self) -> None:
        result = await self._command(["emcli", "scan", "Ethercat"], timeout_seconds=10)
        self.raw_output = result.combined_output
        if result.returncode != 0:
            raise CommandError("接线扫描失败", result)
        after = set(parse_interface_output(result.combined_output, self._local_interfaces()))
        added = sorted(after - self.before_interfaces)
        if len(added) != 1:
            self.stage = "unplug"
            if not added:
                raise ValueError("未检测到新增网卡。请重新拔线并确认插拔的是机器人网线。")
            raise ValueError("检测到多个新增网卡：" + ", ".join(added) + "。请移除其他新设备后重试。")
        self.config["interface"] = added[0]
        save_config(CONFIG_PATH, self.config)
        self.stage = "home"
        self.notice = f"已保存 EtherCAT 网卡：{added[0]}。初始化完成，请选择要使用的功能。"

    @staticmethod
    def _local_interfaces() -> set[str]:
        return {name for _, name in socket.if_nameindex()}

    def _render_connect(self) -> None:
        if CONNECT_IMAGE_PATH.is_file():
            ui.image(str(CONNECT_IMAGE_PATH)).classes("max-w-lg")
        ui.label("请连接机器人并开机，程序会自动检查连接状态。 ").classes("text-lg")
        if self._render_next_step_button("connect"):
            return
        self.connection_status_label = ui.label("正在检查机器人连接……").classes("text-primary")
        ui.timer(0.1, self._start_connection_monitor, once=True)

    def _start_connection_monitor(self) -> None:
        self.connection_monitor_stop = False
        if self.connection_monitoring:
            return
        self.connection_monitoring = True
        asyncio.create_task(self._monitor_robot_connection())

    async def _monitor_robot_connection(self) -> None:
        expected_interface = str(self.config.get("interface", "")).strip()
        plugin = str(self.config.get("plugin", "Ethercat")).strip() or "Ethercat"
        try:
            while self.stage == "connect" and not self.connection_monitor_stop:
                connected = False
                try:
                    result = await self._command(["emcli", "scan", plugin], timeout_seconds=5)
                    if result.returncode == 0:
                        found = parse_interface_output(result.combined_output, self._local_interfaces())
                        connected = expected_interface in found
                except (CommandError, OSError):
                    connected = False
                if connected:
                    self._complete_stage("connect", "已检测到机器人连接。")
                    self.connection_monitoring = False
                    self.refresh()
                    return
                if self.connection_status_label:
                    self.connection_status_label.set_text("请连接机器人并开机。")
                await asyncio.sleep(1)
        finally:
            self.connection_monitoring = False

    def _render_scan(self) -> None:
        ui.label("检查机器人各关节的连接状态。").classes("text-lg")
        if self._render_next_step_button("scan"):
            return
        if self.joints:
            ui.label(f"待检查关节：{len(self.joints)} 个。")
        self._buttons(("扫描关节", self._action(self._run_scan), "primary"))

    def _render_ids(self) -> None:
        ui.label("电机 ID 设置：点击目标关节后扫描全部总线。连接多个电机时需选择当前电机 ID。").classes("text-lg")
        status_color = "text-positive" if self.id_status.startswith("设置成功") else (
            "text-negative" if self.id_status.startswith("设置失败") else "text-primary"
        )
        self.id_status_label = ui.label(self.id_status or "等待选择关节").classes(status_color)
        by_bus: dict[tuple[int, int], list[Joint]] = {}
        for joint in self.joints:
            by_bus.setdefault((joint.slave, joint.bus), []).append(joint)
        for (slave, bus), joints in by_bus.items():
            with ui.row().classes("items-center gap-2"):
                ui.label(f"Bus {bus}").classes("font-bold w-20")
                for joint in joints:
                    ui.button(
                        format_joint_button_label(joint),
                        on_click=self._action(lambda item=joint: self._scan_for_id(item)),
                        color="primary",
                    ).props("disable" if self.busy else "")
        if self.pending_id_choice is not None:
            desired, rows = self.pending_id_choice
            self._render_id_choice_dialog(desired, rows)

    async def _scan_for_id(self, desired: Joint) -> None:
        self.pending_id_choice = None
        self.id_status = f"正在扫描全部总线：{desired.label}"
        if self.id_status_label:
            self.id_status_label.set_text(self.id_status)
        result = await self._command(["emcli", "scan", adapter_target(self.config)], timeout_seconds=15)
        self.raw_output = result.combined_output
        if result.returncode != 0:
            raise CommandError("电机扫描失败", result)
        rows = parse_scan_output(result.combined_output)
        if len(rows) == 1:
            await self._assign_motor_id(rows[0], desired)
            return
        self.id_status = f"扫描到 {len(rows)} 个电机，请选择需要设置的电机。"
        self.pending_id_choice = desired, rows

    def _render_id_choice_dialog(self, desired: Joint, rows) -> None:
        """Present a durable selection dialog after the scan action refreshes."""
        dialog = ui.dialog()
        with dialog, ui.card():
            ui.label("由于连接多个电机，请选择需要设置的电机")
            with ui.row().classes("gap-2"):
                for row, choice_label in zip(rows, format_scan_choice_labels(rows)):
                    async def choose(item=row) -> None:
                        dialog.close()
                        self.pending_id_choice = None
                        await self._action(lambda: self._assign_motor_id(item, desired))()
                    ui.button(choice_label, on_click=choose, color="primary")
            ui.button("取消", on_click=self._dismiss_id_choice, color="secondary").props("flat")
        dialog.open()

    def _dismiss_id_choice(self) -> None:
        self.pending_id_choice = None
        self.id_status = "已取消电机选择。"
        self.refresh()

    async def _assign_motor_id(self, row, desired: Joint) -> None:
        self.pending_id_choice = None
        self._log(f"处理电机 ID：目标 slave={desired.slave} bus={desired.bus} id={desired.motor_id}")
        target = raw_motor_target(self.config, row.raw_slave, row.bus, row.motor_id)
        canfd = await self._command(["emcli", "config", target, "comm", "set", "canfd"], timeout_seconds=8)
        self.raw_output = canfd.combined_output
        if canfd.returncode != 0:
            self.id_status = f"设置失败：CAN FD 设置失败（目标 ID {desired.motor_id}）"
            raise CommandError(self.id_status, canfd)
        set_id = await self._command(["emcli", "config", target, "id", "set", str(desired.motor_id)], timeout_seconds=8)
        self.raw_output += "\n" + set_id.combined_output
        if set_id.returncode != 0:
            self.id_status = f"设置失败：电机 ID 写入失败（目标 ID {desired.motor_id}）"
            raise CommandError(self.id_status, set_id)
        self.id_status = f"设置成功：ID {desired.motor_id} 已设置，请重启电机。"
        self.notice = self.id_status

    async def _run_scan(self) -> None:
        if not self.joints:
            raise ValueError("请先修复 robot_joints.csv")
        result = await self._command(["emcli", "scan", adapter_target(self.config)], timeout_seconds=15)
        self.raw_output = result.combined_output
        if result.returncode != 0:
            if is_ethercat_connection_missing(result.combined_output):
                raise ValueError("未检测到机器人连接。请检查 EtherCAT 网线是否已连接、机器人是否已开机，然后重新扫描。")
            raise CommandError("机器人扫描失败", result)
        rows = parse_scan_output(result.combined_output)
        report = compare_topology(rows, self.joints)
        pieces = []
        if report.extra_ids:
            pieces.append("多出关节 ID：" + "，".join(report.extra_ids))
        if report.missing_labels:
            pieces.append("缺失关节：" + "，".join(report.missing_labels))
        pieces.extend(report.transport_errors)
        if not report.can_proceed:
            raise ValueError("\n".join(pieces))
        if report.extra_ids:
            self.error = "\n".join(pieces)
            self._complete_stage("scan", "预期关节连接检查通过，但检测到多余关节。")
        else:
            self._complete_stage("scan", "关节连接检查通过。")

    def _render_stress(self) -> None:
        ui.label("将进行 30 秒通讯质量检测。请保持机器人通电且不要拔插连接线。 ").classes("text-lg")
        if self._render_next_step_button("stress"):
            return
        self._render_motor_loss_summary("stress")
        self.countdown_label = ui.label("尚未开始").classes("text-xl text-primary")
        self._buttons(("开始 30 秒压测", self._action(self._run_stress), "primary"))
        self._render_emergency_stop()

    def _test_process_started(self, process: asyncio.subprocess.Process) -> None:
        self.active_test_process = process
        self.refresh()

    def _render_emergency_stop(self) -> None:
        process = self.active_test_process
        if process is None or process.returncode is not None:
            return
        self.test_stop_label = ui.label("测试运行中，可随时紧急停止。 ").classes("text-negative")
        ui.button("紧急停止", on_click=self._emergency_stop, color="negative")

    def _emergency_stop(self) -> None:
        process = self.active_test_process
        if process is None or process.returncode is not None:
            return
        self.test_aborted = True
        interrupt_process(process)
        self._log("操作员按下紧急停止：已发送 Ctrl+C")
        if self.test_stop_label:
            self.test_stop_label.set_text("已发送紧急停止信号；如进程仍在运行，可再次点击。")

    async def _run_interruptible_test(self, args: list[str], seconds: float, **kwargs):
        self.test_aborted = False
        try:
            result = await self._timed_command(
                args,
                seconds,
                on_started=self._test_process_started,
                **kwargs,
            )
        except CommandError as exc:
            if self.test_aborted:
                raise ValueError("测试已紧急停止，请重新开始本步骤。") from exc
            raise
        finally:
            self.active_test_process = None
        if self.test_aborted:
            raise ValueError("测试已紧急停止，请重新开始本步骤。")
        return result

    def _loss_messages(self, diagnosis) -> list[str]:
        messages = []
        if diagnosis.over_limit_motors:
            ids = ", ".join(f"slave={s if s is not None else 0} bus={b} id={m}" for s, b, m in diagnosis.over_limit_motors)
            messages.append(f"丢包率超过 {self.system['max_drop_rate']:.4%} 的电机：{ids}")
        if diagnosis.overall_p90_high:
            messages.append(f"整体电机丢包率 P90={diagnosis.overall_p90:.3%}；请检查网线是否插紧、连接是否松动。")
        for slave, bus in diagnosis.abnormal_buses:
            messages.append(f"Bus slave={slave if slave is not None else 0} bus={bus} 异常；请检查终端电阻与连接。")
        for slave, bus, motor in diagnosis.abnormal_motors:
            messages.append(f"电机 slave={slave if slave is not None else 0} bus={bus} id={motor} 异常；请确认 CAN FD 模式并检查连接。")
        return messages

    async def _run_stress(self) -> None:
        self.motor_loss_summaries.pop("stress", None)
        result = await self._run_interruptible_test(
            ["emcli", "stress", "--no-inplace-refresh", adapter_target(self.config)], 30,
            on_tick=lambda remaining: self.countdown_label and self.countdown_label.set_text(f"剩余 {remaining} 秒"),
        )
        self.raw_output = result.combined_output
        stress = parse_stress_final_result(result.combined_output)
        self.motor_loss_summaries["stress"] = self._motor_loss_summary(stress)
        expected = {joint.key for joint in self.joints}
        observed = {(slave, bus, motor) for slave, bus, motor in stress.motors}
        missing = [joint.label for joint in self.joints if joint.key not in observed]
        if missing:
            raise ValueError("压测结果缺少关节：" + "，".join(missing))
        if any(stat.drop_rate > self.system["max_drop_rate"] for stat in stress.motors.values()):
            diagnosis = analyze_packet_loss(stress, **{key: self.system[key] for key in (
                "max_drop_rate", "overall_p90_limit", "relative_p90_factor")})
            raise ValueError("\n".join(self._loss_messages(diagnosis)))
        self.calibration_index = 0
        self._complete_stage("stress", "通讯质量检测通过。")

    def _current_joint(self) -> Joint | None:
        if not self.joints:
            return None
        return self.joints[self.calibration_index] if self.calibration_index < len(self.joints) else None

    def _render_robot_image(self, relative_path: str, *, warn_if_missing: bool = False) -> None:
        if not relative_path:
            return
        robot_directory = self._robot_directory()
        if robot_directory is None:
            return
        image_path = robot_directory / relative_path
        if image_path.is_file():
            ui.image(str(image_path)).classes("max-w-lg")
        elif warn_if_missing:
            ui.label(f"示意图不存在：{relative_path}").classes("text-orange-8")

    def _render_calibration(self) -> None:
        if self._render_next_step_button("calibration"):
            return
        joint = self._current_joint()
        if joint is None:
            self._advance_from("calibration")
            self.refresh()
            return
        ui.label(f"关节配置 {self.calibration_index + 1}/{len(self.joints)}：{joint.label}").classes("text-xl font-bold")
        if self.calibration_phase == "pose":
            ui.label("子步骤 1：零位标定").classes("text-lg font-bold text-primary")
            self._render_robot_image(joint.image, warn_if_missing=True)
            ui.label(joint.text or "将关节移动到规定零点姿态。")
        else:
            ui.label("子步骤 2：限位检测").classes("text-lg font-bold text-primary")
            self._render_robot_image(joint.image2)
        if self.calibration_phase == "pose":
            self._buttons(("姿态已就位，设置零点", self._action(self._set_zero), "primary"),
                          ("上一步", self._previous_joint, "secondary"))
        else:
            self.position_label = ui.label("当前角度：等待读取")
            self.low_limit_label = None
            self.high_limit_label = None
            with ui.row().classes("gap-6"):
                if joint.pmin is not None:
                    self.low_limit_label = ui.label(f"最小限位 {joint.pmin:.1f}°：待确认")
                if joint.pmax is not None:
                    self.high_limit_label = ui.label(f"最大限位 {joint.pmax:.1f}°：待确认")
            ui.label("请分别转动关节到两侧限位。")
            ui.label(
                f"每个需要检测的限位连续读取 {int(self.system['limit_confirmations'])} 次，"
                f"且误差小于 {self.system['limit_tolerance_deg']:g}° 后自动确认。"
            )
            ui.button("上一步", on_click=self._previous_calibration_phase, color="secondary").props("outline")
            # The zero-position write has already succeeded; start the
            # position monitor as soon as this status view is visible.
            ui.timer(0.1, self._start_limit_monitor, once=True)

    async def _set_zero(self) -> None:
        joint = self._current_joint()
        assert joint is not None
        self._log(f"处理电机零点：{joint.label} slave={joint.slave} bus={joint.bus} id={joint.motor_id}")
        command = ["emcli", "config", motor_target(self.config, joint.slave, joint.bus, joint.motor_id), "position", "set", str(joint.pos)]
        result = await self._command(command, timeout_seconds=8)
        self.raw_output = result.combined_output
        if result.returncode != 0:
            raise CommandError("零点写入失败", result)
        actual = parse_position_output(result.combined_output)
        if abs(actual - joint.pos) >= self.system["position_tolerance_deg"]:
            raise ValueError(f"零点回读失败：目标 {joint.pos:.3f}°，实际 {actual:.3f}°；请重新设置。")
        if not joint.requires_limit_check:
            self._advance_calibration_joint()
        else:
            self.calibration_phase = "limits"
            self._save_progress()

    def _advance_calibration_joint(self) -> None:
        self.calibration_index += 1
        self.calibration_phase = "pose"
        self.limit_low_hits = self.limit_high_hits = 0
        self.limit_low_ok = self.limit_high_ok = False
        self.low_limit_label = None
        self.high_limit_label = None
        if self.calibration_index >= len(self.joints):
            self._complete_stage("calibration", "零点设置完成。")
        else:
            self._save_progress()

    def _start_limit_monitor(self) -> None:
        if self.busy or self.calibration_phase != "limits":
            return
        self.limit_monitor_stop = False
        self.busy = True
        self.refresh()
        asyncio.create_task(self._monitor_limits())

    async def _monitor_limits(self) -> None:
        joint = self._current_joint()
        if joint is None:
            return
        self.error = ""
        try:
            while not self.limit_monitor_stop:
                result = await self._command(["emcli", "config", motor_target(self.config, joint.slave, joint.bus, joint.motor_id), "position"],
                                           timeout_seconds=3)
                self.raw_output = result.combined_output
                if result.returncode != 0:
                    raise CommandError("限位位置读取失败", result)
                if self.limit_monitor_stop:
                    return
                position = parse_position_output(result.combined_output)
                if joint.pmin is not None:
                    self.limit_low_hits = self.limit_low_hits + 1 if abs(position - joint.pmin) < self.system["limit_tolerance_deg"] else 0
                if joint.pmax is not None:
                    self.limit_high_hits = self.limit_high_hits + 1 if abs(position - joint.pmax) < self.system["limit_tolerance_deg"] else 0
                confirmations = int(self.system["limit_confirmations"])
                self.limit_low_ok |= joint.pmin is None or self.limit_low_hits >= confirmations
                self.limit_high_ok |= joint.pmax is None or self.limit_high_hits >= confirmations
                if self.position_label:
                    self.position_label.set_text(f"当前角度：{position:.3f}°")
                low = "已确认" if self.limit_low_ok else f"{self.limit_low_hits}/{confirmations}"
                high = "已确认" if self.limit_high_ok else f"{self.limit_high_hits}/{confirmations}"
                if self.low_limit_label:
                    assert joint.pmin is not None
                    self.low_limit_label.set_text(f"最小限位 {joint.pmin:.1f}°：{low}")
                    if self.limit_low_ok:
                        self.low_limit_label.classes("text-positive")
                if self.high_limit_label:
                    assert joint.pmax is not None
                    self.high_limit_label.set_text(f"最大限位 {joint.pmax:.1f}°：{high}")
                    if self.limit_high_ok:
                        self.high_limit_label.classes("text-positive")
                if self.limit_low_ok and self.limit_high_ok:
                    self._advance_calibration_joint()
                    self.busy = False
                    self.refresh()
                    return
                await asyncio.sleep(0.15)
        except (CommandError, ParseError, ValueError) as exc:
            self.error = str(exc)
            result = getattr(exc, "result", None)
            if result:
                self.raw_output = result.combined_output
            self.busy = False
            self.refresh()
            return
        finally:
            self.busy = False

    def _previous_joint(self) -> None:
        self.limit_monitor_stop = True
        if self.calibration_index <= 0:
            return
        self._clear_page_feedback()
        self.calibration_index -= 1
        self.calibration_phase = "pose"
        self.limit_low_hits = self.limit_high_hits = 0
        self.limit_low_ok = self.limit_high_ok = False
        self._save_progress()
        self.refresh()

    def _previous_calibration_phase(self) -> None:
        """Return from limit detection to zero calibration for the same joint."""
        self.limit_monitor_stop = True
        self.busy = False
        self._clear_page_feedback()
        self.calibration_phase = "pose"
        self.limit_low_hits = self.limit_high_hits = 0
        self.limit_low_ok = self.limit_high_ok = False
        self._save_progress()
        self.refresh()

    def _render_imu(self) -> None:
        if self._render_next_step_button("imu"):
            return
        if not self.imus:
            self._advance_from("imu")
            self.refresh()
            return
        imu = self.imus[self.imu_index]
        ui.label(f"IMU 测试 {self.imu_index + 1}/{len(self.imus)}：{imu.label}").classes("text-xl font-bold")
        self._render_robot_image(imu.image)
        self._render_robot_image(imu.image2)
        ui.label(imu.text or "请晃动机器人。")
        self.imu_average_label = ui.label("平均姿态：等待采样")
        self.imu_range_label = ui.label("极差：等待采样")
        self._buttons(("开始 10 秒 IMU 测试", self._action(self._run_imu), "primary"))

    async def _run_imu(self) -> None:
        imu = self.imus[self.imu_index]
        self._log(f"处理 IMU：{imu.label} slave={imu.slave} bus={imu.bus} id={imu.device_id}")
        def report(reports) -> None:
            evaluation = evaluate_imu_reports(reports, imu)
            if self.imu_average_label:
                self.imu_average_label.set_text("平均姿态：" + ", ".join(
                    f"{axis}={value:.2f}°" + ("（忽略）" if target is None else "")
                    for axis, value, target in zip("XYZ", evaluation.average, imu.pos)
                ))
            if self.imu_range_label:
                self.imu_range_label.set_text("极差：" + ", ".join(
                    f"{axis}={value:.2f}°" + ("（忽略）" if target is None else "")
                    for axis, value, target in zip("XYZ", evaluation.ranges, imu.pos)
                ))
                if evaluation.passed:
                    self.imu_average_label.classes("text-positive")
                    self.imu_range_label.classes("text-positive")
        reports = await collect_imu_for_seconds(["emcli", "imu", "show", imu_target(self.config, imu.slave, imu.bus, imu.device_id)],
                                                seconds=10, connection_timeout=2, on_report=report)
        evaluation = evaluate_imu_reports(reports, imu)
        if not evaluation.passed:
            raise ValueError("IMU 未通过：有效轴的平均姿态偏差须在范围内，且至少一个有效轴的极差须超过 pmax-pmin；请重试。")
        self.imu_index += 1
        if self.imu_index >= len(self.imus):
            self._complete_stage("imu", "姿态检测通过。")
        else:
            self.notice = f"{imu.label} 姿态检测通过，请继续检测下一个 IMU。"
            self._save_progress()

    def _render_motion(self) -> None:
        ui.label("将按当前运动轨迹进行机器人运动与通讯质量检测。请保持周围安全、远离活动部件。 ").classes("text-lg")
        if self._render_next_step_button("motion"):
            return
        self._render_motor_loss_summary("motion")
        self.countdown_label = ui.label("尚未开始").classes("text-xl text-primary")
        if not self.play_path or not self.play_path.exists():
            ui.label("缺少 play.csv").classes("text-red-8")
        self._buttons(("开始运动测试", self._action(self._run_motion), "primary"))
        self._render_emergency_stop()

    async def _run_motion(self) -> None:
        self.motor_loss_summaries.pop("motion", None)
        self.motion_log_results.pop("motion", None)
        if not self.play_path or not self.play_path.exists():
            raise ValueError("缺少 play.csv")
        play_seconds = play_duration_seconds(self.play_path)
        runtime_play = prepare_play_file(
            self.play_path,
            ROOT / ".play.runtime.csv",
            str(self.config.get("plugin", "Ethercat")),
            str(self.config["interface"]),
        )
        # 每次运动测试都写入独立的临时日志目录；先清理历史目录避免误读旧日志。
        for old in ROOT.glob("motion-log-*"):
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
        log_dir = Path(tempfile.mkdtemp(prefix="motion-log-", dir=str(ROOT)))
        self._log(f"运动日志目录：{log_dir}")
        result = await self._run_interruptible_test(
            ["emcli", "play", f"./{runtime_play.name}", "--stress", "--max-current", "25",
             f"--log={log_dir}"],
            play_seconds + 6,
            on_tick=lambda remaining: self.countdown_label and self.countdown_label.set_text(f"剩余 {remaining} 秒"),
            on_timeout=lambda: self.countdown_label and self.countdown_label.set_text("正在等待运动完成"),
            stop_on_timeout=False,
            cwd=str(ROOT),
        )
        self.raw_output = result.combined_output
        # 丢包、跟踪精度与温度一律以落盘日志为准，不再从 stdout 判定。
        report = analyze_play_log_directory(log_dir, self.joints)
        self.motion_log_results["motion"] = report
        stress = self._stress_result_from_logs(report)
        missing = [joint.label for joint in self.joints if joint.key not in set(report.motors)]
        if missing:
            raise ValueError("运动测试日志缺少关节：\n" + "、".join(missing))
        if any(stat.drop_rate > self.system["max_drop_rate"] for stat in stress.motors.values()):
            diagnosis = analyze_packet_loss(stress, **{key: self.system[key] for key in (
                "max_drop_rate", "overall_p90_limit", "relative_p90_factor")})
            raise ValueError("\n".join(self._loss_messages(diagnosis)))
        over_mean = sorted(
            (stat for stat in report.motors.values()
             if stat.mean_error_rad > PLAY_LOG_MEAN_ERROR_LIMIT_RAD),
            key=lambda stat: -stat.mean_error_rad,
        )
        over_continuous = sorted(
            (stat for stat in report.motors.values()
             if stat.longest_over_limit_ms > PLAY_LOG_CONTINUOUS_LIMIT_MS),
            key=lambda stat: -stat.longest_over_limit_ms,
        )
        if over_mean or over_continuous:
            lines: list[str] = []
            if over_mean:
                lines.append(f"平均跟踪误差超 {PLAY_LOG_MEAN_ERROR_LIMIT_RAD:.2f} rad：")
                lines.extend(f"{stat.label}：平均 {stat.mean_error_rad:.4f} rad" for stat in over_mean)
            if over_continuous:
                lines.append(f"跟踪误差连续超 {PLAY_LOG_TRACKING_LIMIT_RAD:.2f} rad 超过 "
                             f"{PLAY_LOG_CONTINUOUS_LIMIT_MS} ms（疑似持续失控）：")
                lines.extend(f"{stat.label}：连续 {stat.longest_over_limit_ms} ms，峰值 "
                             f"{stat.max_error_rad:.4f} rad" for stat in over_continuous)
            raise ValueError("\n".join(lines))
        self.motor_loss_summaries["motion"] = self._motor_loss_summary(stress)
        over_temp = sorted(
            (stat for stat in report.motors.values()
             if stat.max_motor_temp_c > PLAY_LOG_TEMPERATURE_WARN_C),
            key=lambda stat: -stat.max_motor_temp_c,
        )
        message = "运动与通讯质量检测通过。"
        if over_temp:
            hot = "、".join(f"{stat.label}（最高 {stat.max_motor_temp_c:.0f}°C）" for stat in over_temp)
            message += f"注意：{hot} 电机温度超过 {PLAY_LOG_TEMPERATURE_WARN_C:.0f}°C，请留意散热。"
        self._complete_stage("motion", message)

    def _stress_result_from_logs(self, report: PlayLogResult) -> StressResult:
        """Convert log-derived stats to the StressResult shape used by the loss analysis."""
        measurement = report.measurement_seconds
        motors: dict[tuple[int | None, int, int], StressStat] = {
            key: StressStat(stat.drop_rate, stat.missing, stat.sent,
                            (stat.received / measurement) if measurement else 0.0)
            for key, stat in report.motors.items()
        }
        buses: dict[tuple[int | None, int], StressStat] = {}
        for bus_key in {key[:2] for key in motors}:
            group = [motors[key] for key in motors if key[:2] == bus_key]
            sent = sum(stat.sent for stat in group)
            received = sum(stat.sent - stat.drop for stat in group)
            buses[bus_key] = StressStat((sent - received) / sent if sent else 0.0, sent - received,
                                        sent, (received / measurement) if measurement else 0.0)
        sent = sum(stat.sent for stat in motors.values())
        received = sum(stat.sent - stat.drop for stat in motors.values())
        overview = StressStat((sent - received) / sent if sent else 0.0, sent - received, sent,
                              (received / measurement) if measurement else 0.0)
        return StressResult(measurement, overview, buses, motors)

    def _render_done(self) -> None:
        ui.label("该机器人已完成初始化配置。").classes("text-2xl text-positive font-bold")
        ui.label(f"配置文件：{CONFIG_PATH.name}")


@ui.page("/")
def index() -> None:
    wizard = RobotWizard()
    ui.keyboard(on_key=wizard._on_key)
    wizard.content = ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-4")
    wizard.refresh()


def stop_after_last_browser_closes(_client: Client) -> None:
    """End the local setup service once its final browser client is gone."""
    # NiceGUI invokes delete handlers before removing the current client from
    # Client.instances, so one entry means this was the final browser window.
    if len(Client.instances) == 1:
        app.shutdown()


app.on_delete(stop_after_last_browser_closes)


ui.run(title="机器人配置向导", favicon=FAVICON_PATH, reload=False, host="127.0.0.1", port=8080, reconnect_timeout=1.0)
