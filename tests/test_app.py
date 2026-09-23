import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path

from robot_runtime import config_directory_sha1, save_progress


def _load_wizard_class():
    fake_nicegui = types.ModuleType("nicegui")
    fake_nicegui.Client = type("Client", (), {})
    fake_nicegui.app = types.SimpleNamespace(on_delete=lambda callback: None)
    fake_nicegui.ui = types.SimpleNamespace(
        page=lambda route: lambda callback: callback,
        run=lambda **kwargs: None,
    )
    previous = sys.modules.get("nicegui")
    sys.modules["nicegui"] = fake_nicegui
    try:
        sys.modules.pop("app", None)
        return importlib.import_module("app").RobotWizard
    finally:
        if previous is None:
            del sys.modules["nicegui"]
        else:
            sys.modules["nicegui"] = previous


RobotWizard = _load_wizard_class()


class CalibrationNavigationTests(unittest.TestCase):
    def test_previous_limit_phase_returns_to_same_joint_pose(self):
        wizard = RobotWizard.__new__(RobotWizard)
        wizard.calibration_index = 2
        wizard.calibration_phase = "limits"
        wizard.busy = True
        wizard.limit_monitor_stop = False
        wizard.limit_low_hits = 3
        wizard.limit_high_hits = 2
        wizard.limit_low_ok = True
        wizard.limit_high_ok = False
        wizard._clear_page_feedback = lambda: None
        wizard._save_progress = lambda: None
        refreshed = []
        wizard.refresh = lambda: refreshed.append(True)

        wizard._previous_calibration_phase()

        self.assertTrue(wizard.limit_monitor_stop)
        self.assertFalse(wizard.busy)
        self.assertEqual(wizard.calibration_index, 2)
        self.assertEqual(wizard.calibration_phase, "pose")
        self.assertEqual(wizard.limit_low_hits, 0)
        self.assertEqual(wizard.limit_high_hits, 0)
        self.assertFalse(wizard.limit_low_ok)
        self.assertFalse(wizard.limit_high_ok)
        self.assertEqual(refreshed, [True])


class InitializationNavigationTests(unittest.IsolatedAsyncioTestCase):
    def test_install_entry_always_allows_robot_type_selection(self):
        wizard = RobotWizard.__new__(RobotWizard)
        wizard._robot_directory = lambda: Path("/configured/robot")
        wizard._clear_page_feedback = lambda: None
        refreshed = []
        wizard.refresh = lambda: refreshed.append(True)

        wizard._enter_install()

        self.assertEqual(wizard.stage, "robot_type")
        self.assertEqual(refreshed, [True])

    async def test_interface_initialization_returns_to_function_menu(self):
        app_module = sys.modules[RobotWizard.__module__]
        original_path = app_module.CONFIG_PATH
        with tempfile.TemporaryDirectory() as temporary_directory:
            app_module.CONFIG_PATH = Path(temporary_directory) / "robot_config.yaml"
            wizard = RobotWizard.__new__(RobotWizard)
            wizard.config = {"robot_type": "H180", "plugin": "Ethercat", "interface": ""}
            wizard.before_interfaces = {"eth0"}
            wizard.raw_output = ""
            wizard._local_interfaces = lambda: {"eth0", "eth1"}
            wizard._first_configuration_stage = lambda: "connect"
            wizard._save_progress = lambda: self.fail("initialization must not start robot configuration progress")

            async def command(*args, **kwargs):
                return types.SimpleNamespace(returncode=0, combined_output="eth0\neth1\n")

            wizard._command = command
            await wizard._scan_with_robot()

            self.assertEqual(wizard.config["interface"], "eth1")
            self.assertEqual(wizard.stage, "home")
            self.assertIn("eth1", wizard.notice)
        app_module.CONFIG_PATH = original_path


class ProgressRestoreTests(unittest.TestCase):
    def test_only_matching_configuration_hash_is_offered_for_restore(self):
        app_module = sys.modules[RobotWizard.__module__]
        original_path = app_module.PROGRESS_PATH
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_directory = root / "H130"
            config_directory.mkdir()
            (config_directory / "robot_joints.csv").write_text("joint", encoding="utf-8")
            progress_path = root / "robot_progress.json"
            app_module.PROGRESS_PATH = progress_path
            wizard = RobotWizard.__new__(RobotWizard)
            wizard.config = {"robot_type": "H130"}
            wizard._robot_directory = lambda: config_directory
            progress = {"robot_type": "H130", "config_sha1": config_directory_sha1(config_directory), "stage": "scan"}
            save_progress(progress_path, progress)

            self.assertEqual(wizard._load_recoverable_progress(), progress)

            (config_directory / "robot_joints.csv").write_text("changed", encoding="utf-8")
            self.assertIsNone(wizard._load_recoverable_progress())
            self.assertFalse(progress_path.exists())
        app_module.PROGRESS_PATH = original_path


class NetworkSafetyTests(unittest.TestCase):
    def test_network_failure_resets_calibration_to_its_initial_substep(self):
        wizard = RobotWizard.__new__(RobotWizard)
        wizard.network_fault = False
        wizard.limit_monitor_stop = False
        wizard.active_command_process = None
        wizard.active_test_process = None
        wizard.test_aborted = False
        wizard.stage = "calibration"
        wizard.calibration_phase = "limits"
        wizard.limit_low_hits = 2
        wizard.limit_high_hits = 1
        wizard.limit_low_ok = True
        wizard.limit_high_ok = False
        wizard.busy = True
        wizard._log = lambda message: None
        wizard._clear_page_feedback = lambda: None
        wizard._save_progress = lambda: None
        refreshed = []
        wizard.refresh = lambda: refreshed.append(True)

        wizard._handle_network_failure()

        self.assertTrue(wizard.network_fault)
        self.assertTrue(wizard.limit_monitor_stop)
        self.assertFalse(wizard.busy)
        self.assertEqual(wizard.calibration_phase, "pose")
        self.assertEqual(wizard.limit_low_hits, 0)
        self.assertEqual(wizard.limit_high_hits, 0)
        self.assertFalse(wizard.limit_low_ok)
        self.assertFalse(wizard.limit_high_ok)
        self.assertEqual(refreshed, [True])


class AdministratorNavigationTests(unittest.TestCase):
    def test_administrator_dialog_uses_ctrl_b_not_ctrl_n(self):
        wizard = RobotWizard.__new__(RobotWizard)
        wizard.network_fault = False
        wizard.admin_dialog_open = False
        opened = []
        wizard._show_admin_jump_dialog = lambda: opened.append(True)

        keydown = types.SimpleNamespace(keydown=True, repeat=False)
        RobotWizard._on_key(wizard, types.SimpleNamespace(
            action=keydown,
            modifiers=types.SimpleNamespace(ctrl=True),
            key="n",
        ))
        self.assertEqual(opened, [])

        RobotWizard._on_key(wizard, types.SimpleNamespace(
            action=keydown,
            modifiers=types.SimpleNamespace(ctrl=True),
            key="b",
        ))
        self.assertEqual(opened, [True])

    def test_jump_opens_without_password_configuration(self):
        wizard = RobotWizard.__new__(RobotWizard)
        wizard.system = {}
        wizard.admin_dialog_open = False
        opened = []
        wizard._show_admin_jump_dialog = lambda: opened.append(True)

        wizard._open_admin_jump()

        self.assertEqual(opened, [True])

    def test_jump_does_not_open_a_duplicate_dialog(self):
        wizard = RobotWizard.__new__(RobotWizard)
        wizard.admin_dialog_open = True
        wizard._show_admin_jump_dialog = lambda: self.fail("dialog already open")

        wizard._open_admin_jump()

    def test_administrator_jump_selects_calibration_joint_and_phase(self):
        wizard = RobotWizard.__new__(RobotWizard)
        wizard.limit_monitor_stop = False
        wizard.connection_monitor_stop = False
        wizard.calibration_index = 0
        wizard.calibration_phase = "pose"
        wizard.limit_low_hits = 4
        wizard.limit_high_hits = 3
        wizard.limit_low_ok = True
        wizard.limit_high_ok = True
        wizard.imu_index = 0
        wizard._clear_page_feedback = lambda: None
        wizard._save_progress = lambda: None
        refreshed = []
        wizard.refresh = lambda: refreshed.append(True)
        closed = []
        dialog = types.SimpleNamespace(close=lambda: closed.append(True))

        wizard._apply_admin_jump(dialog, {
            "stage": "calibration",
            "calibration_index": 5,
            "calibration_phase": "limits",
        })

        self.assertTrue(wizard.limit_monitor_stop)
        self.assertTrue(wizard.connection_monitor_stop)
        self.assertEqual(wizard.stage, "calibration")
        self.assertEqual(wizard.calibration_index, 5)
        self.assertEqual(wizard.calibration_phase, "limits")
        self.assertEqual(wizard.limit_low_hits, 0)
        self.assertEqual(wizard.limit_high_hits, 0)
        self.assertFalse(wizard.limit_low_ok)
        self.assertFalse(wizard.limit_high_ok)
        self.assertEqual(closed, [True])
        self.assertEqual(refreshed, [True])
