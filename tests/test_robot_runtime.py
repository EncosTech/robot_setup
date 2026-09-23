import asyncio
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from robot_runtime import (
    default_config,
    config_directory_sha1,
    imu_target,
    load_progress,
    load_config,
    load_system_settings,
    motor_target,
    parse_interface_link_state,
    inspect_package_archive,
    play_duration_seconds,
    prepare_play_file,
    runtime_package_archive,
    save_progress,
    remove_progress,
    interrupt_process,
    run_timed_command,
    save_config,
)


class TimedCommandTests(unittest.IsolatedAsyncioTestCase):
    async def test_observed_timed_command_waits_for_natural_completion_after_countdown(self):
        waiting = []
        result = await run_timed_command(
            [sys.executable, "-c", "import time; time.sleep(.45); print('finished')"],
            .03,
            stop_on_timeout=False,
            on_timeout=lambda: waiting.append(True),
        )

        self.assertIn("finished", result.stdout)
        self.assertEqual(waiting, [True])

    async def test_interrupt_waits_for_graceful_post_signal_final_output(self):
        program = (
            "import signal,time\n"
            "stopping=False\n"
            "def stop(_sig,_frame):\n global stopping\n stopping=True\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "while not stopping: time.sleep(.01)\n"
            "time.sleep(.05)\n"
            "print('FINAL RESULT AFTER GRACEFUL STOP', flush=True)\n"
        )
        result = await run_timed_command([sys.executable, "-c", program], 1)
        self.assertEqual(result.returncode, 0)
        self.assertIn("FINAL RESULT AFTER GRACEFUL STOP", result.stdout)

    async def test_running_process_can_receive_repeated_interrupts(self):
        program = (
            "import signal,time\n"
            "count=0\n"
            "def stop(_sig,_frame):\n"
            " global count\n"
            " count += 1\n"
            " print(f'interrupt {count}', flush=True)\n"
            " if count >= 2: raise SystemExit(0)\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "while True: time.sleep(.01)\n"
        )

        def started(process):
            async def stop_twice():
                await asyncio.sleep(0.1)
                interrupt_process(process)
                await asyncio.sleep(0.1)
                interrupt_process(process)
            asyncio.create_task(stop_twice())

        result = await run_timed_command([sys.executable, "-c", program], 5, on_started=started)
        self.assertIn("interrupt 1", result.stdout)
        self.assertIn("interrupt 2", result.stdout)


class TargetEncodingTests(unittest.TestCase):
    def test_interface_link_state_requires_admin_and_carrier_up(self):
        self.assertTrue(parse_interface_link_state('[{"ifname":"enx0","flags":["BROADCAST","UP","LOWER_UP"],"operstate":"UP"}]'))
        self.assertFalse(parse_interface_link_state('[{"ifname":"enx0","flags":["BROADCAST","UP"],"operstate":"DOWN"}]'))
        self.assertFalse(parse_interface_link_state('not json'))

    def test_progress_round_trip_and_configuration_hash_detects_changes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_directory = root / "H130"
            config_directory.mkdir()
            (config_directory / "robot_joints.csv").write_text("joint-a", encoding="utf-8")
            (config_directory / "nested").mkdir()
            (config_directory / "nested" / "play.csv").write_text("move-a", encoding="utf-8")
            expected_hash = config_directory_sha1(config_directory)
            progress_path = root / "robot_progress.json"
            progress = {"robot_type": "H130", "config_sha1": expected_hash, "stage": "scan"}

            save_progress(progress_path, progress)

            self.assertEqual(load_progress(progress_path), progress)
            (config_directory / "nested" / "play.csv").write_text("move-b", encoding="utf-8")
            self.assertNotEqual(config_directory_sha1(config_directory), expected_hash)
            remove_progress(progress_path)
            self.assertIsNone(load_progress(progress_path))

    def test_runtime_package_archive_is_fixed_beside_the_program(self):
        runtime_root = Path("/opt/robot_setup")
        self.assertEqual(runtime_package_archive(runtime_root), runtime_root / "packages.zip")

    def test_direct_bus_uses_four_part_motor_and_imu_targets(self):
        config = default_config()
        self.assertEqual(config, {"robot_type": "", "plugin": "Ethercat", "interface": ""})
        config["interface"] = "enx0"
        self.assertEqual(motor_target(config, 0, 0, 7), "Ethercat:enx0:0:7")
        self.assertEqual(imu_target(config, 0, 0, 0), "Ethercat:enx0:0:0")

    def test_robot_type_is_saved_and_loaded_with_machine_configuration(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "robot_config.yaml"
            config = default_config()
            config["robot_type"] = "H130"
            config["interface"] = "enx0"
            save_config(path, config)
            self.assertEqual(load_config(path), config)

    def test_system_settings_reads_step_switches(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "system.yaml"
            path.write_text("enable_scan: false\nenable_imu: false\nmax_drop_rate: 0.002\n", encoding="utf-8")
            settings = load_system_settings(path)
        self.assertFalse(settings["enable_scan"])
        self.assertFalse(settings["enable_imu"])
        self.assertTrue(settings["enable_connect"])
        self.assertTrue(settings["enable_stress"])
        self.assertEqual(settings["max_drop_rate"], 0.002)
        self.assertEqual(settings["limit_tolerance_deg"], 4.0)

    def test_system_settings_ignores_legacy_administrator_password(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "system.yaml"
            path.write_text("admin_password_sha1: abcdef0123456789abcdef0123456789abcdef01\n", encoding="utf-8")
            self.assertNotIn("admin_password_sha1", load_system_settings(path))

    def test_play_template_replaces_plugin_and_interface(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "play.csv"
            destination = root / "runtime.csv"
            source.write_text("time,plugin-replace-me:enx-replace-me:0:1\n0,0\n", encoding="utf-8")
            prepare_play_file(source, destination, "CustomPlugin", "enx-test")
            self.assertEqual(destination.read_text(encoding="utf-8"), "time,CustomPlugin:enx-test:0:1\n0,0\n")

    def test_play_duration_uses_the_largest_time_value(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "play.csv"
            path.write_text("time,motor\n0,0\n1.5,0\n1.0,0\n", encoding="utf-8")
            self.assertEqual(play_duration_seconds(path), 1.5)

    def test_package_archive_requires_matching_release_architecture_and_emcli_version(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "packages.zip"
            with zipfile.ZipFile(archive, "w") as package_zip:
                package_zip.writestr("libencosdriver_3.0.9_jammy_amd64.deb", "placeholder")
                package_zip.writestr("emcli_1.9.9_jammy_amd64.deb", "placeholder")
                package_zip.writestr("emcli_9.9.9_noble_arm64.deb", "placeholder")
            check = inspect_package_archive(archive, "jammy", "amd64")
            self.assertTrue(check.usable)
            self.assertEqual(check.driver_member, "libencosdriver_3.0.9_jammy_amd64.deb")
            self.assertEqual(check.cli_member, "emcli_1.9.9_jammy_amd64.deb")
            with zipfile.ZipFile(archive, "w") as package_zip:
                package_zip.writestr("libencosdriver_3.0.9_jammy_amd64.deb", "placeholder")
                package_zip.writestr("emcli_1.9.8_jammy_amd64.deb", "placeholder")
            self.assertFalse(inspect_package_archive(archive, "jammy", "amd64").usable)


if __name__ == "__main__":
    unittest.main()
