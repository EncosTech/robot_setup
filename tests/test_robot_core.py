import unittest
from pathlib import Path
import tempfile

from robot_core import (
    ImuConfig,
    ImuReport,
    Joint,
    PLAY_LOG_MEAN_ERROR_LIMIT_RAD,
    PLAY_LOG_TRACKING_LIMIT_RAD,
    analyze_packet_loss,
    analyze_play_log_directory,
    angular_distance_degrees,
    compare_topology,
    format_scan_choice_labels,
    is_ethercat_connection_missing,
    parse_imu_reports,
    parse_interface_output,
    parse_position_output,
    parse_play_stress_result,
    parse_scan_output,
    parse_stress_final_result,
    read_hardware_csv,
    evaluate_imu_reports,
    format_joint_button_label,
)


class OutputParserTests(unittest.TestCase):
    @staticmethod
    def _write_play_log_pair(
        directory: Path,
        stem: str,
        command_positions: list[tuple[int, float]],
        status_positions: list[tuple[int, float]],
    ) -> None:
        command_header = "timestamp_ns,a,b,c,position\n"
        status_header = "timestamp_ns,a,position,speed,current,motor_temp,mos_temp\n"
        (directory / f"{stem}_command.csv").write_text(
            command_header
            + "".join(f"{timestamp},0,0,0,{position}\n" for timestamp, position in command_positions),
            encoding="utf-8",
        )
        (directory / f"{stem}_status.csv").write_text(
            status_header
            + "".join(f"{timestamp},0,{position},0,0,30,35\n" for timestamp, position in status_positions),
            encoding="utf-8",
        )

    def test_play_log_rejects_non_finite_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_play_log_pair(root, "motor_0_1_0", [(0, float("nan"))], [(0, float("nan"))])
            with self.assertRaisesRegex(ValueError, "非有限数值"):
                analyze_play_log_directory(root, [Joint("hip", 0, 0, 1, 0, None, None, "", "")])

    def test_play_log_keeps_same_bus_and_motor_id_on_different_slaves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_play_log_pair(root, "motor_0_0_1", [(0, 0.0)], [(0, 0.0)])
            self._write_play_log_pair(root, "motor_1_0_1", [(0, 0.0)], [(0, 0.0)])
            joints = [
                Joint("left", 0, 0, 1, 0, None, None, "", ""),
                Joint("right", 1, 0, 1, 0, None, None, "", ""),
            ]
            result = analyze_play_log_directory(root, joints)
        self.assertEqual(set(result.motors), {(0, 0, 1), (1, 0, 1)})

    def test_play_log_legacy_suffix_maps_only_bus_and_motor_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_play_log_pair(root, "motor_0_1_0", [(0, 0.0)], [(0, 0.0)])
            self._write_play_log_pair(root, "motor_0_2_1", [(0, 0.0)], [(0, 0.0)])
            joints = [
                Joint("motor one", 0, 0, 1, 0, None, None, "", ""),
                Joint("motor two", 0, 0, 2, 0, None, None, "", ""),
            ]
            result = analyze_play_log_directory(root, joints)
        self.assertEqual(set(result.motors), {(0, 0, 1), (0, 0, 2)})

    def test_play_log_uses_timestamps_for_duration_and_continuous_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timestamps = [0, 500_000_000, 1_000_000_000, 1_600_000_000]
            self._write_play_log_pair(
                root,
                "motor_0_1_0",
                [(timestamp, 0.16) for timestamp in timestamps],
                [(timestamp, 0.0) for timestamp in timestamps],
            )
            result = analyze_play_log_directory(
                root,
                [Joint("hip", 0, 0, 1, 0, None, None, "", "")],
            )
        stat = result.motors[(0, 0, 1)]
        self.assertAlmostEqual(result.measurement_seconds, 1.6)
        self.assertEqual(stat.longest_over_limit_ms, 1600)
        self.assertEqual(PLAY_LOG_MEAN_ERROR_LIMIT_RAD, 0.15)
        self.assertEqual(PLAY_LOG_TRACKING_LIMIT_RAD, 0.15)

    def test_imu_evaluation_uses_mean_range_and_thresholds(self):
        config = ImuConfig("imu_a", 0, 0, 0, (1.0, 2.0, 3.0), -0.5, 0.5, "", "")
        reports = [
            ImuReport((0, 0, 0), (0, 0, 0), (0.0, 2.0, 3.0)),
            ImuReport((0, 0, 0), (0, 0, 0), (2.0, 2.0, 3.0)),
            ImuReport((0, 0, 0), (0, 0, 0), (1.0, 2.0, 3.0)),
        ]
        evaluation = evaluate_imu_reports(reports, config)
        self.assertTrue(evaluation.mean_ok)
        self.assertTrue(evaluation.range_ok)
        self.assertTrue(evaluation.passed)
        self.assertEqual(evaluation.ranges, (2.0, 0.0, 0.0))

    def test_imu_tilde_target_ignores_only_that_axis(self):
        config = ImuConfig("imu_a", 0, 0, 0, (1.0, 2.0, None), -0.5, 0.5, "", "")
        reports = [
            ImuReport((0, 0, 0), (0, 0, 0), (0.0, 2.0, 120.0)),
            ImuReport((0, 0, 0), (0, 0, 0), (2.0, 2.0, 120.0)),
            ImuReport((0, 0, 0), (0, 0, 0), (1.0, 2.0, 120.0)),
        ]
        evaluation = evaluate_imu_reports(reports, config)
        self.assertTrue(evaluation.mean_ok)
        self.assertTrue(evaluation.passed)

    def test_imu_numeric_z_target_is_checked(self):
        config = ImuConfig("imu_a", 0, 0, 0, (1.0, 2.0, 3.0), -0.5, 0.5, "", "")
        reports = [ImuReport((0, 0, 0), (0, 0, 0), (1.0, 2.0, 120.0))]
        evaluation = evaluate_imu_reports(reports, config)
        self.assertFalse(evaluation.mean_ok)
        self.assertFalse(evaluation.passed)

    def test_hardware_csv_parses_tilde_as_ignored_imu_axis_and_motor_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot_joints.csv"
            path.write_text(
                "type,label,slave,bus,id,pos,pmin,pmax,image,image2,text\n"
                "motor,hip,0,0,1,0,~,90,,,\n"
                "motor,knee,0,0,2,0,~,~,,,\n"
                'imu,body_imu,0,7,0,"[0,90,~]",-5,5,,,\n',
                encoding="utf-8",
            )
            joints, imus = read_hardware_csv(path)
        self.assertIsNone(joints[0].pmin)
        self.assertEqual(joints[0].pmax, 90.0)
        self.assertTrue(joints[0].requires_limit_check)
        self.assertIsNone(joints[1].pmin)
        self.assertIsNone(joints[1].pmax)
        self.assertFalse(joints[1].requires_limit_check)
        self.assertEqual(imus[0].pos, (0.0, 90.0, None))

    def test_imu_with_all_axes_ignored_skips_pose_and_range_checks(self):
        config = ImuConfig("imu_a", 0, 0, 0, (None, None, None), -0.5, 0.5, "", "")
        reports = [ImuReport((0, 0, 0), (0, 0, 0), (100.0, 200.0, 300.0))]
        evaluation = evaluate_imu_reports(reports, config)
        self.assertTrue(evaluation.mean_ok)
        self.assertTrue(evaluation.range_ok)
        self.assertTrue(evaluation.passed)

    def test_hardware_csv_allows_imu_only_robot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot_joints.csv"
            path.write_text(
                "type,label,slave,bus,id,pos,pmin,pmax,image,image2,text\n"
                'imu,body_imu,0,7,0,"[0,0,0]",-5,5,pose.png,shake.png,shake\n',
                encoding="utf-8",
            )
            joints, imus = read_hardware_csv(path)
        self.assertEqual(joints, [])
        self.assertEqual(len(imus), 1)
        self.assertEqual(imus[0].image, "pose.png")
        self.assertEqual(imus[0].image2, "shake.png")

    def test_hardware_csv_allows_empty_secondary_image(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot_joints.csv"
            path.write_text(
                "type,label,slave,bus,id,pos,pmin,pmax,image,image2,text\n"
                "motor,hip,0,0,1,0,-90,90,pose.png,,move to limit\n",
                encoding="utf-8",
            )
            joints, imus = read_hardware_csv(path)
        self.assertEqual(imus, [])
        self.assertEqual(joints[0].image, "pose.png")
        self.assertEqual(joints[0].image2, "")
    def test_euler_angle_difference_wraps_at_180_degrees(self):
        self.assertEqual(angular_distance_degrees(179.0, -179.0), 2.0)

    def test_interface_scan_ignores_logs_and_only_accepts_interface_tokens(self):
        output = """[warning] driver log\nenx00e04c360470\neth0\nError: not an interface\n"""
        self.assertEqual(parse_interface_output(output, {"enx00e04c360470", "eth0"}), ["enx00e04c360470", "eth0"])

    def test_scan_ignores_driver_logs_and_normalizes_blank_slave(self):
        output = """
[2026-07-31 09:08:42.960] [warn] driver is settling
slave  bus  id  eff  canfd
      0    1    0    1
[warn] unrelated log
2  3  7  0  1
"""
        rows = parse_scan_output(output)
        self.assertEqual([(row.slave, row.bus, row.motor_id) for row in rows], [(0, 0, 1), (2, 3, 7)])
        self.assertIsNone(rows[0].raw_slave)
        self.assertEqual(rows[1].raw_slave, 2)

    def test_id_choice_labels_include_only_ambiguous_address_components(self):
        same_bus = parse_scan_output("slave bus id eff canfd\n0 41 1 0 1\n0 41 2 0 1\n")
        self.assertEqual(format_scan_choice_labels(same_bus), ["1", "2"])

        multiple_buses = parse_scan_output("slave bus id eff canfd\n0 41 1 0 1\n0 42 1 0 1\n")
        self.assertEqual(format_scan_choice_labels(multiple_buses), ["41:1", "42:1"])

        multiple_slaves_and_buses = parse_scan_output("slave bus id eff canfd\n0 41 1 0 1\n2 42 1 0 1\n")
        self.assertEqual(format_scan_choice_labels(multiple_slaves_and_buses), ["0:41:1", "2:42:1"])

    def test_detects_ethercat_initialization_failure_amid_driver_logs(self):
        output = "[warning] retrying\nError: Failed to Initialize EtherCAT\n[warn] another line\n"
        self.assertTrue(is_ethercat_connection_missing(output))
        self.assertTrue(is_ethercat_connection_missing("failed   TO initialize\tethercat"))
        self.assertFalse(is_ethercat_connection_missing("Error: failed to scan motor table"))

    def test_position_uses_last_complete_business_record(self):
        output = """[warn] before\nposition -2.5\n[warn] afterwards\nposition 12.345678\n"""
        self.assertEqual(parse_position_output(output), 12.345678)

    def test_stress_uses_only_last_final_result_and_ignores_logs(self):
        output = """
===== LIVE STATUS (Press Ctrl+C to stop) =====
Overview: drop_rate=99.00% drop=99/100 actual_fps=1.00
================ FINAL RESULT ================
Measurement seconds: 30.002
Overview: drop_rate=0.02% drop=2/10000 actual_fps=333.00
[warn] a driver warning between report rows
  Adapter Ethercat:enx0: drop_rate=0.02% drop=2/10000 actual_fps=333.00
    Bus 0: drop_rate=0.02% drop=2/10000 actual_fps=333.00
      Motor 1: drop_rate=0.02% drop=2/10000 actual_fps=333.00
      Motor 2: drop_rate=0.00% drop=0/10000 actual_fps=333.00
"""
        result = parse_stress_final_result(output)
        self.assertEqual(result.overview.drop, 2)
        self.assertEqual(result.motors[(0, 0, 1)].sent, 10000)
        self.assertAlmostEqual(result.motors[(0, 0, 1)].drop_rate, 0.0002)

    def test_stress_uses_exact_drop_over_sent_not_rounded_display_rate(self):
        output = """
================ FINAL RESULT ================
Overview: drop_rate=0.10% drop=95/100000 actual_fps=1.00
  Adapter Ethercat:eth0: drop_rate=0.10% drop=95/100000 actual_fps=1.00
    Bus 0: drop_rate=0.10% drop=95/100000 actual_fps=1.00
      Motor 1: drop_rate=0.10% drop=95/100000 actual_fps=1.00
"""
        result = parse_stress_final_result(output)
        self.assertAlmostEqual(result.motors[(0, 0, 1)].drop_rate, 0.00095)

    def test_imu_requires_a_complete_report_even_with_logs_between_rows(self):
        output = """
group              x             y             z
accel       0.1  9.8  0.0
[warn] frame delayed
gyro        0.0  0.0  0.0
euler       1.0  2.0  3.0
group              x             y             z
accel       0.1  9.8  0.0
gyro        0.0  0.0  0.0
euler       12.0 2.0  3.0
"""
        reports = parse_imu_reports(output)
        self.assertEqual(len(reports), 2)
        self.assertEqual(reports[-1].euler, (12.0, 2.0, 3.0))

    def test_play_stress_uses_the_last_complete_snapshot_despite_logs(self):
        output = """
===== PLAY STRESS STATUS (Press Ctrl+C to stop) =====
Measurement seconds: 1.000
Motor Ethercat:eth0:0:1: sent=10 received=10 lost=0 loss=0.00%
===== PLAY STRESS STATUS (Press Ctrl+C to stop) =====
Measurement seconds: 22.000
[warn] transient bus message
Motor Ethercat:eth0:0:1: sent=22000 received=21998 lost=2 loss=0.01%
Motor Ethercat:eth0:0:2: sent=22000 received=22000 lost=0 loss=0.00%
"""
        result = parse_play_stress_result(output)
        self.assertEqual(result.motors[(0, 0, 1)].drop, 2)
        self.assertEqual(result.motors[(0, 0, 2)].sent, 22000)


class SafetyAnalysisTests(unittest.TestCase):
    def test_joint_button_label_includes_label_and_motor_id(self):
        joint = Joint("左膝", 0, 1, 7, 0, -90, 90, "", "")
        self.assertEqual(format_joint_button_label(joint), "左膝（ID 7）")

    def test_topology_extra_motor_is_nonblocking_but_missing_or_transport_fault_is_blocking(self):
        expected = [Joint("hip", 0, 0, 1, 0, -90, 90, "", "")]

        extra_only = compare_topology(
            parse_scan_output("slave bus id eff canfd\n0 1 0 1\n0 9 0 1\n"),
            expected,
        )
        self.assertFalse(extra_only.ok)
        self.assertTrue(extra_only.can_proceed)

        missing = compare_topology(
            parse_scan_output("slave bus id eff canfd\n0 9 0 1\n"),
            expected,
        )
        self.assertFalse(missing.can_proceed)

        transport_fault = compare_topology(
            parse_scan_output("slave bus id eff canfd\n0 1 1 0\n"),
            expected,
        )
        self.assertFalse(transport_fault.can_proceed)

    def test_topology_reports_extra_missing_and_transport_faults(self):
        joints = [Joint("hip", 0, 0, 1, 0, -90, 90, "", ""), Joint("knee", 0, 0, 2, 0, -90, 90, "", "")]
        rows = parse_scan_output("slave bus id eff canfd\n0 1 0 1\n0 9 1 0\n")
        report = compare_topology(rows, joints)
        self.assertEqual(report.missing_labels, ["knee"])
        self.assertEqual(report.extra_ids, ["slave=0 bus=0 id=9"])
        self.assertEqual(len(report.transport_errors), 2)

    def test_packet_loss_diagnoses_overall_bus_and_motor_conditions_together(self):
        output = """
================ FINAL RESULT ================
Measurement seconds: 30.000
Overview: drop_rate=0.20% drop=3/1500 actual_fps=30.00
  Adapter Ethercat:eth0: drop_rate=0.20% drop=3/1500 actual_fps=30.00
    Bus 0: drop_rate=0.00% drop=0/500 actual_fps=30.00
      Motor 1: drop_rate=0.00% drop=0/500 actual_fps=30.00
    Bus 1: drop_rate=0.30% drop=3/1000 actual_fps=30.00
      Motor 2: drop_rate=0.60% drop=3/500 actual_fps=30.00
      Motor 3: drop_rate=0.00% drop=0/500 actual_fps=30.00
"""
        result = parse_stress_final_result(output)
        diagnosis = analyze_packet_loss(result, max_drop_rate=0.001, overall_p90_limit=0.00075,
                                        relative_p90_factor=1.1)
        self.assertTrue(diagnosis.failed)
        self.assertTrue(diagnosis.overall_p90_high)
        self.assertIn((0, 1), diagnosis.abnormal_buses)
        self.assertIn((0, 1, 2), diagnosis.abnormal_motors)

    def test_packet_loss_fails_only_when_a_motor_strictly_exceeds_limit(self):
        output = """
================ FINAL RESULT ================
Measurement seconds: 30.000
Overview: drop_rate=0.05% drop=11/20000 actual_fps=30.00
  Adapter Ethercat:eth0: drop_rate=0.05% drop=11/20000 actual_fps=30.00
    Bus 0: drop_rate=0.05% drop=11/20000 actual_fps=30.00
      Motor 1: drop_rate=0.05% drop=5/10000 actual_fps=30.00
      Motor 2: drop_rate=0.06% drop=6/10000 actual_fps=30.00
"""
        result = parse_stress_final_result(output)
        diagnosis = analyze_packet_loss(
            result,
            max_drop_rate=0.0005,
            overall_p90_limit=1.0,
            relative_p90_factor=1.1,
        )
        self.assertEqual(diagnosis.over_limit_motors, [(0, 0, 2)])
        self.assertTrue(diagnosis.failed)

    def test_p90_analysis_does_not_fail_without_an_over_limit_motor(self):
        output = """
================ FINAL RESULT ================
Measurement seconds: 30.000
Overview: drop_rate=0.04% drop=4/10000 actual_fps=30.00
  Adapter Ethercat:eth0: drop_rate=0.04% drop=4/10000 actual_fps=30.00
    Bus 0: drop_rate=0.04% drop=4/10000 actual_fps=30.00
      Motor 1: drop_rate=0.04% drop=4/10000 actual_fps=30.00
"""
        result = parse_stress_final_result(output)
        diagnosis = analyze_packet_loss(
            result,
            max_drop_rate=0.0005,
            overall_p90_limit=0.0,
            relative_p90_factor=0.5,
        )
        self.assertTrue(diagnosis.overall_p90_high)
        self.assertFalse(diagnosis.failed)


if __name__ == "__main__":
    unittest.main()
