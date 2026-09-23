import io
import os
import shutil
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


INSTALLER = Path(__file__).resolve().parents[1] / "installer" / "install_robot_setup.sh"


class InstallerTests(unittest.TestCase):
    def test_missing_curl_is_installed_before_uv_and_apt_failure_stops_installation(self):
        for apt_exit in (0, 19):
            with self.subTest(apt_exit=apt_exit), tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                bin_directory = home / "bin"
                bin_directory.mkdir()
                for command in ("bash", "dirname", "chmod"):
                    (bin_directory / command).symlink_to(shutil.which(command))
                scripts = {
                    "sudo": '#!/usr/bin/env bash\nexec "$@"\n',
                    "apt-get": f'''#!/usr/bin/env bash
echo "apt-get $*" >> "$HOME/commands"
if [[ "$1" == update ]]; then exit {apt_exit}; fi
printf '#!/usr/bin/env bash\\necho curl >> "$HOME/commands"\\nexit 37\\n' > "$HOME/bin/curl"
chmod +x "$HOME/bin/curl"
''',
                    "sh": '#!/usr/bin/env bash\nexit 0\n',
                }
                for name, script in scripts.items():
                    executable = bin_directory / name
                    executable.write_text(script, encoding="utf-8")
                    executable.chmod(0o755)
                environment = os.environ.copy()
                environment.update({"HOME": str(home), "PATH": str(bin_directory)})

                completed = subprocess.run(
                    [str(bin_directory / "bash"), str(INSTALLER), str(self._archive(home))],
                    env=environment, capture_output=True, text=True, timeout=30,
                )

                self.assertEqual(completed.returncode, apt_exit or 37, completed.stderr)
                expected = ["apt-get update"]
                if not apt_exit:
                    expected.extend(["apt-get install -y curl", "curl"])
                self.assertEqual((home / "commands").read_text().splitlines(), expected)

    @staticmethod
    def _archive(home: Path) -> Path:
        archive = home / "installer.tar.gz"
        with tarfile.open(archive, "w:gz") as package:
            payload = b"new version\n"
            info = tarfile.TarInfo("robot_setup/new.txt")
            info.size = len(payload)
            package.addfile(info, io.BytesIO(payload))
        return archive

    @staticmethod
    def _environment(home: Path, uv_exit_code: int = 0) -> dict[str, str]:
        bin_directory = home / "bin"
        bin_directory.mkdir()
        uv = bin_directory / "uv"
        uv.write_text(f"#!/usr/bin/env bash\nexit {uv_exit_code}\n", encoding="utf-8")
        uv.chmod(0o755)
        environment = os.environ.copy()
        environment.update({"HOME": str(home), "PATH": f"{bin_directory}:/usr/bin:/bin"})
        return environment

    def test_existing_installation_is_replaced_and_machine_state_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            target = home / "robot_setup"
            target.mkdir()
            (target / "stale.txt").write_text("old", encoding="utf-8")
            (target / "robot_config.yaml").write_text("interface: eth-test\n", encoding="utf-8")
            (target / "robot_progress.json").write_text('{"stage":"scan"}\n', encoding="utf-8")

            archive = self._archive(home)
            environment = self._environment(home)

            completed = subprocess.run(
                ["bash", str(INSTALLER), str(archive)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((target / "new.txt").read_text(encoding="utf-8"), "new version\n")
            self.assertFalse((target / "stale.txt").exists())
            self.assertEqual((target / "robot_config.yaml").read_text(encoding="utf-8"), "interface: eth-test\n")
            self.assertEqual((target / "robot_progress.json").read_text(encoding="utf-8"), '{"stage":"scan"}\n')

    def test_failed_dependency_install_restores_complete_previous_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            target = home / "robot_setup"
            target.mkdir()
            (target / "old.txt").write_text("keep me\n", encoding="utf-8")
            archive = self._archive(home)

            completed = subprocess.run(
                ["bash", str(INSTALLER), str(archive)],
                env=self._environment(home, uv_exit_code=42),
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(completed.returncode, 42)
            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "keep me\n")
            self.assertFalse((target / "new.txt").exists())

    def test_failed_backup_move_does_not_delete_previous_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            target = home / "robot_setup"
            target.mkdir()
            (target / "old.txt").write_text("keep me\n", encoding="utf-8")
            archive = self._archive(home)
            environment = self._environment(home)
            fake_mv = home / "bin" / "mv"
            fake_mv.write_text("#!/usr/bin/env bash\nexit 23\n", encoding="utf-8")
            fake_mv.chmod(0o755)

            completed = subprocess.run(
                ["bash", str(INSTALLER), str(archive)],
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(completed.returncode, 23)
            self.assertEqual((target / "old.txt").read_text(encoding="utf-8"), "keep me\n")


if __name__ == "__main__":
    unittest.main()
