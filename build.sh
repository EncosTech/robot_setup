#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/dist"
ARCHIVE="$OUTPUT_DIR/robot_setup-installer.tar.gz"
PACKAGES_ARCHIVE="${PACKAGES_ARCHIVE:-/home/bismarck/deb_center/packages/packages.zip}"
STAGING_DIR="$(mktemp -d "$ROOT_DIR/.robot-setup-stage.XXXXXX")"
PROJECT_DIR="$STAGING_DIR/robot_setup"

cleanup() {
    rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

if [[ ! -f "$PACKAGES_ARCHIVE" ]]; then
    echo "packages.zip not found: $PACKAGES_ARCHIVE" >&2
    echo "Set PACKAGES_ARCHIVE to the installation package path and try again." >&2
    exit 1
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR" "$PROJECT_DIR"

# The installation archive holds only runtime source and field-editable robot
# definitions. The generated robot_config.yaml remains machine-specific.
tar \
    --exclude='./.git' \
    --exclude='./.agents' \
    --exclude='./.codex' \
    --exclude='./.venv' \
    --exclude='*/__pycache__' \
    --exclude='*/__pycache__/*' \
    --exclude='./build' \
    --exclude='./dist' \
    --exclude='./.robot-setup-stage.*' \
    --exclude='./installer' \
    --exclude='./tests' \
    --exclude='./build.sh' \
    --exclude='./robot_config.yaml' \
    -cf - -C "$ROOT_DIR" . | tar -xf - -C "$PROJECT_DIR"

install -m 644 "$PACKAGES_ARCHIVE" "$PROJECT_DIR/packages.zip"
install -m 755 "$ROOT_DIR/installer/start_robot_setup.sh" "$PROJECT_DIR/start_robot_setup.sh"

rm -f "$ARCHIVE"
tar -C "$STAGING_DIR" -czf "$ARCHIVE" robot_setup
install -m 755 "$ROOT_DIR/installer/install_robot_setup.sh" "$OUTPUT_DIR/install_robot_setup.sh"

echo "Built: $ARCHIVE"
echo "Installer: $OUTPUT_DIR/install_robot_setup.sh"
