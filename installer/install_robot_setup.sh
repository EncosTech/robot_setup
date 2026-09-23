#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE="${1:-$SCRIPT_DIR/robot_setup-installer.tar.gz}"
TARGET_DIR="$HOME/robot_setup"

usage() {
    cat <<'EOF'
Usage: ./install_robot_setup.sh [robot_setup-installer.tar.gz]

Installs Robot Setup to ~/robot_setup, installs runtime Python dependencies,
and creates a desktop launcher.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$ARCHIVE" ]]; then
    echo "Offline archive not found: $ARCHIVE" >&2
    exit 1
fi

if [[ -e "$TARGET_DIR" && ( ! -d "$TARGET_DIR" || -L "$TARGET_DIR" ) ]]; then
    echo "Install path exists but is not a regular directory: $TARGET_DIR" >&2
    exit 1
fi

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
else
    if ! command -v curl >/dev/null 2>&1; then
        APT_COMMAND=(apt-get)
        if (( EUID != 0 )); then
            APT_COMMAND=(sudo apt-get)
        fi
        "${APT_COMMAND[@]}" update
        "${APT_COMMAND[@]}" install -y curl
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV_BIN="${HOME}/.local/bin/uv"
    if [[ ! -x "$UV_BIN" ]]; then
        UV_BIN="${HOME}/.cargo/bin/uv"
    fi
    if [[ ! -x "$UV_BIN" ]]; then
        echo 'uv installation did not produce an executable.' >&2
        exit 1
    fi
fi

EXTRACT_DIR="$(mktemp -d "$HOME/.robot-setup-extract.XXXXXX")"
BACKUP_DIR=""
BACKUP_COMPLETE=false
INSTALL_COMMITTED=false
cleanup() {
    rm -rf "$EXTRACT_DIR"
    if [[ "$BACKUP_COMPLETE" == true ]]; then
        if [[ "$INSTALL_COMMITTED" == true ]]; then
            rm -rf "$BACKUP_DIR"
        else
            rm -rf "$TARGET_DIR"
            mv "$BACKUP_DIR/robot_setup" "$TARGET_DIR"
            rmdir "$BACKUP_DIR"
        fi
    elif [[ -n "$BACKUP_DIR" ]]; then
        rmdir "$BACKUP_DIR"
    fi
}
trap cleanup EXIT

tar -xzf "$ARCHIVE" -C "$EXTRACT_DIR"
if [[ ! -d "$EXTRACT_DIR/robot_setup" ]]; then
    echo 'Archive layout is invalid: robot_setup directory is missing.' >&2
    exit 1
fi

if [[ -d "$TARGET_DIR" ]]; then
    BACKUP_DIR="$(mktemp -d "$HOME/.robot-setup-backup.XXXXXX")"
    mv "$TARGET_DIR" "$BACKUP_DIR/robot_setup"
    BACKUP_COMPLETE=true
fi
mv "$EXTRACT_DIR/robot_setup" "$TARGET_DIR"
if [[ -n "$BACKUP_DIR" ]]; then
    for state_file in robot_config.yaml robot_progress.json; do
        if [[ -f "$BACKUP_DIR/robot_setup/$state_file" ]]; then
            cp -a "$BACKUP_DIR/robot_setup/$state_file" "$TARGET_DIR/$state_file"
        fi
    done
fi
cd "$TARGET_DIR"
"$UV_BIN" sync --no-dev --frozen

if command -v xdg-user-dir >/dev/null 2>&1; then
    DESKTOP_DIR="$(xdg-user-dir DESKTOP)"
else
    DESKTOP_DIR="$HOME/桌面"
fi
if [[ -z "$DESKTOP_DIR" || "$DESKTOP_DIR" == "$HOME" ]]; then
    DESKTOP_DIR="$HOME/桌面"
fi
DESKTOP_FILE="$DESKTOP_DIR/robot_setup.desktop"
mkdir -p "$DESKTOP_DIR"
cat >"$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=Robot Setup
Comment=Robot commissioning wizard
Exec=$TARGET_DIR/start_robot_setup.sh
Icon=$TARGET_DIR/assets/robot_setup.svg
Terminal=false
Categories=Utility;
EOF
chmod +x "$DESKTOP_FILE"
INSTALL_COMMITTED=true

echo
echo "Installation complete. Start with: $TARGET_DIR/start_robot_setup.sh"
echo "Desktop launcher: $DESKTOP_FILE"
