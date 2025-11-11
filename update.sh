#!/usr/bin/env bash
set -euo pipefail

# Always operate from the repository root (parent directory of this script).
repo_root="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
cd "$repo_root"

# Ensure UTF-8 encodings for commit metadata and logs.
git config i18n.commitEncoding utf-8
git config i18n.logOutputEncoding utf-8

echo "=== git pull origin main ==="
git pull origin main
echo

echo "=== git status ==="
git status
echo

echo "=== git add . ==="
git add .
echo

read -r -p "commit message (please type in Chinese or English): " message

if [[ -z "${message// }" ]]; then
    echo "empty message, abort."
    exit 1
fi

echo "=== git commit ==="
git commit -m "$message"
echo

echo "=== git push origin main ==="
git push origin main
echo

echo "upload done."
# 终端输入：bash /home/yangdaoone/code/Python/update.sh