#!/bin/bash
# post_edit_pytest.sh - Auto-run pytest after Edit/Write in src/
# This hook is triggered by Claude Code after file modifications

set -e

CHANGED_FILE="$1"

# Only run pytest if changes are in src/ or tests/
if [[ "$CHANGED_FILE" == src/* ]] || [[ "$CHANGED_FILE" == tests/* ]]; then
    echo "🔍 Running pytest after edit: $CHANGED_FILE"
    .venv/bin/python -m pytest tests/unit/ -q --tb=short || true
fi
