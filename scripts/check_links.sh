#!/bin/bash

# Script to check for broken markdown links in documentation
# Scans all .md files (excluding .venv/) for [text](path.md) links
# Reports any links to .md files that don't exist

set -e

echo "=== Checking Markdown Links ==="

broken=0
checked=0

for file in $(find . -name "*.md" -type f -not -path "./.venv/*" 2>/dev/null || true); do
    [ ! -f "$file" ] && continue

    # Extract all markdown links: [text](path.md)
    while IFS= read -r line; do
        [ -z "$line" ] && continue

        # Extract path from [text](path.md) format
        target=$(echo "$line" | sed -n 's/.*(\([^)]*\.md\)).*/\1/p')
        [ -z "$target" ] && continue

        checked=$((checked + 1))

        # Get absolute path of current file
        file_abs=$(cd "$(dirname "$file")" && pwd)/$(basename "$file")
        file_dir=$(dirname "$file_abs")

        # Resolve target path relative to file directory
        if [[ "$target" = /* ]]; then
            # Absolute path - check from repo root
            target_path="$target"
        else
            # Relative path - resolve from file's directory
            target_path="$file_dir/$target"
            # Normalize path (remove .. and .)
            target_path=$(python3 -c "import os; print(os.path.normpath('$target_path'))" 2>/dev/null || echo "$target_path")
        fi

        # Check if file exists
        if [ ! -f "$target_path" ]; then
            echo "❌ $file -> $target"
            broken=$((broken + 1))
        fi
    done < <(grep -oP '\[.*?\]\([^)]*\.md\)' "$file" 2>/dev/null || true)
done

echo ""
echo "=== Results ==="
echo "Checked: $checked links"
echo "Broken: $broken links"

if [ $broken -eq 0 ]; then
    echo "✅ All links are valid!"
    exit 0
else
    echo "❌ Found broken links"
    exit 1
fi
