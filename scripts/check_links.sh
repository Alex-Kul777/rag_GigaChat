#!/bin/bash

# Script to check for broken markdown links in documentation
# Scans all .md files (excluding .venv/) for [text](path.md) links
# Reports any links to .md files that don't exist

echo "=== Checking Markdown Links ==="

broken=0
checked=0
project_root=$(pwd)

for file in $(find . -name "*.md" -type f -not -path "./.venv/*"); do
    # Extract all markdown links: [text](path.md)
    while IFS= read -r line; do
        [ -z "$line" ] && continue

        # Extract path from [text](path.md) format
        target=$(echo "$line" | sed -n 's/.*(\([^)]*\.md\)).*/\1/p')
        [ -z "$target" ] && continue

        checked=$((checked + 1))

        # Get file directory for relative path resolution
        file_dir=$(dirname "$file")

        # Resolve relative path from file's directory
        if [[ "$target" == /* ]]; then
            # Absolute path
            target_path="$target"
        else
            # Relative path
            target_path="$file_dir/$target"
            target_path=$(cd "$project_root" && cd "$(dirname "$target_path")" 2>/dev/null && pwd)/$(basename "$target_path")
            target_path="${target_path#$project_root/}"
        fi

        # Check if file exists
        if [ ! -f "$project_root/$target_path" ]; then
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
