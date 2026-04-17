#!/usr/bin/env python3
"""
backlog_index.py - Генератор INDEX.md для backlog/
Читает все BKL-*.md файлы и создаёт индекс с сортировкой по priority/severity.
"""
import re
from pathlib import Path
from typing import List, Dict
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKLOG_DIR = PROJECT_ROOT / "backlog"


def read_backlog_entries() -> List[Dict]:
    """Прочитать все BKL-*.md файлы и распарсить frontmatter"""
    entries = []
    backlog_files = sorted(BACKLOG_DIR.glob("BKL-*.md"))

    for file_path in backlog_files:
        try:
            with open(file_path, "r") as f:
                content = f.read()

            # Извлечь YAML frontmatter
            match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
            if match:
                frontmatter = yaml.safe_load(match.group(1))
                frontmatter["file"] = file_path.name
                entries.append(frontmatter)
        except Exception as e:
            print(f"⚠️  Error reading {file_path.name}: {e}")

    return entries


def sort_entries(entries: List[Dict]) -> List[Dict]:
    """Сортировать записи по priority DESC, severity DESC, created ASC"""
    priority_order = {"high": 3, "medium": 2, "low": 1}
    severity_order = {"critical": 3, "major": 2, "minor": 1}

    def sort_key(entry):
        priority = priority_order.get(entry.get("priority", "low"), 0)
        severity = severity_order.get(entry.get("severity", "minor"), 0)
        created = entry.get("created", "9999-12-31")
        return (-priority, -severity, created)

    return sorted(entries, key=sort_key)


def generate_index(entries: List[Dict]) -> str:
    """Сгенерировать INDEX.md"""
    index = []
    index.append("# Backlog Index\n")
    index.append(f"**Total items**: {len(entries)}\n")

    # Группировка по статусу
    by_status = {}
    for entry in entries:
        status = entry.get("status", "open")
        if status not in by_status:
            by_status[status] = []
        by_status[status].append(entry)

    status_order = ["open", "in-progress", "blocked", "done", "rejected"]

    for status in status_order:
        if status not in by_status:
            continue

        status_entries = by_status[status]
        status_emoji = {
            "open": "🔴",
            "in-progress": "🟡",
            "blocked": "❌",
            "done": "✅",
            "rejected": "🚫",
        }
        emoji = status_emoji.get(status, "❓")

        index.append(f"\n## {emoji} {status.upper()} ({len(status_entries)})\n")

        for entry in sorted(
            status_entries,
            key=lambda e: (
                -{"high": 3, "medium": 2, "low": 1}.get(e.get("priority"), 0),
                -{"critical": 3, "major": 2, "minor": 1}.get(e.get("severity"), 0),
                e.get("created", ""),
            ),
        ):
            bkl_id = entry.get("id", "UNKNOWN")
            title = entry.get("title", "No title")
            priority = entry.get("priority", "—")
            severity = entry.get("severity", "—")
            file = entry.get("file", "—")

            index.append(f"- **{bkl_id}**: {title}\n")
            index.append(f"  - File: `{file}`\n")
            index.append(f"  - Priority: {priority} | Severity: {severity}\n")

    return "".join(index)


def main():
    BACKLOG_DIR.mkdir(parents=True, exist_ok=True)

    entries = read_backlog_entries()
    if not entries:
        print("⚠️  No backlog entries found")
        index_content = "# Backlog Index\n\nNo backlog entries yet.\n"
    else:
        entries = sort_entries(entries)
        index_content = generate_index(entries)

    index_path = BACKLOG_DIR / "INDEX.md"
    with open(index_path, "w") as f:
        f.write(index_content)

    print(f"✅ Generated {index_path}")
    print(f"📊 Total entries: {len(entries)}")


if __name__ == "__main__":
    main()
