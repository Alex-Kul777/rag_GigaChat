#!/usr/bin/env python3
"""
mine_process.py - Process Miner для анализа логов и выявления проблем
Генерирует отчёт с вариантами трасс, bottlenecks, ошибками и аномалиями.
"""
import csv
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"


def find_latest_event_log() -> Path:
    """Найти последний events_*.csv файл"""
    event_logs = list(LOGS_DIR.glob("events_*.csv"))
    if not event_logs:
        raise FileNotFoundError("❌ No events_*.csv files found in logs/")
    latest = max(event_logs, key=lambda p: p.stat().st_mtime)
    logger.info(f"📊 Found event log: {latest}")
    return latest


def read_events(csv_path: Path) -> List[dict]:
    """Прочитать события из CSV"""
    events = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        events = list(reader)
    logger.info(f"✅ Read {len(events)} events")
    return events


def extract_variants(events: List[dict]) -> Dict[str, int]:
    """Извлечь варианты трасс (последовательности активностей)"""
    traces_by_case = defaultdict(list)

    for event in events:
        case_id = event["case_id"]
        activity = event["activity"]
        traces_by_case[case_id].append(activity)

    # Сгруппировать по трассам
    variants = Counter()
    for trace in traces_by_case.values():
        variant_key = " → ".join(trace)
        variants[variant_key] += 1

    return variants


def compute_bottlenecks(events: List[dict]) -> Dict[str, dict]:
    """Вычислить bottlenecks (p50, p95, p99) по активностям"""
    durations_by_activity = defaultdict(list)

    for event in events:
        activity = event["activity"]
        duration = float(event["duration_ms"])
        durations_by_activity[activity].append(duration)

    bottlenecks = {}
    for activity, durations in durations_by_activity.items():
        durations.sort()
        n = len(durations)
        bottlenecks[activity] = {
            "count": n,
            "p50": durations[int(n * 0.5)],
            "p95": durations[int(n * 0.95)],
            "p99": durations[int(n * 0.99)],
            "max": durations[-1],
        }

    return bottlenecks


def extract_errors(events: List[dict]) -> List[dict]:
    """Извлечь ошибки с контекстом"""
    error_events = [e for e in events if e["status"] == "error"]
    errors_with_context = []

    for error_event in error_events:
        case_id = error_event["case_id"]
        # Найти предыдущие события в том же case
        context_events = [
            e for e in events if e["case_id"] == case_id
        ]
        errors_with_context.append({
            "case_id": case_id,
            "activity": error_event["activity"],
            "timestamp": error_event["timestamp"],
            "resource": error_event["resource"],
            "attributes": error_event["attributes"],
            "context": [e["activity"] for e in context_events[:5]],
        })

    return errors_with_context


def detect_anomalies(
    variants: Dict[str, int],
    bottlenecks: Dict[str, dict],
    errors: List[dict],
) -> List[str]:
    """Обнаружить аномалии в логах"""
    anomalies = []
    total_traces = sum(variants.values())

    # 1. Редкие варианты (≤2 случая при N≥10 всего)
    rare_variants = [
        (v, cnt) for v, cnt in variants.items() if cnt <= 2 and total_traces >= 10
    ]
    if rare_variants:
        anomalies.append(f"🔴 **Rare variants** ({len(rare_variants)}):")
        for variant, count in rare_variants[:3]:
            anomalies.append(f"   - {variant} (frequency: {count})")

    # 2. Bimodal distribution (p95/p50 > 3)
    bimodal_activities = [
        (act, metrics)
        for act, metrics in bottlenecks.items()
        if metrics["p50"] > 0 and metrics["p95"] / metrics["p50"] > 3
    ]
    if bimodal_activities:
        anomalies.append(f"🔴 **Bimodal distributions** ({len(bimodal_activities)}):")
        for activity, metrics in bimodal_activities[:3]:
            ratio = metrics["p95"] / metrics["p50"]
            anomalies.append(
                f"   - {activity}: p95/p50 = {ratio:.1f}x "
                f"(p50={metrics['p50']:.0f}ms, p95={metrics['p95']:.0f}ms)"
            )

    # 3. Высокий процент ошибок (>5%)
    error_activities = defaultdict(lambda: {"error": 0, "total": 0})
    for event in bottlenecks.keys():
        error_activities[event]["total"] += bottlenecks[event]["count"]
    for error in errors:
        error_activities[error["activity"]]["error"] += 1

    high_error_rate = [
        (act, metrics["error"] / metrics["total"])
        for act, metrics in error_activities.items()
        if metrics["total"] > 0 and metrics["error"] / metrics["total"] > 0.05
    ]
    if high_error_rate:
        anomalies.append(f"🔴 **High error rates** ({len(high_error_rate)}):")
        for activity, error_rate in high_error_rate[:3]:
            anomalies.append(f"   - {activity}: {error_rate:.1%} errors")

    return anomalies


def generate_summary(
    variants: Dict[str, int],
    bottlenecks: Dict[str, dict],
    errors: List[dict],
) -> str:
    """Сгенерировать сводку в Markdown"""
    anomalies = detect_anomalies(variants, bottlenecks, errors)

    summary = []
    summary.append("# Process Mining Summary\n")
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Variants
    summary.append("## 1. Variants (Activity Sequences)\n")
    for i, (variant, count) in enumerate(
        sorted(variants.items(), key=lambda x: -x[1])[:5], 1
    ):
        summary.append(f"**V{i}** ({count}×): {variant}\n")

    if len(variants) > 5:
        rare_count = sum(1 for cnt in variants.values() if cnt <= 2)
        summary.append(f"... + {rare_count} rare variants (≤2 occurrences)\n")

    # 2. Bottlenecks
    summary.append("\n## 2. Performance Bottlenecks\n")
    summary.append("| Activity | Count | p50 | p95 | p99 | Max |\n")
    summary.append("|----------|-------|-----|-----|-----|-----|\n")

    for activity, metrics in sorted(
        bottlenecks.items(), key=lambda x: -x[1]["p95"]
    )[:10]:
        summary.append(
            f"| {activity} | {metrics['count']} | "
            f"{metrics['p50']:.0f}ms | {metrics['p95']:.0f}ms | "
            f"{metrics['p99']:.0f}ms | {metrics['max']:.0f}ms |\n"
        )

    # 3. Errors
    if errors:
        summary.append(f"\n## 3. Errors ({len(errors)} total)\n")
        for error in errors[:5]:
            summary.append(
                f"- **{error['activity']}** in case {error['case_id']}: "
                f"{error['attributes']}\n"
            )
        if len(errors) > 5:
            summary.append(f"... + {len(errors) - 5} more errors\n")

    # 4. Anomalies
    if anomalies:
        summary.append("\n## 4. Anomalies Detected\n")
        for anomaly in anomalies:
            summary.append(f"{anomaly}\n")
    else:
        summary.append("\n## 4. Anomalies\nNo anomalies detected ✅\n")

    return "".join(summary)


def main():
    try:
        event_log = find_latest_event_log()
        events = read_events(event_log)

        logger.info("🔄 Extracting variants...")
        variants = extract_variants(events)

        logger.info("📊 Computing bottlenecks...")
        bottlenecks = compute_bottlenecks(events)

        logger.info("❌ Extracting errors...")
        errors = extract_errors(events)

        logger.info("🔍 Detecting anomalies...")
        summary = generate_summary(variants, bottlenecks, errors)

        # Save summary
        summary_path = LOGS_DIR / "last_session_summary.md"
        with open(summary_path, "w") as f:
            f.write(summary)

        logger.info(f"✅ Summary saved to {summary_path}")
        print(summary)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
