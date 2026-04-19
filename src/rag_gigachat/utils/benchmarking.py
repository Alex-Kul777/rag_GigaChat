"""
benchmarking.py - Утилиты для benchmarking и сравнения runs

Позволяет:
- Сравнивать время между несколькими запусками
- Отслеживать регрессии производительности
- Генерировать отчеты benchmarking
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import statistics


class BenchmarkRun:
    """Один run benchmark"""

    def __init__(self, name: str, json_log_file: str):
        """
        Args:
            name: Имя run (например, "opt-125m", "qwen")
            json_log_file: Путь к JSON лог файлу
        """
        self.name = name
        self.json_log_file = json_log_file
        self.metrics = self._parse_logs()

    def _parse_logs(self) -> dict:
        """Парсить логи и извлечь метрики"""
        if not Path(self.json_log_file).exists():
            return {}

        logs = []
        with open(self.json_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    logs.append(json.loads(line))
                except:
                    continue

        # Группируем по stage и request_id
        metrics = {}
        for log in logs:
            stage = log.get('stage', 'UNKNOWN')
            action = log.get('action', '')

            if stage not in metrics:
                metrics[stage] = {}

            if action == 'START':
                metrics[stage]['start_time'] = log.get('timestamp')
                metrics[stage]['start_metrics'] = log.get('metrics', {})
            elif action == 'END':
                metrics[stage]['end_time'] = log.get('timestamp')
                metrics[stage]['end_metrics'] = log.get('metrics', {})
                metrics[stage]['duration_ms'] = log.get('metrics', {}).get('duration_ms', 0)

        return metrics

    def get_stage_time(self, stage: str) -> float:
        """Получить время этапа в миллисекундах"""
        if stage in self.metrics:
            return self.metrics[stage].get('duration_ms', 0)
        return 0

    def get_total_time(self) -> float:
        """Получить общее время всех этапов"""
        return sum(self.get_stage_time(s) for s in self.metrics.keys())

    def summary(self) -> dict:
        """Получить сводку по метрикам"""
        total_time = self.get_total_time()
        stages = {}
        for stage, metrics in self.metrics.items():
            stages[stage] = {
                'duration_ms': metrics.get('duration_ms', 0),
                'percent': (metrics.get('duration_ms', 0) / total_time * 100) if total_time > 0 else 0
            }

        return {
            'name': self.name,
            'total_ms': total_time,
            'stages': stages,
            'timestamp': datetime.now().isoformat()
        }


class BenchmarkComparator:
    """Сравнение нескольких benchmark runs"""

    def __init__(self, runs: List[BenchmarkRun]):
        self.runs = runs

    def compare_stage_times(self, stage: str) -> pd.DataFrame:
        """Сравнить время одного этапа между runs"""
        data = []
        for run in self.runs:
            data.append({
                'Run': run.name,
                'Stage': stage,
                'Time (ms)': run.get_stage_time(stage)
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df['Relative'] = df['Time (ms)'] / df['Time (ms)'].min()

        return df

    def compare_total_times(self) -> pd.DataFrame:
        """Сравнить общее время между runs"""
        data = []
        for run in self.runs:
            data.append({
                'Run': run.name,
                'Total Time (ms)': run.get_total_time()
            })

        df = pd.DataFrame(data)
        if not df.empty:
            baseline = df['Total Time (ms)'].min()
            df['Relative'] = df['Total Time (ms)'] / baseline
            df['Delta (%)'] = ((df['Total Time (ms)'] - baseline) / baseline * 100).round(2)

        return df

    def find_regressions(self, baseline_run: str = None, threshold_percent: float = 10.0) -> List[dict]:
        """Найти регрессии производительности"""
        if baseline_run is None:
            baseline_run = self.runs[0].name

        baseline = next((r for r in self.runs if r.name == baseline_run), None)
        if not baseline:
            return []

        regressions = []
        for run in self.runs:
            if run.name == baseline_run:
                continue

            for stage, baseline_metrics in baseline.metrics.items():
                if stage in run.metrics:
                    baseline_time = baseline_metrics.get('duration_ms', 0)
                    current_time = run.metrics[stage].get('duration_ms', 0)

                    if baseline_time > 0:
                        delta_percent = ((current_time - baseline_time) / baseline_time) * 100
                        if delta_percent > threshold_percent:
                            regressions.append({
                                'run': run.name,
                                'stage': stage,
                                'baseline_ms': baseline_time,
                                'current_ms': current_time,
                                'delta_percent': delta_percent
                            })

        return regressions

    def generate_report(self) -> str:
        """Генерировать текстовый отчет сравнения"""
        report = []
        report.append("=" * 60)
        report.append("BENCHMARK COMPARISON REPORT")
        report.append("=" * 60)

        # Общее время
        report.append("\n📊 TOTAL TIMES:")
        report.append("-" * 60)
        df_total = self.compare_total_times()
        report.append(df_total.to_string(index=False))

        # Этапы
        report.append("\n⏱️ STAGE BREAKDOWN:")
        report.append("-" * 60)

        all_stages = set()
        for run in self.runs:
            all_stages.update(run.metrics.keys())

        for stage in sorted(all_stages):
            report.append(f"\n{stage}:")
            df_stage = self.compare_stage_times(stage)
            if not df_stage.empty:
                report.append(df_stage.to_string(index=False))

        # Регрессии
        regressions = self.find_regressions()
        if regressions:
            report.append("\n⚠️ PERFORMANCE REGRESSIONS (>10%):")
            report.append("-" * 60)
            for reg in regressions:
                report.append(
                    f"  {reg['run']} - {reg['stage']}: "
                    f"{reg['baseline_ms']:.0f}ms → {reg['current_ms']:.0f}ms "
                    f"({reg['delta_percent']:+.1f}%)"
                )

        report.append("\n" + "=" * 60)
        return "\n".join(report)


class PerformanceAnalyzer:
    """Анализ производительности одного run"""

    def __init__(self, benchmark_run: BenchmarkRun):
        self.run = benchmark_run

    def get_bottleneck(self) -> Dict[str, any]:
        """Найти основной bottleneck"""
        metrics = self.run.metrics
        if not metrics:
            return {}

        bottleneck_stage = max(metrics.items(), key=lambda x: x[1].get('duration_ms', 0))
        total_time = self.run.get_total_time()

        return {
            'stage': bottleneck_stage[0],
            'duration_ms': bottleneck_stage[1].get('duration_ms', 0),
            'percent_of_total': (bottleneck_stage[1].get('duration_ms', 0) / total_time * 100) if total_time > 0 else 0
        }

    def get_recommendations(self) -> List[str]:
        """Получить рекомендации по оптимизации"""
        bottleneck = self.get_bottleneck()
        recommendations = []

        if not bottleneck:
            return []

        stage = bottleneck['stage']
        percent = bottleneck['percent_of_total']

        if percent > 50:
            if stage == 'GENERATION':
                recommendations.append("🤖 GENERATION занимает >50% времени:")
                recommendations.append("  • Используйте меньшую модель (facebook/opt-125m вместо Qwen)")
                recommendations.append("  • Уменьшите max_tokens (сейчас возможно 256+)")
                recommendations.append("  • Используйте квантование (8-bit или 4-bit)")
                recommendations.append("  • Включите кэширование токенов")
            elif stage == 'RETRIEVAL':
                recommendations.append("🔍 RETRIEVAL занимает >50% времени:")
                recommendations.append("  • Используйте более быстрый индекс (IVF вместо FLAT)")
                recommendations.append("  • Уменьшите k (количество документов)")
                recommendations.append("  • Используйте более быструю модель embedding")
                recommendations.append("  • Включите кэширование embedding")
            elif stage == 'EMBEDDING':
                recommendations.append("🔗 EMBEDDING занимает >50% времени:")
                recommendations.append("  • Используйте более быструю модель")
                recommendations.append("  • Включите батчинг документов")
                recommendations.append("  • Используйте GPU для ускорения")

        return recommendations

    def get_timeline_stats(self) -> dict:
        """Получить статистику по временной шкале"""
        metrics = self.run.metrics
        times = sorted([m.get('duration_ms', 0) for m in metrics.values() if m.get('duration_ms', 0) > 0])

        if not times:
            return {}

        return {
            'min_stage_ms': min(times),
            'max_stage_ms': max(times),
            'median_stage_ms': statistics.median(times),
            'mean_stage_ms': statistics.mean(times),
            'stdev_stage_ms': statistics.stdev(times) if len(times) > 1 else 0
        }
