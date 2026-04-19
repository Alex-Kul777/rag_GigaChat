"""
metrics_dashboard.py - Streamlit dashboard для анализа логов и метрик RAG pipeline

Визуализирует:
- Время по этапам
- Памяти per-stage
- Request ID трассировка
- Анализ bottleneck
- Экспорт в Excel
"""
import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime


def load_json_logs(log_file: str = "logs/rag_app.json") -> pd.DataFrame:
    """Загрузить JSON логи в DataFrame"""
    if not Path(log_file).exists():
        return pd.DataFrame()

    logs = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except:
                continue

    return pd.DataFrame(logs)


def get_stage_metrics(df: pd.DataFrame) -> dict:
    """Извлечь метрики по этапам из логов"""
    metrics = {}

    # Группируем по stage и request_id
    for (stage, request_id), group in df.groupby(['stage', 'request_id']):
        start = group[group['action'] == 'START']
        end = group[group['action'] == 'END']

        if not start.empty and not end.empty:
            start_time = pd.to_datetime(start.iloc[0]['timestamp'])
            end_time = pd.to_datetime(end.iloc[0]['timestamp'])
            duration = (end_time - start_time).total_seconds() * 1000

            if stage not in metrics:
                metrics[stage] = {
                    'count': 0,
                    'total_ms': 0,
                    'min_ms': float('inf'),
                    'max_ms': 0,
                    'avg_ms': 0,
                }

            metrics[stage]['count'] += 1
            metrics[stage]['total_ms'] += duration
            metrics[stage]['min_ms'] = min(metrics[stage]['min_ms'], duration)
            metrics[stage]['max_ms'] = max(metrics[stage]['max_ms'], duration)

        if stage in metrics:
            metrics[stage]['avg_ms'] = metrics[stage]['total_ms'] / metrics[stage]['count']

    return metrics


def create_timeline_chart(df: pd.DataFrame) -> go.Figure:
    """Создать временную шкалу для визуализации этапов"""
    if df.empty:
        return go.Figure()

    # Получаем request_id с максимальным количеством логов
    request_id = df['request_id'].value_counts().index[0] if 'request_id' in df.columns else None

    if request_id:
        df_request = df[df['request_id'] == request_id].copy()
    else:
        df_request = df.copy()

    df_request['timestamp'] = pd.to_datetime(df_request['timestamp'])
    df_request = df_request.sort_values('timestamp')

    fig = go.Figure()

    stages = df_request['stage'].unique()
    colors = px.colors.qualitative.Plotly

    for i, stage in enumerate(stages):
        stage_data = df_request[df_request['stage'] == stage]
        for _, row in stage_data.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['timestamp'], row['timestamp']],
                y=[stage, stage],
                mode='markers+text',
                marker=dict(size=12, color=colors[i % len(colors)]),
                text=f"{row['action']}: {row['message'][:30]}",
                name=stage,
                showlegend=(stage not in [t.name for t in fig.data])
            ))

    fig.update_layout(
        title=f"Pipeline Timeline для Request ID: {request_id}",
        xaxis_title="Время",
        yaxis_title="Этап",
        hovermode='closest',
        height=400
    )

    return fig


def create_duration_chart(metrics: dict) -> go.Figure:
    """Создать столбчатую диаграмму времени по этапам"""
    if not metrics:
        return go.Figure()

    stages = list(metrics.keys())
    avg_times = [metrics[s]['avg_ms'] for s in stages]

    fig = go.Figure(data=[
        go.Bar(
            x=stages,
            y=avg_times,
            marker=dict(
                color=avg_times,
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Время (ms)")
            ),
            text=[f"{t:.0f}ms" for t in avg_times],
            textposition='outside'
        )
    ])

    fig.update_layout(
        title="Среднее время по этапам",
        xaxis_title="Этап",
        yaxis_title="Время (ms)",
        height=400
    )

    return fig


def create_pie_chart(metrics: dict) -> go.Figure:
    """Создать круговую диаграмму распределения времени"""
    if not metrics:
        return go.Figure()

    stages = list(metrics.keys())
    times = [metrics[s]['avg_ms'] for s in stages]

    fig = go.Figure(data=[go.Pie(
        labels=stages,
        values=times,
        text=[f"{t:.0f}ms" for t in times],
        textposition='inside'
    )])

    fig.update_layout(
        title="Распределение времени по этапам",
        height=400
    )

    return fig


def main():
    st.set_page_config(page_title="RAG Pipeline Metrics", layout="wide")
    st.title("📊 RAG Pipeline Metrics Dashboard")

    # Боковая панель для управления
    st.sidebar.title("⚙️ Параметры")
    log_file = st.sidebar.text_input("JSON лог файл", "logs/rag_app.json")

    # Загружаем логи
    df = load_json_logs(log_file)

    if df.empty:
        st.warning("❌ JSON логи не найдены. Запустите RAG pipeline сначала.")
        st.info("Команда: `RAG_DEBUG_MODE=true python app.py --mode query --query 'Что такое RAG?'`")
        return

    st.success(f"✅ Загружено {len(df)} логов")

    # Вкладки для различных анализов
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Временная шкала",
        "⏱️ Длительность этапов",
        "📊 Распределение",
        "🔍 Детали"
    ])

    with tab1:
        st.subheader("Временная шкала выполнения")
        fig_timeline = create_timeline_chart(df)
        st.plotly_chart(fig_timeline, use_container_width=True)

    with tab2:
        st.subheader("Среднее время по этапам")
        metrics = get_stage_metrics(df)
        fig_duration = create_duration_chart(metrics)
        st.plotly_chart(fig_duration, use_container_width=True)

        # Таблица деталей
        st.subheader("Детали по этапам")
        details = []
        for stage, m in metrics.items():
            details.append({
                'Этап': stage,
                'Количество': m['count'],
                'Мин (ms)': f"{m['min_ms']:.0f}",
                'Макс (ms)': f"{m['max_ms']:.0f}",
                'Среднее (ms)': f"{m['avg_ms']:.0f}",
                'Всего (ms)': f"{m['total_ms']:.0f}"
            })
        st.dataframe(pd.DataFrame(details), use_container_width=True)

    with tab3:
        st.subheader("Распределение времени")
        metrics = get_stage_metrics(df)
        fig_pie = create_pie_chart(metrics)
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab4:
        st.subheader("Полные логи")

        # Фильтры
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_stage = st.multiselect("Фильтр по этапу", df['stage'].unique())
        with col2:
            selected_action = st.multiselect("Фильтр по действию", df['action'].unique())
        with col3:
            selected_level = st.multiselect("Фильтр по уровню", df['level'].unique())

        # Применяем фильтры
        filtered_df = df.copy()
        if selected_stage:
            filtered_df = filtered_df[filtered_df['stage'].isin(selected_stage)]
        if selected_action:
            filtered_df = filtered_df[filtered_df['action'].isin(selected_action)]
        if selected_level:
            filtered_df = filtered_df[filtered_df['level'].isin(selected_level)]

        # Отображаем логи
        st.dataframe(
            filtered_df[['timestamp', 'level', 'stage', 'action', 'message', 'request_id']],
            use_container_width=True,
            height=400
        )

        # Экспорт в Excel
        st.subheader("📥 Экспорт метрик")
        if st.button("Сохранить в Excel"):
            try:
                excel_file = "metrics_export.xlsx"
                with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='Логи', index=False)

                    # Добавляем лист с метриками
                    metrics = get_stage_metrics(df)
                    metrics_df = pd.DataFrame([
                        {
                            'Этап': stage,
                            'Количество': m['count'],
                            'Среднее (ms)': m['avg_ms'],
                            'Мин (ms)': m['min_ms'],
                            'Макс (ms)': m['max_ms']
                        }
                        for stage, m in metrics.items()
                    ])
                    metrics_df.to_excel(writer, sheet_name='Метрики', index=False)

                st.success(f"✅ Экспортировано в {excel_file}")
                with open(excel_file, 'rb') as f:
                    st.download_button(
                        label="Скачать Excel файл",
                        data=f.read(),
                        file_name=f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    )
            except Exception as e:
                st.error(f"❌ Ошибка экспорта: {e}")


if __name__ == "__main__":
    main()
