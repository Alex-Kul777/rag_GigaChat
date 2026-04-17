"""
components.py - Переиспользуемые UI компоненты для Streamlit
Содержит ConfigModal, FileListPanel, DocumentViewer, HighlightedAnswer
"""

import streamlit as st
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ConfigModal:
    """
    Модальное окно с расширенными настройками.
    Открывается по кнопке "⚙️ Расширенные настройки"
    """

    @staticmethod
    def show():
        """Показать модальное окно настроек"""
        if st.button("⚙️ Расширенные настройки", key="config_button"):
            st.session_state.show_config_modal = True

        # Использование st.dialog для модального окна (Streamlit >= 1.30)
        if st.session_state.get("show_config_modal", False):
            with st.dialog("Расширенные настройки", width="large"):
                ConfigModal._render_content()

    @staticmethod
    def _render_content():
        """Рендерить содержимое модального окна"""

        # 1️⃣ Группа "Модели"
        st.subheader("🤖 Модели")
        col1, col2 = st.columns(2)

        with col1:
            llm_model = st.text_input(
                "LLM модель",
                value=st.session_state.get("llm_model", "GigaChat-2-Max"),
                help="Название модели для генерации ответов"
            )
            st.session_state.llm_model = llm_model

        with col2:
            embedding_model = st.text_input(
                "Embedding модель",
                value=st.session_state.get("embedding_model", "GigaChat-2-Max"),
                help="Модель для создания эмбеддингов"
            )
            st.session_state.embedding_model = embedding_model

        col3, col4 = st.columns(2)
        with col3:
            max_tokens = st.slider(
                "Max tokens для генерации",
                min_value=100,
                max_value=4000,
                value=st.session_state.get("max_tokens", 2000),
                step=100
            )
            st.session_state.max_tokens = max_tokens

        with col4:
            temperature = st.slider(
                "Temperature (творчество)",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get("temperature", 0.7),
                step=0.1
            )
            st.session_state.temperature = temperature

        # 2️⃣ Группа "Поиск"
        st.divider()
        st.subheader("🔍 Поиск")
        col5, col6 = st.columns(2)

        with col5:
            k_retrieve = st.slider(
                "Top-K документов",
                min_value=1,
                max_value=20,
                value=st.session_state.get("k_retrieve", 5),
                help="Количество релевантных документов для поиска"
            )
            st.session_state.k_retrieve = k_retrieve

        with col6:
            max_context = st.slider(
                "Max контекст (символов)",
                min_value=500,
                max_value=5000,
                value=st.session_state.get("max_context", 2000),
                step=100
            )
            st.session_state.max_context = max_context

        retrieval_type = st.radio(
            "Тип поиска",
            options=["dense", "sparse", "hybrid"],
            horizontal=True,
            index=["dense", "sparse", "hybrid"].index(
                st.session_state.get("retrieval_type", "hybrid")
            )
        )
        st.session_state.retrieval_type = retrieval_type

        # 3️⃣ Группа "Чанкирование"
        st.divider()
        st.subheader("📄 Чанкирование")
        col7, col8 = st.columns(2)

        with col7:
            chunk_size = st.slider(
                "Размер чанка",
                min_value=100,
                max_value=2000,
                value=st.session_state.get("chunk_size", 500),
                step=50
            )
            st.session_state.chunk_size = chunk_size

        with col8:
            chunk_overlap = st.slider(
                "Перекрытие чанков",
                min_value=0,
                max_value=500,
                value=st.session_state.get("chunk_overlap", 80),
                step=10
            )
            st.session_state.chunk_overlap = chunk_overlap

        # 4️⃣ Группа "GigaChat"
        st.divider()
        st.subheader("💬 GigaChat")

        col9, col10 = st.columns(2)
        with col9:
            top_p = st.slider(
                "Top-P (разнообразие)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("top_p", 0.9),
                step=0.05
            )
            st.session_state.top_p = top_p

        with col10:
            repeat_penalty = st.slider(
                "Penalty повторений",
                min_value=1.0,
                max_value=2.0,
                value=st.session_state.get("repeat_penalty", 1.1),
                step=0.1
            )
            st.session_state.repeat_penalty = repeat_penalty

        ocr_enabled = st.checkbox(
            "Включить OCR для сканированных PDF",
            value=st.session_state.get("ocr_enabled", True)
        )
        st.session_state.ocr_enabled = ocr_enabled

        # Кнопки действий
        st.divider()
        col_apply, col_reset, col_close = st.columns(3)

        with col_apply:
            if st.button("✅ Применить", use_container_width=True):
                st.success("✓ Настройки применены!")
                st.session_state.show_config_modal = False
                st.rerun()

        with col_reset:
            if st.button("🔄 Сброс", use_container_width=True):
                ConfigModal._reset_defaults()
                st.info("Настройки сброшены на значения по умолчанию")
                st.rerun()

        with col_close:
            if st.button("✕ Закрыть", use_container_width=True):
                st.session_state.show_config_modal = False
                st.rerun()

    @staticmethod
    def _reset_defaults():
        """Сбросить настройки на значения по умолчанию"""
        defaults = {
            "llm_model": "GigaChat-2-Max",
            "embedding_model": "GigaChat-2-Max",
            "max_tokens": 2000,
            "temperature": 0.7,
            "k_retrieve": 5,
            "max_context": 2000,
            "retrieval_type": "hybrid",
            "chunk_size": 500,
            "chunk_overlap": 80,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "ocr_enabled": True,
        }
        for key, value in defaults.items():
            st.session_state[key] = value


class FileListPanel:
    """
    Правый сайдбар (ширина ~300px) со списком файлов
    Содержит поиск по имени файла и кнопку обновления индекса
    """

    @staticmethod
    def show(documents_dirs: Dict[str, Path]):
        """
        Показать панель списка файлов

        Args:
            documents_dirs: Dict с доменами и путями к документам
        """
        with st.sidebar:
            st.markdown("---")
            st.subheader("📁 Документы")

            # Выбор домена
            selected_domain = st.selectbox(
                "Домен",
                options=list(documents_dirs.keys()),
                key="selected_domain"
            )

            domain_path = documents_dirs.get(selected_domain)
            if not domain_path:
                st.error(f"❌ Домен '{selected_domain}' не найден")
                return

            # Поиск по файлам
            search_query = st.text_input(
                "🔍 Поиск файла",
                placeholder="например: документ.pdf",
                key="file_search"
            )

            # Получить список PDF файлов
            pdf_files = FileListPanel._get_pdf_files(domain_path, search_query)

            if not pdf_files:
                st.info("📭 Файлы не найдены. Обновите индекс.")
            else:
                # Отобразить файлы как чекбоксы или кнопки
                st.write(f"📊 Найдено: **{len(pdf_files)}** файлов")

                # Инициализировать selected_files в session_state если нет
                if "selected_files" not in st.session_state:
                    st.session_state.selected_files = []

                for file_path in pdf_files:
                    col1, col2 = st.columns([3, 1], gap="small")

                    with col1:
                        # При клике на файл - открыть в DocumentViewer
                        if st.button(
                            f"📄 {file_path.stem}",
                            key=f"file_{file_path.stem}",
                            use_container_width=True
                        ):
                            st.session_state.selected_file = str(file_path)
                            st.session_state.show_document_viewer = True
                            st.rerun()

                    with col2:
                        # Показать количество страниц (если известно)
                        try:
                            import PyPDF2
                            with open(file_path, 'rb') as f:
                                reader = PyPDF2.PdfReader(f)
                                pages = len(reader.pages)
                                st.caption(f"{pages}p")
                        except Exception:
                            st.caption("?p")

            # Кнопка обновления индекса
            st.markdown("---")
            col_refresh, col_clear = st.columns(2)

            with col_refresh:
                if st.button("🔄 Обновить индекс", use_container_width=True):
                    st.session_state.force_reload_index = True
                    st.success("✓ Индекс будет обновлён...")
                    st.rerun()

            with col_clear:
                if st.button("🗑️ Очистить", use_container_width=True):
                    st.session_state.selected_files = []
                    st.session_state.selected_file = None
                    st.rerun()

            # Статистика
            st.markdown("---")
            st.caption(f"📂 Путь: `{domain_path.name}`")

    @staticmethod
    def _get_pdf_files(domain_path: Path, search_query: str = "") -> List[Path]:
        """
        Получить список PDF файлов из директории

        Args:
            domain_path: Путь к директории с документами
            search_query: Строка для поиска

        Returns:
            Список путей к PDF файлам
        """
        if not domain_path.exists():
            return []

        pdf_files = sorted(domain_path.rglob("*.pdf"))

        if search_query:
            pdf_files = [
                f for f in pdf_files
                if search_query.lower() in f.name.lower()
            ]

        return pdf_files


class DocumentViewer:
    """
    Вкладка/Модальное окно для просмотра PDF документа
    Показывает PDF через iframe и поддерживает переход на нужную страницу
    """

    @staticmethod
    def show(file_path: Optional[str] = None, page: int = 1):
        """
        Показать просмотр PDF документа

        Args:
            file_path: Путь к PDF файлу
            page: Номер страницы для открытия
        """
        if not file_path or not Path(file_path).exists():
            st.error("❌ Файл не найден")
            return

        file_path = Path(file_path)

        # Заголовок с информацией о файле
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"### 📄 {file_path.stem}")

        with col2:
            # Выбор страницы
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    total_pages = len(reader.pages)

                    selected_page = st.number_input(
                        "Страница",
                        min_value=1,
                        max_value=total_pages,
                        value=page,
                        key="doc_page_input"
                    )
            except Exception as e:
                st.warning(f"⚠️ Ошибка чтения PDF: {e}")
                total_pages = 1
                selected_page = 1

        with col3:
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            st.caption(f"{file_size_mb:.1f} MB")

        # Показать PDF через iframe (используем blob URL)
        st.markdown("---")
        DocumentViewer._render_pdf(file_path, selected_page)

        # Информация о документе
        st.markdown("---")
        with st.expander("📋 Информация о документе"):
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.metric("Размер файла", f"{file_size_mb:.1f} MB")
                st.metric("Всего страниц", total_pages)

            with col_info2:
                st.metric("Путь", file_path.parent.name)
                st.metric("Создан", file_path.stat().st_ctime)

    @staticmethod
    def _render_pdf(file_path: Path, page: int = 1):
        """
        Рендерить PDF в HTML с помощью PDF.js

        Args:
            file_path: Путь к PDF файлу
            page: Номер страницы (1-indexed)
        """
        # Читаем PDF как base64
        import base64

        try:
            with open(file_path, 'rb') as f:
                pdf_data = base64.b64encode(f.read()).decode()

            # HTML с PDF.js для отображения
            pdf_display = f"""
            <html>
            <head>
                <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
                <style>
                    body {{
                        margin: 0;
                        padding: 10px;
                        background: #f5f5f5;
                        font-family: Arial, sans-serif;
                    }}
                    #pdf-container {{
                        background: white;
                        box-shadow: 0 0 10px rgba(0,0,0,0.1);
                        border-radius: 8px;
                        padding: 20px;
                        max-width: 100%;
                        margin: 0 auto;
                    }}
                    canvas {{
                        max-width: 100%;
                        display: block;
                        margin: 0 auto;
                    }}
                    .error {{
                        color: #d32f2f;
                        padding: 20px;
                        background: #ffebee;
                        border-radius: 4px;
                    }}
                </style>
            </head>
            <body>
                <div id="pdf-container">
                    <canvas id="pdf-canvas"></canvas>
                </div>
                <script>
                    // Инициализация PDF.js worker
                    pdfjsLib.GlobalWorkerOptions.workerSrc =
                        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

                    const pdfData = atob('{pdf_data}');
                    const pdfBinary = new Uint8Array(pdfData.length);
                    for (let i = 0; i < pdfData.length; i++) {{
                        pdfBinary[i] = pdfData.charCodeAt(i);
                    }}

                    // Загрузить PDF
                    pdfjsLib.getDocument({{data: pdfBinary}})
                        .promise.then(pdf => {{
                            const pageNum = {page};
                            const actualPage = Math.min(pageNum, pdf.numPages);

                            pdf.getPage(actualPage).then(page => {{
                                const scale = 1.5;
                                const viewport = page.getViewport({{scale: scale}});

                                const canvas = document.getElementById('pdf-canvas');
                                const context = canvas.getContext('2d');
                                canvas.height = viewport.height;
                                canvas.width = viewport.width;

                                const renderContext = {{
                                    canvasContext: context,
                                    viewport: viewport
                                }};

                                page.render(renderContext).promise.then(() => {{
                                    console.log('PDF страница {page} отрисована');
                                }});
                            }});
                        }})
                        .catch(error => {{
                            document.getElementById('pdf-container').innerHTML =
                                '<div class="error">❌ Ошибка при загрузке PDF: ' + error.message + '</div>';
                        }});
                </script>
            </body>
            </html>
            """

            st.components.v1.html(pdf_display, height=800, scrolling=True)

        except Exception as e:
            st.error(f"❌ Ошибка при отображении PDF: {e}")


class HighlightedAnswer:
    """
    Компонент для отображения ответа с подсветкой источников
    Показывает ответ с ссылками вида "[Источник: file.pdf, стр. 5]"
    """

    @staticmethod
    def show(
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        documents_dirs: Dict[str, Path],
        show_sources: bool = True
    ):
        """
        Показать ответ с подсветкой источников

        Args:
            answer: Текст ответа от LLM
            retrieved_docs: Список найденных документов с метаданными
            documents_dirs: Dict с доменами и путями к документам
            show_sources: Показывать ли список источников
        """

        # Основной ответ
        st.markdown("### 🤖 Ответ")

        # Обработать ответ и добавить ссылки на источники
        processed_answer = HighlightedAnswer._process_answer_with_links(
            answer,
            retrieved_docs
        )

        st.markdown(processed_answer)

        # Список источников (если показывать)
        if show_sources and retrieved_docs:
            st.markdown("---")
            with st.expander("📚 Источники и релевантные отрывки"):
                HighlightedAnswer._show_sources(
                    retrieved_docs,
                    documents_dirs
                )

    @staticmethod
    def _process_answer_with_links(
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> str:
        """
        Обработать ответ и добавить ссылки на источники в конец

        Args:
            answer: Текст ответа
            retrieved_docs: Список документов

        Returns:
            Ответ с добавленными ссылками на источники
        """

        # Добавить сноски с источниками в конец ответа
        sources_html = "\n\n**Источники:**\n"

        for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3
            doc_id = doc.get('doc_id', 'Unknown')
            score = doc.get('score', 0.0)

            # Парсить doc_id вида "filename_pN"
            parts = doc_id.rsplit('_p', 1)
            filename = parts[0] if parts else doc_id
            page = int(parts[1]) if len(parts) > 1 else 1

            # Сноска с ссылкой на документ
            sources_html += f"\n{i}. [{filename}.pdf, стр. {page}](file={filename}|page={page}) (релевантность: {score:.2f})"

        return answer + sources_html

    @staticmethod
    def _show_sources(
        retrieved_docs: List[Dict[str, Any]],
        documents_dirs: Dict[str, Path]
    ):
        """
        Показать список источников и релевантные отрывки

        Args:
            retrieved_docs: Список найденных документов
            documents_dirs: Dict с доменами и путями к документам
        """

        for i, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.get('doc_id', 'Unknown')
            score = doc.get('score', 0.0)
            text = doc.get('text', '')

            # Парсить doc_id
            parts = doc_id.rsplit('_p', 1)
            filename = parts[0] if parts else doc_id
            page = int(parts[1]) if len(parts) > 1 else 1

            # Контейнер источника
            with st.container(border=True):
                col_title, col_score = st.columns([3, 1])

                with col_title:
                    st.markdown(f"**#{i}. {filename}.pdf**, страница **{page}**")

                with col_score:
                    st.metric("Релевантность", f"{score:.2f}", delta=None)

                # Кнопка открыть в просмотре
                if st.button(
                    f"👁️ Открыть документ",
                    key=f"open_doc_{i}",
                    use_container_width=True
                ):
                    st.session_state.selected_file = filename
                    st.session_state.selected_page = page
                    st.session_state.show_document_viewer = True
                    st.rerun()

                # Отрывок текста с подсветкой
                st.markdown("### Отрывок:")

                # Подсветить первые 200 символов
                preview_text = text[:300]
                if len(text) > 300:
                    preview_text += "..."

                # Жёлтая подсветка для релевантного текста
                highlighted_text = f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 3px;">{preview_text}</mark>'
                st.markdown(highlighted_text, unsafe_allow_html=True)

                st.caption(f"📌 Дополнение к ответу на основе этого источника")


class AnswerInteraction:
    """
    Интерактивные элементы для ответов
    Позволяет пользователю копировать, оценивать, делиться
    """

    @staticmethod
    def show_actions(answer: str, answer_id: str = "answer"):
        """
        Показать кнопки действий для ответа

        Args:
            answer: Текст ответа
            answer_id: Уникальный ID ответа для отслеживания
        """

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button("📋 Копировать", key=f"copy_{answer_id}"):
                st.toast("✓ Скопировано в буфер обмена")

        with col2:
            if st.button("👍 Полезно", key=f"helpful_{answer_id}"):
                st.session_state.feedback = ("helpful", answer_id)
                st.toast("✓ Спасибо за обратную связь!")

        with col3:
            if st.button("👎 Не полезно", key=f"unhelpful_{answer_id}"):
                st.session_state.feedback = ("unhelpful", answer_id)
                st.toast("✓ Мы учтём ваше мнение")

        with col4:
            if st.button("💾 Сохранить", key=f"save_{answer_id}"):
                st.session_state.saved_answers = st.session_state.get(
                    "saved_answers", []
                )
                st.session_state.saved_answers.append(answer)
                st.toast("✓ Ответ сохранён")
