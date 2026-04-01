# RAG System - Retrieval-Augmented Generation System

## 📋 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system for answering questions based on PDF documents. The system combines document retrieval with LLM generation to provide accurate, context-aware answers.

## 🚀 Features

- **Document Processing**: Load PDF documents with metadata extraction and caching
- **Vector Search**: FAISS-based semantic search with configurable embeddings
- **Hybrid Retrieval**: Support for dense, sparse, and hybrid retrieval strategies
- **LLM Integration**: Support for local models (HuggingFace) and GigaChat
- **Experiment Framework**: Run experiments with different configurations
- **Evaluation Metrics**: 
  - Retrieval: MAP, MRR, Precision@k, Recall@k, NDCG@k
  - Generation: ROUGE, BLEU, BERTScore
  - Advanced: Faithfulness, Answer Relevancy, Context Relevancy (RAGAS)
- **Web Interface**: Streamlit-based chat UI
- **Reporting**: Excel reports with experiment summaries

## 📁 Project Structure

```
.
├── app.py                 # Main entry point (UI, query, experiment modes)
├── config.py              # Centralized configuration
├── data_loader.py         # Document loading with caching
├── rag_core.py            # Core RAG pipeline (FAISS, LangGraph)
├── models.py              # Data models (dataclasses, enums)
├── evaluator.py           # Evaluation metrics
├── experiment.py          # Experiment runner
├── excel_reporter.py      # Excel report generation
├── ui_streamlit.py        # Streamlit web interface
├── create_wikieval_dataset.py  # Dataset creation utility
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- [GigaChat API key](https://developers.sber.ru/) (optional, for GigaChat models)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your GigaChat API key
```

### Configuration

The system uses centralized configuration in `config.py`:

```python
# Model settings
model_config.llm_model_name = "GigaChat-2-Max"
model_config.embedding_model_name = "GigaChat-2-Max"
model_config.temperature = 0.7
model_config.default_k_retrieve = 5

# Data settings
data_config.chunk_size = 800
data_config.chunk_overlap = 100
data_config.documents_dirs = {
    "debug": Path("data/domain_2_Debug/books"),
    "ai": Path("data/domain_1_AI/books"),
    "test": Path("data/test_docs")
}

# GigaChat settings
gigachat_config.api_key = os.getenv("GIGACHAT_API_KEY", "")
gigachat_config.model = "GigaChat-2-Max"
```

## 🚦 Usage

### 1. Web Interface (Recommended)

```bash
python app.py --mode ui
```
Then open `http://localhost:8501` in your browser.

### 2. Single Query Mode

```bash
python app.py --mode query --query "What is RAG?" --retrieval_type dense --k 5
```

Options:
- `--query`: Question text
- `--retrieval_type`: `dense`, `sparse`, or `hybrid`
- `--k`: Number of documents to retrieve
- `--output`: Save result to JSON file

### 3. Experiment Mode

```bash
python app.py --mode experiment --testset testset.json --experiment_name my_exp --retrieval_type hybrid
```

Options:
- `--testset`: Path to JSON file with test queries
- `--experiment_name`: Name for this experiment
- `--retrieval_type`: Retrieval method
- `--k`: Number of documents to retrieve
- `--output_dir`: Directory for results

### 4. Dataset Creation

Create a test dataset from WikiEval:

```bash
python create_wikieval_dataset.py --max_samples 10 --output_dir data/my_dataset
```

## 📊 Test Dataset Format

The test dataset JSON should follow this format:

```json
{
    "q000": {
        "query": "What is machine learning?",
        "relevant_docs": ["document1.pdf"],
        "reference_answer": "Machine learning is a subset of AI..."
    },
    "q001": {
        "query": "Explain neural networks",
        "relevant_docs": ["document2.pdf", "document3.pdf"],
        "reference_answer": "Neural networks are computational models..."
    }
}
```

## 📈 Experiment Results

Experiments are saved in the `experiments/` directory:

```
experiments/
├── results/               # JSON files with detailed results
│   ├── experiment_20250315_120000.json
│   └── experiment_20250315_120000_config.json
├── reports/               # Text reports
├── checkpoints/           # Experiment checkpoints
└── experiments_summary.csv  # CSV summary
```

Generate Excel report:
```bash
python excel_reporter.py
```

## 🧪 Evaluation Metrics

### Retrieval Metrics
- **MAP** (Mean Average Precision): Average precision across queries
- **MRR** (Mean Reciprocal Rank): Position of first relevant document
- **Precision@k**: Proportion of relevant documents in top-k
- **Recall@k**: Proportion of relevant documents retrieved in top-k
- **NDCG@k**: Normalized Discounted Cumulative Gain

### Generation Metrics
- **ROUGE**: Overlap with reference answers
- **BLEU**: N-gram precision
- **BERTScore**: Semantic similarity using BERT

### RAGAS Metrics (Advanced)
- **Faithfulness**: How factually accurate the answer is
- **Answer Relevancy**: How relevant the answer is to the question
- **Context Relevancy**: How relevant the retrieved context is

## 🔧 Configuration Options

### Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `llm_model_name` | LLM model name | `GigaChat-2-Max` |
| `embedding_model_name` | Embedding model | `GigaChat-2-Max` |
| `max_new_tokens` | Max tokens to generate | `1000` |
| `temperature` | Generation randomness | `0.7` |
| `default_k_retrieve` | Default documents to retrieve | `5` |
| `device` | Computation device | `cpu` |

### Data Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `chunk_size` | Document chunk size | `800` |
| `chunk_overlap` | Overlap between chunks | `100` |
| `force_reload` | Reload documents from source | `False` |
| `cache_enabled` | Enable document caching | `True` |

## 🐛 Troubleshooting

### Common Issues

1. **"GigaChat API key not found"**
   - Ensure `.env` file exists with `GIGACHAT_API_KEY=your_key`
   - Check that `python-dotenv` is installed

2. **"FAISS index not initialized"**
   - Load documents first using `load_from_pdf_directory_with_metadata()`
   - Check that the PDF directory exists and contains files

3. **"Module not found"**
   - Install missing dependencies: `pip install -r requirements.txt`
   - Ensure virtual environment is activated

4. **Out of memory errors**
   - Reduce `chunk_size` and `chunk_overlap`
   - Use CPU mode (`model_config.device = "cpu"`)

## 📝 Logging

The system uses a filtered logging system:
- **Console logs**: Only from modules in `OUR_MODULES` (debug, experiment, rag_core, etc.)
- **File logs**: All logs saved to `logs/rag_app.log`
- **Third-party library logs**: Suppressed by default

Configure logging in `config.py`:
```python
logging_config.log_level = "DEBUG"  # or INFO, WARNING
logging_config.log_to_file = True
logging_config.log_to_console = True
```

## 🔄 Workflow

1. **Document Loading**: PDFs are loaded, split into chunks, and stored in FAISS
2. **Query Processing**: User query is embedded and used for similarity search
3. **Retrieval**: Top-k relevant chunks are retrieved from FAISS
4. **Context Building**: Retrieved chunks are combined into a context
5. **Generation**: LLM generates answer based on context and question
6. **Evaluation**: Metrics computed for retrieval and generation quality

## 🧪 Running Tests

### Test Document Loading
```bash
python data_loader.py --pdf_dir data/domain_4_WikiEval_2row/books
```

### Test RAG Pipeline
```bash
python rag_core.py
```

### Run Experiment
```bash
python experiment.py --pdf_dir data/domain_4_WikiEval_2row/books --testset data/domain_4_WikiEval_2row/testset.json --name my_experiment --force_reload
```

## 📚 Dependencies

- `langchain` - RAG framework
- `langchain-gigachat` - GigaChat integration
- `faiss-cpu` - Vector search
- `streamlit` - Web interface
- `transformers` - Local LLMs
- `pandas` - Data manipulation
- `openpyxl` - Excel reports
- `ragas` - Advanced metrics
- `rouge-score` - ROUGE metrics
- `bert-score` - BERTScore metrics
- `reportlab` - PDF generation
- `datasets` - HuggingFace datasets

## 👨‍💻 Development

### Adding a New Document Source

1. Add document loading method in `data_loader.py`
2. Update `CorpusLoader.load_from_*` method
3. Add support in `RAGPipeline.load_from_*`

### Adding a New LLM

1. Extend `LLMManager` with new loading method
2. Add configuration in `ModelConfig`
3. Update `RAGPipeline` to support new type

### Adding New Metrics

1. Add metric calculation in `evaluator.py`
2. Update `ExperimentResult` to include new metric
3. Modify `excel_reporter.py` to display new metrics

## 📄 License

[Your License Here]

## 📧 Contact

[Your Contact Information]

---

**Note**: This system requires a GigaChat API key for using GigaChat models. For local models, ensure you have sufficient memory (16GB+ recommended for 7B models).