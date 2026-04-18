#!/usr/bin/env python
"""
Quick test to debug document loading
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_gigachat.core.rag_pipeline import RAGPipeline
from rag_gigachat.config import data_config

# Create pipeline
pipeline = RAGPipeline()

# Try to load documents
domain_path = Path(__file__).parent / "data/domain_2_Debug"
print(f"Loading from: {domain_path}")
print(f"Directory exists: {domain_path.exists()}")

try:
    pipeline.load_from_pdf_directory(directory=domain_path, recursive=True, force_reload=True)
    print(f"✅ Loading completed")
    print(f"vector_store_initialized: {pipeline.vector_store_initialized}")
    print(f"manager.is_initialized: {pipeline.vector_store_manager.is_initialized}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
