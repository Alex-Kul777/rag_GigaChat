from setuptools import setup, find_packages

setup(
    name="rag_gigachat",
    version="0.1.0",
    description="RAG system with GigaChat LLM and FAISS vector search",
    author="Alex Kul",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies from requirements.txt are listed there
        # This setup.py just makes the package discoverable
    ],
)
