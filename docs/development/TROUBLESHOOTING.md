# Troubleshooting Guide

Comprehensive troubleshooting guide for RAG GigaChat project.

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [GPU & Docker Issues](#gpu--docker-issues)
3. [API & Configuration Issues](#api--configuration-issues)
4. [Runtime Issues](#runtime-issues)
5. [Performance Issues](#performance-issues)
6. [Advanced Debugging](#advanced-debugging)

---

## Installation Issues

### Python Version Mismatch

**Problem:**
```
ERROR: Python 3.9 not supported. Required: 3.11+
```

**Solution:**
```bash
# Check Python version
python --version

# Install Python 3.11
# macOS (with Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11 python3.11-venv

# Windows
# Download from python.org or use Windows Package Manager
winget install Python.Python.3.11
```

### Virtual Environment Issues

**Problem:**
```
ModuleNotFoundError: No module named 'venv'
```

**Solution:**
```bash
# Linux/Mac
sudo apt-get install python3.11-venv

# Then create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Pip/Dependency Issues

**Problem:**
```
ERROR: Could not find a version that satisfies the requirement
```

**Solution:**
```bash
# Clear pip cache
pip cache purge

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Reinstall requirements with verbose output
pip install -r requirements.txt -v

# If specific package fails:
pip install <package-name>==<exact-version> -v
```

---

## GPU & Docker Issues

### NVIDIA Docker Runtime Not Found

**Problem:**
```
Error: could not select device driver "" with capabilities: [[gpu]]
```

**Solution:**
```bash
# Install NVIDIA Container Toolkit
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Then try:
GPU_COUNT=1 docker-compose up
```

### Docker Permission Denied

**Problem:**
```
Got permission denied while trying to connect to Docker daemon
```

**Solution:**
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Or run with sudo (not recommended)
sudo docker-compose up
```

### Port 8501 Already in Use

**Problem:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Check what's using the port
lsof -i :8501  # macOS/Linux
netstat -ano | findstr :8501  # Windows

# Kill the process
kill -9 <PID>  # macOS/Linux
taskkill /PID <PID> /F  # Windows

# Or use different port
docker-compose up -p 8502:8501
# or
streamlit run src/rag_gigachat/ui/streamlit_app.py --server.port 8502
```

### GPU Not Detected in Docker

**Problem:**
```
torch.cuda.is_available() returns False
No GPU detected in container
```

**Solution:**
```bash
# Test GPU on host
nvidia-smi
nvidia-docker run --rm --gpus all ubuntu nvidia-smi

# Check Docker compose settings
cat docker-compose.yml | grep -A 5 "devices:"

# Rebuild with GPU flag
GPU_ENABLED=true docker-compose up --build

# Verify in running container
docker-compose exec rag-gigachat nvidia-smi
```

---

## API & Configuration Issues

### GigaChat API Key Not Found

**Problem:**
```
AuthenticationError: Missing GigaChat API key
```

**Solution:**
1. Get your API key: https://lk.sbercloud.ru/fusion/auth/login
2. Create `.env` file:
```bash
cp .env.example .env
```

3. Edit `.env`:
```bash
GIGACHAT_API_KEY=your-actual-key-here
```

4. Verify it's loaded:
```bash
# Python
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv('GIGACHAT_API_KEY'))  # Should show your key
```

### Invalid API Key Format

**Problem:**
```
Invalid API key format or expired
```

**Solution:**
```bash
# 1. Check key format (should be base64)
echo "your-key" | base64 -d

# 2. Verify key in Sber Cloud dashboard
# Go to: https://lk.sbercloud.ru/fusion/auth/login

# 3. Get new key if expired
# Keys expire after ~30 days

# 4. Test connection:
python -c "
from src.rag_gigachat.core.llm_manager import LLMManager
llm = LLMManager()
print(llm.test_connection())
"
```

### .env File Not Being Read

**Problem:**
```
GIGACHAT_API_KEY is None/empty
```

**Solution:**
```bash
# 1. Check .env file exists
ls -la .env

# 2. Check content
cat .env

# 3. Ensure no spaces around '='
# ✅ Correct:
GIGACHAT_API_KEY=your-key

# ❌ Wrong:
GIGACHAT_API_KEY = your-key
GIGACHAT_API_KEY= your-key

# 4. In Docker, restart:
docker-compose down
docker-compose up --build
```

---

## Runtime Issues

### FAISS Index Not Initialized

**Problem:**
```
RuntimeError: Vector store not initialized. Load documents first.
```

**Solution:**
```bash
# 1. Upload documents in UI (recommended)
# - Open http://localhost:8501
# - Go to "Upload Documents" tab
# - Upload PDF files

# 2. Or load from directory
python app.py --mode query --pdf_dir data/documents/

# 3. Verify index exists
ls -la data/vectorstore/
# Should see: index.faiss, index_metadata.pkl
```

### Out of Memory Error

**Problem:**
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate
```

**Solution:**

**Step 1: Check available memory**
```bash
# GPU
nvidia-smi  # Look at "Memory" column

# CPU
free -h  # Linux
vm_stat  # macOS
systeminfo  # Windows
```

**Step 2: Reduce chunk size**
```python
# In config.py or environment
CHUNK_SIZE=256  # Default is 512
CHUNK_OVERLAP=50  # Default is 100
```

**Step 3: Reduce batch size**
```python
# In rag_pipeline.py
BATCH_SIZE=8  # Default is 32
```

**Step 4: Clear cache**
```bash
rm -rf data/cache/
rm -rf data/vectorstore/
```

**Step 5: Use CPU mode**
```bash
# For embeddings
export DEVICE=cpu

# Don't use GPU
GPU_COUNT=0 docker-compose up
```

### Module Not Found / Import Error

**Problem:**
```
ModuleNotFoundError: No module named 'langchain'
ImportError: cannot import name 'ChatGigaChat'
```

**Solution:**
```bash
# 1. Verify venv is activated
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
which python  # Should show .venv/bin/python

# 2. Reinstall dependencies
pip install --upgrade -r requirements.txt

# 3. Check specific package
pip show langchain-gigachat

# 4. In Docker, rebuild
docker-compose down
docker-compose up --build --no-cache
```

### Timeout Errors

**Problem:**
```
TimeoutError: Request to GigaChat API timed out
ConnectionError: Unable to connect to API
```

**Solution:**
```bash
# 1. Check network connection
ping lk.sbercloud.ru

# 2. Check API status (Sber status page)
# https://lk.sbercloud.ru/

# 3. Increase timeout
# In config.py or environment:
API_TIMEOUT=60  # Default is 30

# 4. Check firewall
sudo ufw status  # Linux
# Check Windows Firewall

# 5. Try proxy if needed
# In config.py:
# proxies = {"https": "http://proxy:port"}
```

---

## Performance Issues

### Slow PDF Processing

**Problem:**
```
PDF processing taking >30 seconds
OCR running on every PDF
```

**Solution:**

**Check what's slow:**
```bash
# Debug logging
LOG_LEVEL=DEBUG python app.py --mode query

# Profile code
python -m cProfile -s cumtime app.py --mode query
```

**Optimize:**
```python
# 1. Disable OCR for native PDFs
OCR_ENABLED=false

# 2. Reduce chunk size
CHUNK_SIZE=256

# 3. Use caching
CACHE_ENABLED=true

# 4. Clear old cache
rm -rf data/cache/
```

### Slow Vector Search

**Problem:**
```
Similarity search taking >5 seconds
High GPU memory usage
```

**Solution:**
```bash
# 1. Check index size
ls -lh data/vectorstore/index.faiss

# 2. Use fewer results
TOP_K=3  # Instead of 5

# 3. Use faster FAISS option
# In vector_store.py:
# metric_type = "L2"  # Faster than cosine

# 4. Reduce vector dimension
# Use smaller embedding model
```

### Slow LLM Generation

**Problem:**
```
LLM response taking >30 seconds
High latency to GigaChat API
```

**Solution:**
```bash
# 1. Check API status
curl https://api.gigachat.ai/health

# 2. Reduce token limits
MAX_TOKENS=500  # Instead of 1000

# 3. Reduce context length
CONTEXT_LENGTH=2000  # Instead of 4000

# 4. Use faster model
# GigaChat-2-Lite instead of GigaChat-2-Max

# 5. Increase timeout
API_TIMEOUT=60
```

---

## Advanced Debugging

### Enable Debug Logging

```bash
# Full debug output
LOG_LEVEL=DEBUG python app.py --mode query --query "test"

# Save to file
LOG_LEVEL=DEBUG python app.py > debug.log 2>&1
```

### Check Environment Variables

```bash
# Show all RAG-related vars
env | grep -E "GIGACHAT|CHUNK|FAISS|DEVICE"

# Or in Python
import os
print(os.environ.get('GIGACHAT_API_KEY', 'NOT SET'))
```

### Docker Container Debugging

```bash
# View logs
docker-compose logs -f rag-gigachat

# Execute command in running container
docker-compose exec rag-gigachat bash
docker-compose exec rag-gigachat python -c "import torch; print(torch.cuda.is_available())"

# Inspect image
docker images ls
docker inspect <image-id>
```

### Test GigaChat Connection

```python
# test_gigachat.py
from src.rag_gigachat.core.llm_manager import LLMManager
from src.rag_gigachat.config import gigachat_config

# Check config
print(f"API Key: {gigachat_config.api_key[:10]}...")
print(f"Model: {gigachat_config.model}")

# Test connection
llm = LLMManager()
try:
    response = llm.llm.invoke("Hello, world!")
    print(f"✅ Connection successful: {response.content[:50]}...")
except Exception as e:
    print(f"❌ Connection failed: {e}")
```

Run it:
```bash
python test_gigachat.py
```

### Test FAISS Index

```python
# test_faiss.py
from src.rag_gigachat.core.vector_store import VectorStore
from src.rag_gigachat.config import data_config

vs = VectorStore(index_path=data_config.faiss_index_path)

print(f"Index initialized: {vs.is_initialized}")
print(f"Index size: {vs.get_index_size()}")

if vs.is_initialized:
    # Test search
    results = vs.search("What is RAG?", top_k=3)
    print(f"Search results: {len(results)} documents found")
else:
    print("❌ Index not initialized. Load documents first.")
```

### Profile Memory Usage

```bash
# Monitor memory during execution
# macOS/Linux
while true; do ps aux | grep python; sleep 1; done

# Or use profiler
pip install memory-profiler
python -m memory_profiler app.py --mode query

# GPU memory (if using CUDA)
watch -n 1 nvidia-smi
```

---

## Getting Help

If you can't find a solution:

1. **Check GitHub Issues**: https://github.com/Alex-Kul777/rag_GigaChat/issues
2. **Enable debug logging** and share output
3. **Provide system info**:
   ```bash
   python --version
   pip show langchain
   nvidia-smi 2>/dev/null || echo "No GPU"
   docker --version 2>/dev/null || echo "No Docker"
   ```
4. **Create detailed issue** with:
   - Error message (full traceback)
   - Steps to reproduce
   - System information
   - Debug logs

---

**Last updated**: 2026-04-17
