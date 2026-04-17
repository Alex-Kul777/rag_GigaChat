#!/bin/bash
# Setup script for RAG GigaChat project
# Supports both GPU and CPU modes

set -e

echo "🚀 RAG GigaChat Setup Script"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $1${NC}"
    else
        echo -e "${RED}❌ $1${NC}"
    fi
}

# Function to check NVIDIA drivers
check_nvidia_drivers() {
    echo ""
    echo "🔍 Checking NVIDIA drivers and GPU support..."
    echo "----------------------------------------------"

    # Level 1: Check if nvidia-smi exists
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${YELLOW}⚠️  nvidia-smi not found${NC}"
        echo "   → GPU support is not available"
        echo "   → Using CPU mode (slower, but works)"
        return 1
    fi

    # Level 2: Get driver version
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    if [ -z "$DRIVER_VERSION" ]; then
        echo -e "${RED}❌ Failed to get driver version${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ NVIDIA Driver version: $DRIVER_VERSION${NC}"

    # Level 3: Check CUDA Compute Capability
    CUDA_CAPABILITY=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null)
    if [ -z "$CUDA_CAPABILITY" ]; then
        echo -e "${YELLOW}⚠️  Could not detect CUDA compute capability${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ CUDA Compute Capability: $CUDA_CAPABILITY${NC}"

    # Level 4: Check CUDA libraries
    if ! ldconfig -p 2>/dev/null | grep -q libcuda.so; then
        echo -e "${YELLOW}⚠️  libcuda.so not in LD_LIBRARY_PATH${NC}"
        echo "   → Set: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
        return 1
    fi
    echo -e "${GREEN}✅ CUDA libraries found${NC}"

    return 0
}

# Function to setup Python environment
setup_python_env() {
    echo ""
    echo "🐍 Setting up Python environment..."
    echo "------------------------------------"

    if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3.11 -m venv .venv || python3.10 -m venv .venv || python3 -m venv .venv
        print_status "Virtual environment created"
    else
        echo "Virtual environment already exists"
    fi

    # Activate venv
    source .venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"

    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    print_status "pip upgraded"

    # Install dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    print_status "Dependencies installed"
}

# Function to setup .env file
setup_env_file() {
    echo ""
    echo "⚙️  Setting up .env file..."
    echo "----------------------------"

    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            echo -e "${YELLOW}⚠️  Created .env from .env.example${NC}"
            echo "   → Please edit .env and add your GIGACHAT_API_KEY"
            return 1
        else
            echo -e "${RED}❌ Neither .env nor .env.example found${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}✅ .env file exists${NC}"

        # Check if API key is configured
        if grep -q "GIGACHAT_API_KEY=your-api-key-here" .env || ! grep -q "GIGACHAT_API_KEY=" .env; then
            echo -e "${YELLOW}⚠️  GIGACHAT_API_KEY not configured${NC}"
            echo "   → Get key from: https://lk.sbercloud.ru/fusion/auth/login"
            return 1
        else
            echo -e "${GREEN}✅ GIGACHAT_API_KEY configured${NC}"
        fi
    fi
}

# Function to check Docker
check_docker() {
    echo ""
    echo "🐳 Checking Docker setup..."
    echo "----------------------------"

    if ! command -v docker &> /dev/null; then
        echo -e "${YELLOW}⚠️  Docker not found${NC}"
        echo "   → Install from: https://docs.docker.com/get-docker/"
        return 1
    fi
    echo -e "${GREEN}✅ Docker found ($(docker --version))${NC}"

    if ! command -v docker-compose &> /dev/null; then
        echo -e "${YELLOW}⚠️  docker-compose not found${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ docker-compose found ($(docker-compose --version))${NC}"

    # Check Docker daemon
    if ! docker ps > /dev/null 2>&1; then
        echo -e "${RED}❌ Docker daemon not running${NC}"
        echo "   → Start Docker and try again"
        return 1
    fi
    echo -e "${GREEN}✅ Docker daemon is running${NC}"
}

# Main setup flow
main() {
    # Check prerequisites
    check_docker

    # Check NVIDIA support
    GPU_AVAILABLE=false
    if check_nvidia_drivers; then
        GPU_AVAILABLE=true
        echo ""
        echo -e "${GREEN}🚀 GPU support available!${NC}"
        echo "   → Run with: GPU_COUNT=1 docker-compose up --build"
        echo "   → Or: GPU_ENABLED=true docker-compose up --build"
    else
        echo ""
        echo -e "${YELLOW}💡 Using CPU mode (slower, but works)${NC}"
        echo "   → Run with: docker-compose up --build"
    fi

    # Setup Python env (for local development)
    read -p "Setup Python venv for local development? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_python_env
    fi

    # Setup .env file
    if setup_env_file; then
        echo ""
        echo -e "${GREEN}✅ Setup complete!${NC}"
        echo ""
        echo "Next steps:"
        echo "1. Add your GIGACHAT_API_KEY to .env file"
        echo "2. Run: docker-compose up --build"
        echo "3. Open: http://localhost:8501"
        echo ""
    else
        echo ""
        echo -e "${YELLOW}⚠️  Setup incomplete. Please configure .env file.${NC}"
        echo ""
    fi
}

# Run main setup
main
