# Agentic AI Marketing Agent Platform - Setup Guide 🛠️

This guide provides detailed instructions for setting up the platform using either Docker (recommended) or manual installation.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Option 1: Docker Setup (Recommended)](#option-1-docker-setup-recommended)
- [Option 2: Manual Setup](#option-2-manual-setup)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### For Docker Setup (Recommended)

| Requirement | Minimum | Recommended |
|---|---|---|
| **Docker** | v20.10+ | Latest |
| **Docker Compose** | v2.0+ | Latest |
| **RAM** | 16 GB | 32 GB |
| **Disk Space** | 20 GB | 50 GB |
| **GPU (Optional)** | — | NVIDIA GPU with CUDA for GPU-accelerated bandits & Ollama |
| **NVIDIA Container Toolkit** | — | Required if using GPU |

### For Manual Setup

| Requirement | Version |
|---|---|
| **Python** | 3.11+ |
| **PostgreSQL** | 16 with pgvector extension |
| **Redis** | 7+ |
| **Node.js** | 18+ (for CI only) |
| **Git** | 2.0+ |

### API Keys & Services

| Service | Required? | Purpose |
|---|---|---|
| **OpenAI API Key** | **Yes** (or Ollama) | Content generation, safety validation |
| **Ollama** | Alternative to OpenAI | Free local LLM (configure via dashboard) |
| **Apify API Token** | Optional | Market scraping / competitive intelligence |
| **LinkedIn Credentials** | Optional | LinkedIn campaign deployment |
| **X (Twitter) Credentials** | Optional | X campaign deployment |
| **SendGrid / Mailgun API Key** | Optional | Email campaign deployment |
| **HubSpot API Key** | Optional | CRM integration |
| **Cal.com API Key** | Optional | Calendar scheduling integration |

> **Note:** All optional API keys can be configured at runtime through the **Dashboard → Operations → Configuration** page. They do not need to be set in `.env`.

---

## Option 1: Docker Setup (Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/agentic-marketing-platform.git
cd agentic-marketing-platform
```

### Step 2: Create Environment File

```bash
cp .env.example .env
```

Edit `.env` and set the **required** values:

```bash
# --- Database (required) ---
POSTGRES_DB=agentic
POSTGRES_USER=agentic
POSTGRES_PASSWORD=your-secure-password-here    # CHANGE THIS

# --- Security (required) ---
SECRET_KEY=generate-a-random-64-char-string     # CHANGE THIS

# --- Monitoring (required) ---
GRAFANA_USER=admin
GRAFANA_PASSWORD=your-grafana-password-here     # CHANGE THIS
```

> **Important:** Only infrastructure secrets go in `.env`. All application settings (API keys, LLM config, feature flags, thresholds) are managed through the **DB-backed Configuration Service** accessible via **Dashboard → Operations → Configuration**.

### Step 3: (Optional) GPU Setup

If you have an NVIDIA GPU and want GPU-accelerated bandits or local LLM via Ollama:

```bash
# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU is accessible in Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

### Step 4: Start the Platform

```bash
# Start all 16 services
make up

# Or, without Make:
docker-compose up -d
```

Wait approximately 30–60 seconds for all services to become healthy. Verify:

```bash
docker-compose ps
```

All services should show `healthy` status.

### Step 5: Initialize the Database

```bash
# Run database initialization (creates tables, extensions, indexes)
docker-compose exec api python scripts/project/init_db.py

# Run Alembic migrations
docker-compose exec api alembic upgrade head

# Seed with demo data (optional but recommended for first setup)
docker-compose exec api python scripts/project/seed_data.py
```

### Step 6: Configure Application Settings

Open the dashboard at **http://localhost:8501** and navigate to:

**📡 System Transparency → Configuration** (or **⚙️ Operations → Configuration**)

Set your API keys and preferences:
- `OPENAI_API_KEY` — Your OpenAI API key
- `OPENAI_MODEL` — Model to use (default: `gpt-4-turbo-preview`)
- `USE_LOCAL_LLM` — Set to `true` to use Ollama instead of OpenAI
- `OLLAMA_HOST` — Ollama server URL (default: `http://localhost:11434`)
- `OLLAMA_MODEL` — Local model name (default: `qwen3:8b`)
- Platform API keys (LinkedIn, X, SendGrid, Mailgun, etc.)

### Step 7: (Optional) Set Up Ollama for Local LLM

```bash
# Ollama runs as a Docker service. Pull a model:
docker-compose exec ollama ollama pull qwen3:8b

# Or use the setup script:
bash scripts/project/setup_ollama.sh
```

Then enable local LLM in **Dashboard → Configuration** by setting `USE_LOCAL_LLM = true`.

### Step 8: Access the Platform

| Service | URL | Description |
|---|---|---|
| **Dashboard** | http://localhost:8501 | Streamlit interactive dashboard (16 pages) |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger/OpenAPI documentation |
| **API ReDoc** | http://localhost:8000/redoc | Alternative API documentation |
| **Grafana** | http://localhost:3000 | Monitoring dashboards (login with GRAFANA_USER/PASSWORD) |
| **Prometheus** | http://localhost:9090 | Metrics exploration and queries |
| **MLflow** | http://localhost:5000 | ML experiment tracking and model registry |
| **Alertmanager** | http://localhost:9093 | Alert management |
| **Loki** | http://localhost:3100 | Log aggregation (accessed via Grafana) |

### Step 9: Verify Installation

```bash
# Check API health
curl http://localhost:8000/health | python -m json.tool

# Check all service health
make health

# Run the golden test suite
make golden
```

---

## Option 2: Manual Setup

### Step 1: Install System Dependencies

**Ubuntu/Debian:**

```bash
sudo apt-get update && sudo apt-get install -y \
    python3.11 python3.11-venv python3.11-dev \
    postgresql-16 postgresql-16-pgvector \
    redis-server \
    gcc g++ curl git
```

**macOS:**

```bash
brew install python@3.11 postgresql@16 redis git
# pgvector: install from source or via brew
brew install pgvector
```

### Step 2: Clone & Create Virtual Environment

```bash
git clone https://github.com/your-username/agentic-marketing-platform.git
cd agentic-marketing-platform

python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** This installs ~100 packages including PyTorch. For GPU support, install the CUDA-specific PyTorch wheel first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Step 4: Set Up PostgreSQL

```bash
# Start PostgreSQL
sudo systemctl start postgresql

# Create user and database
sudo -u postgres psql -c "CREATE USER agentic WITH PASSWORD 'your-password';"
sudo -u postgres psql -c "CREATE DATABASE agentic OWNER agentic;"
sudo -u postgres psql -d agentic -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d agentic -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

### Step 5: Set Up Redis

```bash
sudo systemctl start redis
redis-cli ping  # Should return PONG
```

### Step 6: Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
POSTGRES_DB=agentic
POSTGRES_USER=agentic
POSTGRES_PASSWORD=your-password
SECRET_KEY=your-secret-key
GRAFANA_USER=admin
GRAFANA_PASSWORD=your-password
```

### Step 7: Initialize Database

```bash
export PYTHONPATH=$(pwd)
export DATABASE_URL=postgresql://agentic:your-password@localhost:5432/agentic
export REDIS_URL=redis://localhost:6379

# Initialize database schema
python scripts/project/init_db.py

# Run migrations
alembic upgrade head

# Seed demo data (optional)
python scripts/project/seed_data.py
```

### Step 8: Start Services

```bash
# Terminal 1: Start API server
export PYTHONPATH=$(pwd)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Dashboard
streamlit run dashboard/app.py --server.port 8501

# Terminal 3: Start Background Worker
rq worker --url redis://localhost:6379 default high low
```

### Step 9: (Optional) Install Monitoring Stack

For the full monitoring stack (Prometheus, Grafana, Loki), use the Docker services even in manual mode:

```bash
docker-compose up -d prometheus grafana loki promtail alertmanager \
    postgres-exporter redis-exporter node-exporter cadvisor
```

---

## Configuration

### `.env` File (Infrastructure Only)

The `.env` file contains **only infrastructure secrets** needed before the database is available:

| Variable | Required | Default | Description |
|---|---|---|---|
| `POSTGRES_DB` | Yes | `agentic` | PostgreSQL database name |
| `POSTGRES_USER` | Yes | `agentic` | PostgreSQL username |
| `POSTGRES_PASSWORD` | **Yes** | — | PostgreSQL password |
| `SECRET_KEY` | **Yes** | — | JWT/encryption key |
| `GRAFANA_USER` | Yes | `admin` | Grafana admin username |
| `GRAFANA_PASSWORD` | **Yes** | — | Grafana admin password |

### DB-Backed Configuration Service

All other settings are managed via the **Configuration Service** (`src/config/configuration_service.py`), which stores values in the PostgreSQL database with optional encryption for secrets.

Access via: **Dashboard → ⚙️ Operations → Configuration**

Key configurable settings include:

| Category | Settings |
|---|---|
| **LLM** | `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`, `OPENAI_MAX_TOKENS`, `USE_LOCAL_LLM`, `OLLAMA_HOST`, `OLLAMA_MODEL` |
| **Safety** | `SAFETY_SCORE_THRESHOLD`, `TOXICITY_THRESHOLD`, `AUTO_APPROVE_THRESHOLD` |
| **Costs** | `MAX_DAILY_API_COST`, `MAX_CAMPAIGN_COST` |
| **Mock Mode** | `MOCK_MODE_ENABLED`, `ENABLE_MOCK_DEPLOYMENT`, `ENABLE_MOCK_EXPERIMENTS` |
| **Platforms** | `LINKEDIN_ACCESS_TOKEN`, `X_API_KEY`, `SENDGRID_API_KEY`, `MAILGUN_API_KEY`, `APIFY_API_TOKEN`, `HUBSPOT_PRIVATE_APP_TOKEN` |
| **Simulation** | `SIMULATION_DURATION_DAYS`, `SIMULATION_TIME_STEP` |

---

## Troubleshooting

### Common Issues

**1. `POSTGRES_PASSWORD` not set error on `docker-compose up`**
```bash
# The .env file must have POSTGRES_PASSWORD set (cannot be empty)
echo "POSTGRES_PASSWORD=your-secure-password" >> .env
```

**2. GPU not detected in containers**
```bash
# Verify NVIDIA Container Toolkit is installed
nvidia-ctk --version

# Verify GPU access in Docker
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi

# If GPU is not needed, remove the `deploy.resources.reservations.devices`
# section from the api and worker services in docker-compose.yml
```

**3. `ImportError: No module named 'langchain_openai'`**
```bash
pip install langchain-openai langchain-community langchain-ollama
```

**4. `DatabaseError: extension 'vector' does not exist`**
```bash
# The pgvector/pgvector:pg16 Docker image includes it.
# For manual install:
sudo -u postgres psql -d agentic -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**5. `OpenAI API authentication failed`**
```
Configure your OpenAI key via Dashboard → Operations → Configuration
Set OPENAI_API_KEY to your valid key, or enable USE_LOCAL_LLM for Ollama.
```

**6. `ConnectionError: Redis connection refused`**
```bash
# Docker: Check Redis is running
docker-compose ps redis

# Manual: Start Redis
sudo systemctl start redis
```

**7. Port conflicts**
```bash
# Check what's using a port
sudo lsof -i :8000
sudo lsof -i :8501

# Change ports in docker-compose.yml if needed
```

**8. API container keeps restarting**
```bash
# Check logs for the error
docker-compose logs api --tail=50

# Common cause: DATABASE_URL not matching running PostgreSQL
```

**9. Dashboard cannot connect to API**
```
The dashboard connects via http://host.docker.internal:8000.
Ensure the API container is running and healthy:
  docker-compose ps api
  curl http://localhost:8000/health
```

**10. Alembic migration errors**
```bash
# Reset migrations (development only!)
alembic stamp head
alembic upgrade head

# Or reinitialize the database
python scripts/project/init_db.py
alembic upgrade head
```

### Diagnostic Commands

```bash
# Full health check
make health

# View service status
docker-compose ps

# API health with details
curl -s http://localhost:8000/api/v1/health | python -m json.tool
```
