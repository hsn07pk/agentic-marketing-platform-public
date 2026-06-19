# Agentic Marketing Platform

A multi-agent system for governed, simulation-driven B2B marketing content. Specialized LLM agents are orchestrated through a LangGraph decision loop over a SimPy market simulation, with retrieval-augmented generation, contextual-bandit optimization gated by off-policy evaluation, and human-in-the-loop governance.

> **Public, de-identified release of a master's-thesis platform.** Company-specific data, the RAG knowledge base, and brand / persona / product configuration have been removed. This repository contains the engineering and architecture only. Provide your own data and configuration to run it end to end.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stack](https://img.shields.io/badge/FastAPI%20%2B%20Streamlit-Docker-2496ED)

## Overview

The platform turns a campaign brief into governed, citation-backed marketing content. LLM agents draft and critique copy, a retrieval layer grounds every factual claim in an approved source, a bandit policy chooses between content strategies, and an off-policy evaluation gate estimates lift before anything is deployed. A SimPy market model provides synthetic engagement so policies can be evaluated offline, and a Streamlit console plus a Prometheus/Grafana stack make the whole loop observable.

## Architecture

Layered design under `src/`:

- **AI layer** (`src/ai_layer/`): LLM agents (content generation, safety validation), a LangGraph orchestration graph, prompt-injection shielding, and a learning subsystem (MLflow tracking, autonomous MLOps, multi-touch attribution).
- **Automation layer** (`src/automation_layer/`): outbound connectors (HubSpot, email, blog, Mailchimp, Mailgun) and a canary rollout for staged deployment.
- **Data layer** (`src/data_layer/`): SQLAlchemy models over PostgreSQL, with pgvector for embeddings.
- **API** (`src/api/`): a modular FastAPI service (20+ routers covering configuration, governance, calibration, funnel attribution, operations, and health).
- **Dashboard** (`dashboard/`): a multi-page Streamlit control center (campaigns, analytics, experiments, governance, cost control, policy evaluation, canary deployments, system monitoring).
- **Monitoring** (`monitoring/`): Prometheus, Grafana, Loki, and Promtail configuration.

## Key capabilities

- **Multi-agent orchestration.** A LangGraph `StateGraph` routes work across content, safety-judge, strategy, and research agents, with retries and state checkpointing.
- **Grounded generation (RAG).** pgvector similarity search over a SentenceTransformer (`all-MiniLM-L6-v2`) index, with claim-citation enforcement so factual statements must reference an approved source.
- **Optimization with off-policy evaluation.** Thompson Sampling and LinUCB contextual bandits select content strategies; a doubly-robust OPE gate (inverse propensity scoring plus a direct method) estimates expected lift before rollout.
- **Governance and human-in-the-loop.** A multi-dimensional safety judge scores every draft on toxicity, factuality, brand alignment, and regulatory compliance, with thresholds for auto-approve, human review, or rejection, and a full audit trail.
- **Market simulation.** A SimPy model generates synthetic engagement for offline experimentation and policy evaluation.
- **Observability and MLOps.** MLflow experiment tracking alongside a Prometheus / Grafana / Loki stack.

## Tech stack

Python 3.11 · FastAPI · Streamlit · LangChain + LangGraph · OpenAI API or local Ollama · sentence-transformers · PostgreSQL + pgvector · SQLAlchemy + Alembic · Redis + RQ · MLflow · SimPy · Prometheus / Grafana / Loki · Docker Compose · GitHub Actions · DVC

## Repository structure

```
src/
  ai_layer/          LLM agents, LangGraph orchestration, learning/MLOps, prompt security
  automation_layer/  platform connectors and canary deployment
  data_layer/        SQLAlchemy models (PostgreSQL + pgvector)
  api/               FastAPI service and routers
  config/            configuration service and encryption helpers
  monitoring/        runtime monitors
dashboard/           Streamlit multi-page console
monitoring/          Prometheus / Grafana / Loki / Promtail config
alembic/             database migrations
scripts/             setup, seeding, calibration, research/reporting
tests/               unit, integration, and golden-output tests
```

## Quickstart

Requires Docker, and either an OpenAI API key or a local Ollama install.

```bash
cp .env.example .env        # set POSTGRES_PASSWORD, SECRET_KEY, and LLM settings
make up                     # or: docker compose up -d --build
```

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs

See `SETUP_GUIDE.md` for manual (non-Docker) setup and configuration details.

## What is not included

To keep this release free of proprietary content, the following were removed and must be supplied to run the platform end to end:

- the RAG knowledge base and datasets (`data/`)
- brand voice, audience personas, product catalog, and the claim library (`config/`)
- the original thesis documents

## License

MIT. See `LICENSE`.
