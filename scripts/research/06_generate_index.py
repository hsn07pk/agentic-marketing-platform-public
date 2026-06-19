#!/usr/bin/env python3
"""
Generate the comprehensive thesis research index with all data, references, and methodology.
This is the master document tying all research artifacts together.
"""
import json
import os
import csv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent.parent / 'agentic' / 'thesis-research'

def generate_references():
    """Compile all references from the research plan."""
    refs = """# References

## Academic Papers & Books

1. Rand, W., & Rust, R. T. (2011). Agent-based modeling in marketing: Guidelines for rigor. *International Journal of Research in Marketing*, 28(3), 181-193.
2. Cheng, Y., et al. (2024). Exploring large language model based intelligent agents: Definitions, methods, and prospects. *arXiv preprint arXiv:2401.03428*.
3. Multi-Agent Reinforcement Learning: A Review of Challenges and Applications. (2021). *Applied Sciences*, 11(11), 4948. MDPI.
4. LLMs for Customized Marketing Content Generation and Evaluation at Scale. (2025). *arXiv:2506.17863*.

## Technical Frameworks & Libraries

5. LangGraph — LangChain. Multi-agent orchestration framework. https://www.langchain.com/langgraph
6. FastAPI — Modern Python web framework. https://fastapi.tiangolo.com/
7. SimPy — Discrete-event simulation library. https://simpy.readthedocs.io/
8. MLflow — Open-source ML lifecycle platform. https://mlflow.org/
9. Ollama — Local LLM runtime. https://ollama.ai/
10. Prometheus — Monitoring and alerting toolkit. https://prometheus.io/
11. Grafana — Observability platform. https://grafana.com/
12. Streamlit — Data app framework. https://streamlit.io/
13. PostgreSQL — Relational database. https://www.postgresql.org/
14. pgvector — Vector similarity search for PostgreSQL. https://github.com/pgvector/pgvector
15. ChromaDB — Open-source vector database. https://www.trychroma.com/
16. Redis — In-memory data store. https://redis.io/
17. Docker Compose — Container orchestration. https://docs.docker.com/compose/
18. cAdvisor — Container resource monitoring. https://github.com/google/cadvisor

## Industry & Applied Research

19. Agent-Based Modeling in Marketing: Transforming Data-Driven Strategies. SmythOS. https://smythos.com/
20. Reinforcement Learning for Budget and Bid Optimization in Online Ad Auctions. (2025). EA Journals.
21. Maximizing Marketing Budget With AI and Machine Learning. Pecan AI.
22. Building Intelligent AI Agents with RL and LLMs. Medium.
23. The Impact of Multi-Agent Reinforcement Learning (MARL). Rapid Innovation.
24. Multi-Agent Reinforcement Learning for Marketing Optimization. sig.ai.
25. AI Agent Orchestration — IBM. https://www.ibm.com/think/topics/ai-agent-orchestration
26. Benchmarking Multi-Agent Architectures — LangChain Blog.
27. CrewAI vs. AutoGen: Comparing AI Agent Frameworks. Oxylabs.
28. MLOps Best Practices. Signity Solutions; Clarifai.
29. AI Agents with Human-in-the-Loop. Creatio.
30. LLM-based Agents Suffer from Hallucinations. (2025). *arXiv:2509.18970*.

## Thesis Templates & Guidelines

31. University of Oulu Master's Thesis Template (CSE/BA/BME).
32. University of Oulu Evaluation Instructions for Master's Thesis.

## Tools & Platform Documentation

33. Cal.com — Open-source scheduling. https://cal.com/
34. SendGrid — Email delivery platform. https://sendgrid.com/
35. Slack — Messaging platform. https://slack.com/
36. HubSpot — CRM platform. https://www.hubspot.com/
37. Apify — Web scraping platform. https://apify.com/

## Methodology & Best Practices

38. Thompson Sampling for contextual bandits — Bayesian approach to exploration-exploitation trade-off.
39. LinUCB — Linear upper confidence bound algorithm for contextual bandits.
40. Importance-Weighted Estimator (IWE) — Off-policy evaluation technique for bandit/RL.
41. Cohen's d — Standardized effect size measure for comparing group means.
42. Human-in-the-Loop (HITL) — Governance pattern for AI content approval pipelines.
"""
    
    with open(BASE_DIR / 'references.md', 'w') as f:
        f.write(refs)
    print("✅ references.md")

def generate_methodology():
    """Generate methodology documentation."""
    methodology = """# Research Methodology

## 1. Research Design

This research employs a **Design Science Research (DSR)** methodology, combining system implementation with rigorous empirical evaluation. The approach follows the standard DSR cycle:

1. **Problem Identification** — Autonomous marketing systems lack proper governance, safety, and attribution frameworks
2. **Solution Design** — Six-layer OODA-G architecture with multi-agent coordination
3. **Development** — Full-stack implementation (17 Docker containers, 16 dashboard pages)
4. **Evaluation** — Controlled experiments testing three hypotheses
5. **Communication** — Thesis document with reproducible results

## 2. Research Questions

- **RQ1:** How can a Multi-Agent System (MAS) be designed to autonomously manage LinkedIn marketing campaigns while maintaining brand safety and regulatory compliance?
- **RQ2:** How accurately can a SimPy-based simulation predict the performance of marketing strategies before live deployment?

## 3. Hypotheses

| ID | Hypothesis | Test Method | Success Criteria |
|----|-----------|-------------|------------------|
| H1 | MARL policy coordination outperforms individual contextual bandits in campaign optimization | Simulation A/B/C test (n=30 per group) | ≥20% lift with p<0.05 |
| H2 | LLM-generated content with safety governance outperforms template-based content | Safety score distribution analysis + CTR comparison | Safety score >0.7 mean; CTR uplift >10% |
| H3 | AgentOps approach reduces operational overhead by >50% compared to manual campaign management | Task completion rate + time-to-execution comparison | >50% reduction in human intervention |

## 4. Experimental Design

### 4.1 Simulation A/B/C Test (H1)

Three groups tested across 30 independent simulation runs of 100 steps each:

- **Group A (Baseline):** Rule-based automation — fixed strategy, no adaptation. 2% base conversion rate.
- **Group B (Bandit Policy):** Thompson Sampling contextual bandit — adapts based on click feedback. Learns to ~3.5% conversion rate.
- **Group C (MARL Policy):** Multi-agent reinforcement learning with coordination — agents share learning signals. Reaches ~5% conversion rate with coordination bonus.

Statistical significance tested via independent-samples t-test with Welch's correction.
Effect size measured via Cohen's d.

### 4.2 Content Safety Evaluation (H2)

- **Automated LLM-as-a-Judge:** Every generated content item scored on:
  - Toxicity (0-1, target <0.1)
  - Factuality (0-1, target >0.8)
  - Brand Alignment (0-1, target >0.8)
  - Overall Safety Score (composite)
- **Human-in-the-Loop Review:** Override rate tracked (target <5%)
- **Golden Test Suite:** Set of known-good/known-bad test cases

### 4.3 AgentOps Efficiency (H3)

- Autonomous task completion rate across agent types
- End-to-end campaign execution time (campaign creation → deployment)
- Human intervention frequency (HITL reviews per campaign)

## 5. Data Collection

All data extracted from live production system running 17 Docker containers:

| Data Source | Collection Method | Volume |
|------------|-------------------|--------|
| Campaign metrics | PostgreSQL direct query | All campaigns |
| Content governance | PostgreSQL + safety scores | All content items |
| Workflow events | Event log table | Up to 5000 events |
| Delayed rewards | Attribution ledger | All registered rewards |
| Bandit decisions | Decision log | Up to 10,000 decisions |
| Agent memory | Episodic memory table | All agent tasks |
| Experiment data | Experiment variants table | All experiments |
| Infrastructure | Prometheus metrics API | 6-hour windows |
| Container metrics | cAdvisor via Prometheus | 16 services (15 active) |

## 6. Evaluation Framework

The Unified KPI Dashboard (Section 10.2 of research plan) consolidates all metrics:

| Category | Metrics |
|----------|---------|
| Funnel-Specific | CPL, Booked Call Rate, Show Rate, Lead Quality |
| Engagement | CTR, Weekly Uplift Summary |
| LLM Safety | Toxicity, Factuality, Golden Test Pass Rate, Override Rate |
| Learning | OPE Lift, Bandit Regret |
| Cost & Efficiency | Cost-Per-Campaign, Cache Hit Rate, Human Time Saved |

## 7. Threats to Validity

| Threat | Category | Mitigation |
|--------|----------|------------|
| Simulated vs. real-world | External | SimPy calibrated against historical data; live bandit validation |
| Single company context | External | Architecture designed for generalization; parameterized personas |
| LLM model sensitivity | Internal | Ollama local (deterministic seed); safety scoring multi-dimensional |
| Small sample sizes | Statistical | 30 simulation runs per group; effect size reporting alongside p-values |
| Observer bias in HITL | Internal | Structured rubric; override rate tracking |

## 8. Ethical Considerations

- All content passes multi-stage safety governance before deployment
- HITL approval mandatory for production content
- No real user PII in research data
- Platform credentials handled via encrypted configuration service
- Budget guardrails prevent runaway spending
"""
    
    with open(BASE_DIR / 'methodology.md', 'w') as f:
        f.write(methodology)
    print("✅ methodology.md")

def generate_architecture_docs():
    """Generate architecture documentation."""
    arch_dir = BASE_DIR / 'architecture'
    arch_dir.mkdir(parents=True, exist_ok=True)
    
    doc = """# System Architecture Documentation

## 1. OODA-G Loop Architecture

The system implements a novel six-layer architecture based on the military OODA (Observe-Orient-Decide-Act) loop, extended with a Governance (G) layer for AI safety:

### Layer 1: Simulation Layer (Observe)
- **Technology:** SimPy discrete-event simulation
- **Purpose:** Digital twin of the marketing funnel; generates synthetic customer behavior for offline training and validation
- **Components:** Customer persona agents, competitor agents, market environment
- **Key Feature:** Calibrated against historical Agentic data for >90% accuracy target

### Layer 2: AI Layer (Orient & Decide)
- **Technology:** LangGraph (orchestration), Ollama (LLM), Custom RL
- **Purpose:** Content generation, strategy optimization, multi-agent coordination
- **Components:**
  - **LangGraph Supervisor:** Graph-based workflow orchestration with state machines
  - **Content Generator Agent:** Uses Ollama (llama3.2) for marketing copy generation
  - **Strategy Optimizer Agent:** Contextual bandits (Thompson Sampling, LinUCB) for arm selection
  - **MARL Policy Agent:** Multi-agent RL for coordinated campaign optimization
  - **RAG Pipeline:** Knowledge base retrieval via ChromaDB + pgvector

### Layer 3: Data Layer (Memory)
- **Technology:** PostgreSQL, pgvector, ChromaDB, Redis, MLflow
- **Purpose:** Persistent storage, vector similarity search, caching, experiment tracking
- **Components:**
  - PostgreSQL with pgvector for structured data + vector embeddings
  - ChromaDB for development RAG pipeline
  - Redis for semantic cache + task queue
  - MLflow for model versioning and experiment tracking
  - Agent episodic memory system for self-improvement

### Layer 4: Automation Layer (Act)
- **Technology:** API integrations (Cal.com, SendGrid, Slack, HubSpot, LinkedIn)
- **Purpose:** Execute marketing actions, deliver content, track leads
- **Components:**
  - LinkedIn posting (via approved schedulers/API)
  - SendGrid email alerts and notifications
  - Cal.com scheduling + delayed reward attribution
  - Slack alert delivery for governance notifications
  - HubSpot CRM integration for lead quality scoring

### Layer 5: Governance Layer (Govern)
- **Technology:** Custom HITL queue, LLM-as-a-Judge, safety scoring
- **Purpose:** Ensure brand safety, regulatory compliance, content quality
- **Components:**
  - Automated safety scoring (toxicity, factuality, brand alignment)
  - Human-in-the-Loop approval queue
  - Golden Test Suite (blocks deployment on failure)
  - Override rate monitoring (<5% target)
  - Canary deployments with automatic rollback

### Layer 6: Cost Control Facility (Govern)
- **Technology:** Token tracking, semantic cache, budget guardrails
- **Purpose:** Manage LLM API costs and campaign spending
- **Components:**
  - Per-campaign token tracking
  - Semantic cache (Redis) — target >20% hit rate
  - Budget guardrails — auto-pause at configurable thresholds
  - Cost-per-campaign reporting

## 2. Docker Compose Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Docker Compose Network                 │
├─────────────┬─────────────┬─────────────┬───────────────┤
│  Frontend   │   Backend   │    Data     │  Monitoring   │
├─────────────┼─────────────┼─────────────┼───────────────┤
│ Streamlit   │ FastAPI     │ PostgreSQL  │ Prometheus    │
│ (16 pages)  │ (REST API)  │ pgvector    │ Grafana       │
│             │ Worker (RQ) │ Redis       │ Loki          │
│             │ Nginx       │ ChromaDB    │ Promtail      │
│             │             │ MLflow      │ cAdvisor      │
│             │ Ollama      │             │ Node Exporter │
│             │ (LLM)       │             │ AlertManager  │
│             │             │             │ PG Exporter   │
└─────────────┴─────────────┴─────────────┴───────────────┘
                      16 services total
```

## 3. Campaign Workflow (State Machine)

```
DRAFT → CONTENT_GENERATION → SAFETY_CHECK → HITL_REVIEW
  → STRATEGY_OPTIMIZATION → MARL_EVALUATION → DEPLOYMENT
  → RUNNING → (budget/time triggers) → COMPLETED
```

Each transition is logged as a WorkflowEvent for full system transparency.

## 4. Data Flow

```
User creates campaign → LangGraph generates content →
Safety scoring (toxicity/factuality/brand) →
HITL queue (if needed) → Strategy optimization (bandits) →
MARL evaluation (OPE lift check) → Deployment →
LinkedIn posting → Impression/click tracking →
Lead registration → Cal.com booking → Delayed reward attribution →
Revenue attribution → Campaign completion
```

## 5. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Orchestration | LangGraph over CrewAI | Explicit state machine; graph-based flow control; native checkpointing |
| LLM | Ollama local over API | Zero cost for thesis; unlimited experimentation; no rate limits |
| Database | PostgreSQL over MongoDB | ACID compliance; pgvector native; mature ecosystem |
| Monitoring | Prometheus+Grafana over SaaS | Open-source; self-hosted; no vendor lock-in; thesis reproducibility |
| Frontend | Streamlit over React | Rapid prototyping; Python-native; 16 pages in weeks not months |
| Deployment | Docker Compose over K8s | Appropriate for thesis; K8s scaling guide provided for production |
"""
    
    with open(arch_dir / 'system_architecture.md', 'w') as f:
        f.write(doc)
    print("✅ architecture/system_architecture.md")

def generate_research_index():
    """Generate the master index of all research artifacts."""
    
    # Scan all generated files
    data_files = sorted((BASE_DIR / 'data').glob('*')) if (BASE_DIR / 'data').exists() else []
    viz_files = sorted((BASE_DIR / 'visualizations').glob('*')) if (BASE_DIR / 'visualizations').exists() else []
    table_files = sorted((BASE_DIR / 'tables').glob('*')) if (BASE_DIR / 'tables').exists() else []
    arch_files = sorted((BASE_DIR / 'architecture').glob('*')) if (BASE_DIR / 'architecture').exists() else []
    
    index = f"""# Thesis Research Data — Master Index

**Project:** Agentic AI Marketing Platform for Agentic
**University:** University of Oulu
**Program:** Computer Science and Engineering (CSE) — Master's Thesis
**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}

---

## Quick Navigation

| Chapter | Data Directory | Key Files |
|---------|---------------|-----------|
| Ch 1: Introduction | — | methodology.md |
| Ch 2: Literature Review | references.md | 42 references |
| Ch 3: Architecture | architecture/ | system_architecture.md, figures |
| Ch 4: Implementation | data/ | All extracted system data |
| Ch 5: Evaluation | tables/, visualizations/ | KPI tables, comparison figures |
| Ch 6: Conclusion | tables/hypothesis_evaluation.csv | Hypothesis summary |

---

## Directory Structure

```
agentic/thesis-research/
├── README.md              ← This file (master index)
├── methodology.md         ← Research design, hypotheses, evaluation framework
├── references.md          ← 42 compiled references
├── architecture/          ← System architecture documentation
│   └── system_architecture.md
├── data/                  ← Raw extracted data (JSON)
│   ├── campaigns.json
│   ├── content_governance.json
│   ├── workflow_events.json
│   ├── delayed_rewards.json
│   ├── experiments.json
│   ├── agent_memory.json
│   ├── bandit_decisions.json
│   ├── system_config.json
│   ├── canary_deployments.json
│   ├── marl_events.json
│   ├── marl_statistical_analysis.json
│   ├── container_metrics.json
│   ├── api_metrics.json
│   └── prometheus_targets.json
├── tables/                ← CSV and LaTeX tables
│   ├── unified_kpi_dashboard.csv/.tex
│   ├── campaign_summary.csv/.tex
│   ├── safety_analysis.csv
│   ├── agent_performance.csv/.tex
│   ├── experiment_results.csv
│   ├── technology_stack.csv/.tex
│   ├── hypothesis_evaluation.csv/.tex
│   ├── risk_register.csv/.tex
│   ├── marl_hypothesis_test.csv
│   └── container_resources.csv
├── visualizations/        ← Publication-quality figures (PNG, 150 DPI)
│   ├── fig_architecture_ooda_g.png
│   ├── fig_technology_stack.png
│   ├── fig_funnel_attribution.png
│   ├── fig_platform_performance.png
│   ├── fig_safety_scores.png
│   ├── fig_agent_learning.png
│   ├── fig_bandit_learning.png
│   ├── fig_workflow_events.png
│   ├── fig_campaign_lifecycle.png
│   ├── fig_marl_abc_test.png
│   ├── fig_ope_comparison.png
│   └── fig_infrastructure.png
└── evaluation/            ← Evaluation summaries and analysis
    └── (generated from data/)
```

---

## Data Files Detail

### data/ — Raw System Data
"""
    
    for f in data_files:
        try:
            size_kb = f.stat().st_size / 1024
            index += f"\n- **{f.name}** ({size_kb:.1f} KB)"
            if f.suffix == '.json':
                with open(f) as fh:
                    d = json.load(fh)
                    if 'total_campaigns' in d:
                        index += f" — {d['total_campaigns']} campaigns"
                    elif 'total_content' in d:
                        index += f" — {d['total_content']} content items"
                    elif 'total_events' in d:
                        index += f" — {d['total_events']} events"
                    elif 'total_rewards' in d:
                        index += f" — {d['total_rewards']} rewards"
                    elif 'total_experiments' in d:
                        index += f" — {d['total_experiments']} experiments"
                    elif 'total_memories' in d:
                        index += f" — {d['total_memories']} memories"
                    elif 'total_decisions' in d:
                        index += f" — {d['total_decisions']} decisions"
        except:
            index += f"\n- **{f.name}**"
    
    index += "\n\n### tables/ — Ready-to-Use Tables\n"
    for f in table_files:
        size_kb = f.stat().st_size / 1024
        index += f"\n- **{f.name}** ({size_kb:.1f} KB)"
    
    index += "\n\n### visualizations/ — Publication Figures\n"
    for f in viz_files:
        size_kb = f.stat().st_size / 1024
        index += f"\n- **{f.name}** ({size_kb:.1f} KB)"
    
    index += """

---

## Thesis Chapter Mapping

### Chapter 1: Introduction
- **Problem:** Agentic needs autonomous LinkedIn marketing with brand safety
- **Context:** B2B supplement company, targeting fitness professionals
- **RQ1:** MAS design for autonomous campaigns with safety/compliance
- **RQ2:** Simulation accuracy >90% for strategy validation
- **Data sources:** methodology.md, references.md

### Chapter 2: Literature Review
- Agent-based modeling in marketing (refs 1-4)
- LLMs for content generation (refs 9-15)
- Multi-agent reinforcement learning (refs 16-23)
- MAS orchestration frameworks (refs 24-31)
- MLOps and AgentOps (refs 46-49, 55-58)
- **Data sources:** references.md

### Chapter 3: System Architecture
- Six-layer OODA-G architecture diagram → fig_architecture_ooda_g.png
- Technology stack decision matrix → technology_stack.csv/tex
- Docker Compose deployment → fig_infrastructure.png
- Campaign state machine → architecture/system_architecture.md
- **Data sources:** architecture/, fig_technology_stack.png

### Chapter 4: Implementation
- Campaign workflow events → workflow_events.json, fig_workflow_events.png
- Content generation + RAG → content_governance.json
- Bandit algorithm implementation → bandit_decisions.json, fig_bandit_learning.png
- Agent memory system → agent_memory.json, fig_agent_learning.png
- Governance layer → safety_analysis.csv, fig_safety_scores.png
- System configuration → system_config.json
- **Data sources:** data/*.json, visualizations/

### Chapter 5: Evaluation & Results
- **H1 (MARL vs Bandits):** fig_marl_abc_test.png, marl_hypothesis_test.csv, marl_statistical_analysis.json
- **H2 (LLM Safety):** fig_safety_scores.png, safety_analysis.csv
- **H3 (AgentOps Efficiency):** agent_performance.csv, fig_agent_learning.png
- Unified KPI Dashboard → unified_kpi_dashboard.csv/tex
- Campaign performance → campaign_summary.csv, fig_platform_performance.png
- Marketing funnel → fig_funnel_attribution.png
- OPE results → fig_ope_comparison.png
- Risk register → risk_register.csv/tex
- **Data sources:** tables/, visualizations/

### Chapter 6: Conclusion
- Hypothesis evaluation summary → hypothesis_evaluation.csv/tex
- Limitations and threats to validity → methodology.md
- Future work directions → architecture/system_architecture.md

---

## How to Use This Data

1. **Tables:** CSV files can be opened in Excel or imported directly into LaTeX with `\\input{tables/filename.tex}`
2. **Figures:** PNG files at 150 DPI — suitable for direct inclusion in LaTeX with `\\includegraphics`
3. **Raw data:** JSON files contain complete extracted data — use for any additional analysis
4. **References:** Markdown file — convert to BibTeX as needed
5. **Architecture docs:** Markdown — include ASCII diagrams in thesis or recreate with draw.io

## Regenerating Data

All scripts are in `scripts/research/`:
```bash
# 1. Extract raw data from system
python scripts/research/01_extract_system_metrics.py

# 2. Generate visualizations
python scripts/research/02_generate_visualizations.py

# 3. Generate KPI tables
python scripts/research/03_generate_tables.py

# 4. Run MARL comparison analysis
python scripts/research/04_marl_comparison.py

# 5. Extract infrastructure metrics
python scripts/research/05_infrastructure_metrics.py

# 6. Generate index and documentation
python scripts/research/06_generate_index.py
```

**Important:** System must be running (all 17 Docker containers) for data extraction.
"""
    
    with open(BASE_DIR / 'README.md', 'w') as f:
        f.write(index)
    print("✅ README.md (master index)")

if __name__ == "__main__":
    print("=" * 60)
    print("THESIS RESEARCH: Generating Documentation & Index")
    print("=" * 60)
    
    print("\n📄 Generating references...")
    generate_references()
    
    print("\n📄 Generating methodology...")
    generate_methodology()
    
    print("\n📄 Generating architecture documentation...")
    generate_architecture_docs()
    
    print("\n📄 Generating master index...")
    generate_research_index()
    
    print("\n" + "=" * 60)
    print("✅ Documentation generation complete!")
    print(f"Output: {BASE_DIR}")
    print("=" * 60)
