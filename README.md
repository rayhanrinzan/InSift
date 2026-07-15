# InSift

InSift is a production-minded MVP for discovering evidence-backed startup opportunities from online discussions. It is designed to connect every opportunity to real user pain, cluster similar problems, research competitors, and rank opportunities with explainable scores.

This repository implements Phases 1-8: the project foundation, ingestion, evidence-grounded extraction, semantic clustering, competitor research, researched white-space scoring, ranked opportunities, auditable user corrections, and production-minded application polish.

## Main Features

- Typed settings loaded from environment variables or `.env`
- Structured JSON logging without API key leakage
- SQLAlchemy models for evidence, clusters, competitors, scores, and user feedback
- SQLite local development database with PostgreSQL-ready configuration
- Alembic migration support
- Repository layer for database persistence
- Deterministic demo seed data
- Manual, CSV, Reddit URL, subreddit, and keyword ingestion with duplicate protection
- Bounded Reddit OAuth collection with source attribution and configurable limits
- Deterministic evidence pre-filter and OpenAI structured problem extraction
- Offline mock extraction that works without an API key
- OpenAI and local Sentence Transformers embeddings, plus deterministic demo embeddings
- Incremental cosine-similarity clustering with recomputed centroids
- Evidence-backed cluster summaries with source and author independence counts
- Explainable Problem, Opportunity, and Confidence scores
- Broad competitor-query generation with every query stored for auditability
- Tavily search integration with bounded retries and deterministic mock search
- OpenAI structured direct, adjacent, substitute, and irrelevant classification
- Deduplicated competitor persistence with source snippets and reasoning
- Five-component White-Space Score based on need, differentiation, weaknesses, niche, and density
- Evidence, customer, and competitor corrections with feedback history
- Cluster merge and split workflows with automatic score recomputation
- Ranked and filterable Streamlit opportunity views with evidence drill-down
- Responsive page shell with consistent navigation and operational status
- Paginated opportunity, evidence, competitor, query, and feedback views
- Cached read models with automatic invalidation after writes
- Loading states and live progress for ingestion, scoring, and competitor research
- Actionable database and provider error messages without credential exposure
- Docker packaging, health checks, and PostgreSQL deployment guidance
- Pytest coverage for ingestion, extraction, clustering, scoring, and persistence

## Architecture

```mermaid
flowchart TD
    UI[Streamlit UI] --> Services[Service layer]
    Scripts[CLI scripts] --> Services
    Services --> Repositories[Repository layer]
    Repositories --> Models[SQLAlchemy models]
    Models --> DB[(SQLite local / PostgreSQL production)]
    Alembic[Alembic migrations] --> DB
    Reddit[Reddit OAuth] --> Services
    Extractors[Mock / OpenAI structured output] --> Services
    Embeddings[Demo hash / Sentence Transformers / OpenAI] --> Services
    Search[Mock / Tavily search provider] --> Services
    Corrections[Correction service + audit history] --> Services
```

## Data Pipeline

```text
Online discussions
        ↓
Evidence filtering
        ↓
Structured problem extraction
        ↓
Semantic clustering
        ↓
Opportunity generation
        ↓
Competitor research
        ↓
White-space analysis
        ↓
Explainable scoring
        ↓
Ranked opportunity dashboard
```

## Scoring Methodology

Scores are stored as versioned `OpportunityScore` records with structured explanations. Phases 4 and 6 calculate:

- Problem Score from pain severity, frequency, willingness to pay, and evidence quality
- White-Space Score from unmet need, differentiation, competitor weakness, niche specificity, and direct-competitor density
- Opportunity Score from problem strength, white-space, feasibility, and accessibility
- Confidence Score from source independence, extraction confidence, recency, agreement among sources, and available research coverage

```text
Problem Score = 35% Pain Severity + 25% Problem Frequency
              + 20% Willingness to Pay + 20% Evidence Quality

Opportunity Score = 25% Pain Severity + 15% Problem Frequency
                  + 15% Willingness to Pay + 10% Evidence Quality
                  + 15% White-Space + 10% Build Feasibility
                  + 10% Market Accessibility
```

Before research, white-space is neutral and explicitly labeled. After research, its five components are calculated from stored evidence, queries, competitor weaknesses, and supported gaps. Missing competitor results are never automatically rewarded. Build feasibility and market accessibility remain neutral pending dedicated validation.

```text
White-Space Score = 30% Unmet Customer Need
                  + 25% Differentiation Potential
                  + 20% Competitor Weakness
                  + 15% Niche Specificity
                  + 10% Low Direct-Competitor Density
```

The confidence score is intentionally separate from the opportunity score. A cluster may look promising while still having limited evidence.

## Local Setup

Run every command from the repository root, the directory containing `requirements.txt` and `streamlit_app.py`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python scripts/initialize_database.py
python scripts/seed_demo_data.py
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`. If that port is occupied, Streamlit selects another local port and prints it in the terminal.

## Live Mode

Live mode never silently falls back to deterministic providers. Open Settings in the running application and configure:

- OpenAI API key and model for extraction and competitor classification
- OpenAI or Sentence Transformers embeddings
- Tavily API key for competitor research
- Reddit OAuth client ID, client secret, and a descriptive user agent

Settings are written to the ignored local `.env` file. Existing secret values are never displayed and blank password fields preserve the stored value. For a clean local live database, use `DATABASE_URL=sqlite:///insift_live.db`, set `APP_ENV=production` and `DEMO_MODE=false`, then run:

```bash
python scripts/initialize_database.py
python -m streamlit run streamlit_app.py
```

OpenAI requests use strict structured JSON output and disable response storage for submitted source text. Hosted deployments should inject the same settings through their secret manager instead of the in-app local file.

## Environment Variables

```text
APP_ENV=development
LOG_LEVEL=INFO
DATABASE_URL=sqlite:///insift.db
LLM_PROVIDER=
LLM_API_KEY=
LLM_MODEL=gpt-5.6-luna
OPENAI_BASE_URL=https://api.openai.com/v1
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
SEARCH_PROVIDER=
SEARCH_API_KEY=
SEARCH_DEPTH=basic
CLUSTER_SIMILARITY_THRESHOLD=0.78
MINIMUM_EXTRACTION_CONFIDENCE=0.45
MAX_SEARCH_RESULTS=10
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=InSift/1.0 by configured-user
DEMO_MODE=true
```

API keys should be stored in `.env` or deployment secrets. Do not commit them.

## Demo Mode

Demo mode is enabled by default. It uses deterministic extraction, embeddings, search results, and classification. The seed script inserts evidence, clusters, research queries, competitors, researched score breakdowns, and correction-ready records so the complete Phase 1-8 workflow can be explored without paid APIs.

```bash
python scripts/initialize_database.py
python scripts/seed_demo_data.py
streamlit run streamlit_app.py
```

The seed script is safe to run repeatedly; it avoids duplicating demo evidence and competitors.

To ingest one discussion from the command line:

```bash
python scripts/ingest_sources.py --text "We still use Excel and this manual process takes hours every week."
```

To ingest a CSV, include one of `raw_text`, `text`, `body`, `discussion`, or `content` as a column:

```bash
python scripts/ingest_sources.py --csv discussions.csv --max-rows 100
python scripts/score_opportunities.py
```

## Testing

```bash
pytest
```

Tests cover persistence, ingestion validation, OpenAI request and retry behavior, Reddit OAuth normalization, extraction failures, clustering boundaries, query generation, competitor classification and deduplication, researched white-space, confidence, corrections, merge/split behavior, audit history, and score recomputation.

## Database Migrations

```bash
alembic upgrade head
```

For local development, `python scripts/initialize_database.py` can create tables directly from SQLAlchemy metadata. Production-style deployments should prefer Alembic migrations.

## Deployment

The repository includes a non-root Docker image, Streamlit health check, secret-safe build context, and a restrained application theme.

```bash
docker build -t insift .
```

Production deployments should use PostgreSQL, run `alembic upgrade head` as a release step, and provide API keys through a managed secret store. See [Deployment](docs/DEPLOYMENT.md) for Docker demo commands, PostgreSQL setup, Streamlit Community Cloud guidance, health checks, and rollback notes.

## Known Limitations

- Reddit ingestion requires formally approved OAuth credentials and intentionally caps each collection request at 100 items.
- The deterministic demo embedding is intentionally small. Live mode can use OpenAI or the configured local Sentence Transformers model.
- Merge operations archive the source cluster rather than deleting its historical scores and feedback.
- Real search currently supports Tavily; Brave Search remains a future provider adapter.
- Streamlit read caches are process-local with a 30-second TTL; horizontally scaled instances may briefly show different snapshots after a write.
- SQLite is intended for local demos. Hosted deployments require a persistent PostgreSQL database.

## Platform Data-Use Warning

Do not build or operate InSift as an unrestricted Reddit scraper. Review each platform's terms before commercial use. The MVP should preserve source attribution, collect only necessary text, use formally permitted API access where available, and avoid using collected content for model training.

## Future Roadmap

1. Add a Brave Search adapter and more permitted source-specific ingestion adapters.
2. Add richer feasibility and market-access validation inputs.
3. Add authentication, workspaces, and role-aware correction approval.
4. Add distributed cache invalidation and production tracing for multi-instance deployments.
