# FlowSift AI Deployment

FlowSift AI can run as a local SQLite demo, a Docker container, or a hosted Streamlit application backed by PostgreSQL. Production deployments should use PostgreSQL and execute Alembic migrations before starting the web process.

## Runtime Requirements

- Python 3.11 or newer
- A persistent PostgreSQL database for production
- Outbound HTTPS access to OpenAI, Tavily, and Reddit when live providers are enabled
- Environment variables supplied by the hosting platform's secret manager

Never bake `.env`, API keys, or a local database file into an image.

## Docker Demo

Build the image from the repository root:

```bash
docker build -t flowsift-ai .
docker volume create flowsift-ai-data
```

Initialize and seed a persistent SQLite demo volume:

```bash
docker run --rm \
  --env DATABASE_URL=sqlite:////data/flowsift.db \
  --volume flowsift-ai-data:/data \
  flowsift-ai python scripts/initialize_database.py

docker run --rm \
  --env DATABASE_URL=sqlite:////data/flowsift.db \
  --volume flowsift-ai-data:/data \
  flowsift-ai python scripts/seed_demo_data.py
```

Start the application:

```bash
docker run --rm \
  --publish 8501:8501 \
  --env DATABASE_URL=sqlite:////data/flowsift.db \
  --volume flowsift-ai-data:/data \
  flowsift-ai
```

Open `http://localhost:8501`. The container health check uses Streamlit's `/_stcore/health` endpoint.

## PostgreSQL Deployment

Set a SQLAlchemy-compatible connection string through the platform secret manager:

```text
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@HOST:5432/DATABASE
```

Run the migration as a release command or one-off job:

```bash
alembic upgrade head
```

Then start the service:

```bash
streamlit run streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=${PORT:-8501}
```

Do not run the demo seed script in a production database unless deterministic sample records are explicitly wanted.

## Streamlit Community Cloud

1. Create an application from the GitHub repository.
2. Set the entrypoint to `streamlit_app.py`.
3. Select Python 3.11 in Advanced settings.
4. Add root-level values in the app's secrets settings for `DATABASE_URL`, provider names, model names, API keys, and Reddit OAuth credentials. Root-level Streamlit secrets are exposed as environment variables.
5. Use a hosted PostgreSQL URL. The local SQLite filesystem is not durable across redeployments.
6. FlowSift AI creates the current schema automatically for an empty database. Run `alembic upgrade head` before deployment when upgrading an existing production database.

## Production Checklist

- Set `APP_ENV=production` and `DEMO_MODE=false`.
- Set `LLM_PROVIDER=openai`, `EMBEDDING_PROVIDER=openai` or `sentence_transformers`, and `SEARCH_PROVIDER=tavily`.
- Set `LOG_LEVEL=INFO` or the platform's preferred level.
- Use managed PostgreSQL with encrypted connections, backups, and restricted credentials.
- Store `LLM_API_KEY`, `SEARCH_API_KEY`, and `REDDIT_CLIENT_SECRET` only in a secret manager.
- Configure `REDDIT_CLIENT_ID` and a descriptive `REDDIT_USER_AGENT` before enabling Reddit intake.
- Restrict database network access to the application environment.
- Route container health checks to `/_stcore/health`.
- Capture structured JSON logs and alert on extraction, search, and database failures.
- Review source-platform terms and retention requirements before collecting real discussion data.
- Confirm that application instances share the same database; the 30-second UI cache is process-local and automatically expires.

## Rollback

Deploy the previous application image, then use an explicit Alembic downgrade only after reviewing whether the migration is reversible without data loss:

```bash
alembic history
alembic downgrade -1
```

Database backups are the primary rollback mechanism for production data.
