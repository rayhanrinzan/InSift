"""Database engine and session helpers."""

from collections.abc import Generator
from typing import Optional

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings, get_settings
from src.database.base import Base
from src.database import models  # noqa: F401


def create_database_engine(settings: Optional[Settings] = None) -> Engine:
    """Create a SQLAlchemy engine from settings."""

    settings = settings or get_settings()
    connect_args = {}
    if settings.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(settings.database_url, connect_args=connect_args, future=True)


def create_session_factory(engine: Optional[Engine] = None) -> sessionmaker[Session]:
    """Create a session factory bound to an engine."""

    engine = engine or create_database_engine()
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


SessionLocal = create_session_factory()


def initialize_database(engine: Optional[Engine] = None) -> None:
    """Create all configured tables for local development."""

    engine = engine or create_database_engine()
    Base.metadata.create_all(bind=engine)


def get_session() -> Generator[Session, None, None]:
    """Yield a database session for framework integrations."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
