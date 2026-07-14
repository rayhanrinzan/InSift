"""Shared pytest fixtures."""

from collections.abc import Generator

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.base import Base
from src.database import models  # noqa: F401


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    """Return an isolated in-memory SQLite session."""

    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    SessionFactory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with SessionFactory() as session:
        yield session
    Base.metadata.drop_all(bind=engine)
