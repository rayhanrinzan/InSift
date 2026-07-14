"""Initialize the local InSift database."""

import logging
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_settings, redacted_database_url
from src.database.session import create_database_engine, initialize_database
from src.logging_config import log_event, setup_logging


logger = logging.getLogger(__name__)


def main() -> None:
    """Create database tables for local development."""

    settings = get_settings()
    setup_logging(settings)
    engine = create_database_engine(settings)
    initialize_database(engine)
    log_event(
        logger,
        logging.INFO,
        "database_initialized",
        {"database_url": redacted_database_url(settings.database_url)},
    )
    print(f"Initialized database at {redacted_database_url(settings.database_url)}")


if __name__ == "__main__":
    main()
