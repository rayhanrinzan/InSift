"""Deployment-critical Streamlit configuration checks."""

from pathlib import Path
import tomllib


def test_streamlit_server_runs_headless() -> None:
    """Cloud startup must not pause for Streamlit's onboarding prompt."""

    config_path = Path(__file__).parents[1] / ".streamlit" / "config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["server"]["headless"] is True
