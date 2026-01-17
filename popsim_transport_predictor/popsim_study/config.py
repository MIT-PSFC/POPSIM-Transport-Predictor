from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class StudyConfig(BaseSettings):
    """Configuration for dataset paths."""

    cmod_dataset_path: Path | None = None
    tcv_dataset_path: Path | None = None
    d3d_lp_dataset_path: Path | None = None
    d3d_hp_dataset_path: Path | None = None

    model_config = SettingsConfigDict(
        env_prefix="PTPS_",  # Put in .env like PTPS_CMOD_DATASET_PATH
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = StudyConfig()
