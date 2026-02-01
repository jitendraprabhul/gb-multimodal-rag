"""
Tests for application settings.

Covers configuration, validation, domain-based model selection, and paths.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from config.settings import Domain, Environment, Settings, get_settings


class TestDomainEnum:
    def test_values(self):
        assert Domain.FINANCE == "finance"
        assert Domain.HEALTHCARE == "healthcare"

    def test_length(self):
        assert len(Domain) == 2


class TestEnvironmentEnum:
    def test_values(self):
        assert Environment.DEVELOPMENT == "development"
        assert Environment.STAGING == "staging"
        assert Environment.PRODUCTION == "production"


class TestSettings:
    def test_defaults(self):
        # Note: Settings loads from environment variables including .env file
        # so we test the values that would be loaded in dev environment
        settings = Settings()
        assert settings.app_name == "graphrag-multimodal"
        assert settings.app_env == Environment.DEVELOPMENT
        # debug value comes from .env (DEBUG=true) if present, otherwise False
        assert isinstance(settings.debug, bool)
        assert settings.log_level == "INFO"
        assert settings.domain == Domain.FINANCE

    def test_api_defaults(self):
        settings = Settings()
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000
        assert settings.api_workers == 4

    def test_neo4j_defaults(self):
        settings = Settings()
        assert settings.neo4j_uri == "bolt://localhost:7687"
        assert settings.neo4j_user == "neo4j"

    def test_qdrant_defaults(self):
        settings = Settings()
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.qdrant_grpc_port == 6334

    def test_embedding_defaults(self):
        settings = Settings()
        assert settings.text_embedding_model == "BAAI/bge-base-en-v1.5"
        assert settings.image_embedding_model == "openai/clip-vit-base-patch32"
        assert settings.embedding_device == "cpu"
        assert settings.embedding_batch_size == 32

    def test_chunking_defaults(self):
        settings = Settings()
        assert settings.chunk_size == 512
        assert settings.chunk_overlap == 50
        assert settings.max_chunk_size == 1024

    def test_retrieval_defaults(self):
        settings = Settings()
        assert settings.top_k_vector == 20
        assert settings.top_k_final == 5
        assert settings.graph_hop_limit == 2
        assert settings.max_graph_nodes == 1000


class TestSettingsValidation:
    def test_valid_log_level(self):
        settings = Settings(log_level="DEBUG")
        assert settings.log_level == "DEBUG"

    def test_log_level_case_insensitive(self):
        settings = Settings(log_level="debug")
        assert settings.log_level == "DEBUG"

    def test_invalid_log_level_raises(self):
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID")

    def test_api_port_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            Settings(api_port=0)

    def test_api_port_too_high_raises(self):
        with pytest.raises(ValidationError):
            Settings(api_port=70000)

    def test_chunk_size_too_small_raises(self):
        with pytest.raises(ValidationError):
            Settings(chunk_size=50)

    def test_chunk_size_too_large_raises(self):
        with pytest.raises(ValidationError):
            Settings(chunk_size=3000)

    def test_graph_hop_limit_bounds(self):
        with pytest.raises(ValidationError):
            Settings(graph_hop_limit=0)
        with pytest.raises(ValidationError):
            Settings(graph_hop_limit=6)

    def test_embedding_batch_size_positive(self):
        with pytest.raises(ValidationError):
            Settings(embedding_batch_size=0)


class TestSettingsProperties:
    def test_ner_model_finance(self):
        settings = Settings(domain=Domain.FINANCE)
        assert settings.ner_model == settings.ner_model_finance

    def test_ner_model_healthcare(self):
        settings = Settings(domain=Domain.HEALTHCARE)
        assert settings.ner_model == settings.ner_model_healthcare

    def test_is_production_true(self):
        settings = Settings(app_env=Environment.PRODUCTION)
        assert settings.is_production is True

    def test_is_production_false(self):
        settings = Settings(app_env=Environment.DEVELOPMENT)
        assert settings.is_production is False

    def test_is_production_staging(self):
        settings = Settings(app_env=Environment.STAGING)
        assert settings.is_production is False


class TestSettingsPaths:
    def test_default_paths(self):
        settings = Settings()
        assert settings.raw_docs_path == Path("./data/raw")
        assert settings.processed_docs_path == Path("./data/processed")
        assert settings.models_cache_path == Path("./models")
        assert settings.temp_path == Path("./tmp")

    def test_ensure_paths(self, tmp_path):
        settings = Settings(
            raw_docs_path=tmp_path / "raw",
            processed_docs_path=tmp_path / "processed",
            models_cache_path=tmp_path / "models",
            temp_path=tmp_path / "tmp",
        )
        settings.ensure_paths()
        assert (tmp_path / "raw").exists()
        assert (tmp_path / "processed").exists()
        assert (tmp_path / "models").exists()
        assert (tmp_path / "tmp").exists()

    def test_ensure_paths_nested(self, tmp_path):
        settings = Settings(
            raw_docs_path=tmp_path / "a" / "b" / "c",
            processed_docs_path=tmp_path / "processed",
            models_cache_path=tmp_path / "models",
            temp_path=tmp_path / "tmp",
        )
        settings.ensure_paths()
        assert (tmp_path / "a" / "b" / "c").exists()


class TestGetSettings:
    def test_returns_settings(self):
        # Clear cache first
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_cached(self):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
        get_settings.cache_clear()


class TestSettingsEnvironmentOverride:
    def test_override_via_env(self):
        with patch.dict(os.environ, {"LOG_LEVEL": "ERROR"}):
            settings = Settings()
            assert settings.log_level == "ERROR"

    def test_override_domain_via_env(self):
        with patch.dict(os.environ, {"DOMAIN": "healthcare"}):
            settings = Settings()
            assert settings.domain == Domain.HEALTHCARE
