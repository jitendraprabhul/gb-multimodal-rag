"""
Tests for core exception hierarchy.

Covers all custom exception classes, their attributes, and string representations.
"""

import pytest

from src.core.exceptions import (
    ConfigurationError,
    DocumentProcessingError,
    EmbeddingError,
    GraphError,
    GraphRAGError,
    LLMError,
    NERError,
    RetrievalError,
    ValidationError,
    VectorDBError,
)


class TestGraphRAGError:
    def test_basic_message(self):
        err = GraphRAGError("Something failed")
        assert str(err) == "Something failed"
        assert err.message == "Something failed"
        assert err.details == {}
        assert err.cause is None

    def test_with_details(self):
        err = GraphRAGError("Failed", details={"key": "value"})
        assert err.details == {"key": "value"}
        assert "Details: {'key': 'value'}" in str(err)

    def test_with_cause(self):
        cause = ValueError("original error")
        err = GraphRAGError("Failed", cause=cause)
        assert err.cause is cause
        assert "Caused by: original error" in str(err)

    def test_with_details_and_cause(self):
        cause = RuntimeError("boom")
        err = GraphRAGError("Failed", details={"op": "test"}, cause=cause)
        s = str(err)
        assert "Failed" in s
        assert "Details:" in s
        assert "Caused by:" in s

    def test_is_exception(self):
        err = GraphRAGError("test")
        assert isinstance(err, Exception)

    def test_none_details_default(self):
        err = GraphRAGError("test", details=None)
        assert err.details == {}


class TestConfigurationError:
    def test_inherits_from_base(self):
        err = ConfigurationError("Bad config")
        assert isinstance(err, GraphRAGError)
        assert isinstance(err, Exception)
        assert err.message == "Bad config"


class TestDocumentProcessingError:
    def test_with_document_info(self):
        err = DocumentProcessingError(
            "Parse failed",
            document_path="/path/to/doc.pdf",
            document_type="pdf",
        )
        assert err.details["document_path"] == "/path/to/doc.pdf"
        assert err.details["document_type"] == "pdf"

    def test_without_document_info(self):
        err = DocumentProcessingError("Parse failed")
        assert "document_path" not in err.details

    def test_with_cause(self):
        cause = IOError("file not found")
        err = DocumentProcessingError("Parse failed", cause=cause)
        assert err.cause is cause

    def test_inherits_from_base(self):
        assert issubclass(DocumentProcessingError, GraphRAGError)


class TestEmbeddingError:
    def test_with_model_info(self):
        err = EmbeddingError(
            "Embedding failed",
            model_name="bge-base",
            modality="text",
        )
        assert err.details["model_name"] == "bge-base"
        assert err.details["modality"] == "text"

    def test_without_model_info(self):
        err = EmbeddingError("Embedding failed")
        assert "model_name" not in err.details

    def test_inherits_from_base(self):
        assert issubclass(EmbeddingError, GraphRAGError)


class TestGraphError:
    def test_with_query_info(self):
        err = GraphError(
            "Query failed",
            query="MATCH (n) RETURN n",
            node_id="node-1",
        )
        assert err.details["query"] == "MATCH (n) RETURN n"
        assert err.details["node_id"] == "node-1"

    def test_without_extras(self):
        err = GraphError("Query failed")
        assert err.details == {}


class TestLLMError:
    def test_with_model_info(self):
        err = LLMError(
            "Generation failed",
            model="mistral:7b",
            prompt_length=500,
        )
        assert err.details["model"] == "mistral:7b"
        assert err.details["prompt_length"] == 500

    def test_without_extras(self):
        err = LLMError("timeout")
        assert err.details == {}


class TestRetrievalError:
    def test_with_query(self):
        err = RetrievalError(
            "Search failed",
            query="What is the revenue?",
            retrieval_type="hybrid",
        )
        assert err.details["query"] == "What is the revenue?"
        assert err.details["retrieval_type"] == "hybrid"

    def test_long_query_truncated(self):
        long_query = "x" * 200
        err = RetrievalError("Search failed", query=long_query)
        assert err.details["query"].endswith("...")
        assert len(err.details["query"]) == 103  # 100 + "..."

    def test_short_query_not_truncated(self):
        err = RetrievalError("Failed", query="short")
        assert err.details["query"] == "short"


class TestValidationError:
    def test_with_field_info(self):
        err = ValidationError(
            "Invalid field",
            field="email",
            value="not-an-email",
        )
        assert err.details["field"] == "email"
        assert err.details["value"] == "not-an-email"

    def test_long_value_truncated(self):
        long_value = "v" * 200
        err = ValidationError("Invalid", field="data", value=long_value)
        assert len(err.details["value"]) == 100

    def test_none_value_not_stored(self):
        err = ValidationError("Invalid", field="test", value=None)
        assert "value" not in err.details


class TestVectorDBError:
    def test_with_collection_info(self):
        err = VectorDBError(
            "Upsert failed",
            collection="text_chunks",
            operation="upsert",
        )
        assert err.details["collection"] == "text_chunks"
        assert err.details["operation"] == "upsert"

    def test_without_extras(self):
        err = VectorDBError("Connection lost")
        assert err.details == {}


class TestNERError:
    def test_with_model_info(self):
        err = NERError(
            "NER failed",
            model="en_core_web_sm",
            text_length=5000,
        )
        assert err.details["model"] == "en_core_web_sm"
        assert err.details["text_length"] == 5000

    def test_without_extras(self):
        err = NERError("Failed")
        assert err.details == {}

    def test_inherits_from_base(self):
        assert issubclass(NERError, GraphRAGError)


class TestExceptionHierarchy:
    """Verify all exceptions inherit from GraphRAGError."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            DocumentProcessingError,
            EmbeddingError,
            GraphError,
            LLMError,
            RetrievalError,
            ValidationError,
            VectorDBError,
            NERError,
        ],
    )
    def test_inherits_from_graphrag_error(self, exc_class):
        assert issubclass(exc_class, GraphRAGError)
        assert issubclass(exc_class, Exception)

    @pytest.mark.parametrize(
        "exc_class",
        [
            ConfigurationError,
            DocumentProcessingError,
            EmbeddingError,
            GraphError,
            LLMError,
            RetrievalError,
            ValidationError,
            VectorDBError,
            NERError,
        ],
    )
    def test_can_be_raised_and_caught(self, exc_class):
        with pytest.raises(GraphRAGError):
            raise exc_class("test error")
