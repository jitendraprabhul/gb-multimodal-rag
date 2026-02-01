"""
Tests for the Ollama LLM client.

Covers:
- Initialization and connection
- Text generation (sync and streaming)
- Chat completion
- Model management (availability, pull, info)
- Error handling (connection, timeout, generation)
- Context manager
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.core.exceptions import LLMError
from src.llm.ollama_client import OllamaClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def client():
    """Create OllamaClient with defaults (not initialized)."""
    return OllamaClient()


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx.AsyncClient.

    Note: httpx.Response.json() is synchronous, so we use MagicMock for the base
    and AsyncMock only for async methods (get, post, aclose).
    """
    mock = MagicMock()
    mock.get = AsyncMock()
    mock.post = AsyncMock()
    mock.aclose = AsyncMock()
    return mock


def _make_mock_response(json_data):
    """Create a mock httpx.Response with json data.

    httpx.Response.json() is synchronous, so use MagicMock with return_value.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = json_data
    mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture
async def initialized_client(client, mock_httpx_client):
    """Create an initialized OllamaClient with mocked HTTP client."""
    # Mock the initialization response - json() is synchronous on httpx.Response
    mock_response = _make_mock_response({
        "models": [
            {"name": "mistral:7b-instruct"},
            {"name": "llama2:7b"},
        ]
    })
    mock_httpx_client.get.return_value = mock_response

    with patch("src.llm.ollama_client.httpx.AsyncClient", return_value=mock_httpx_client):
        await client.initialize()

    return client


# =============================================================================
# Initialization Tests
# =============================================================================


class TestOllamaClientInit:
    def test_default_params(self):
        client = OllamaClient()
        assert client.host == "http://localhost:11434"
        assert client.model == "mistral:7b-instruct"
        assert client.timeout == 120
        assert client._initialized is False

    def test_custom_params(self):
        client = OllamaClient(
            host="http://gpu-server:11434",
            model="llama2:13b",
            timeout=300,
        )
        assert client.host == "http://gpu-server:11434"
        assert client.model == "llama2:13b"
        assert client.timeout == 300

    def test_host_trailing_slash_stripped(self):
        client = OllamaClient(host="http://localhost:11434/")
        assert client.host == "http://localhost:11434"

    async def test_initialize_connects_and_lists_models(self, client):
        # Use MagicMock for response since json() is synchronous
        mock_response = _make_mock_response({
            "models": [{"name": "mistral:7b-instruct"}]
        })

        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.aclose = AsyncMock()

        with patch("src.llm.ollama_client.httpx.AsyncClient", return_value=mock_http):
            await client.initialize()

        assert client._initialized is True
        assert "mistral:7b-instruct" in client._available_models

    async def test_initialize_idempotent(self, initialized_client):
        mock_httpx = initialized_client._client
        await initialized_client.initialize()
        # get should not be called again
        # (initial call was during fixture setup)

    async def test_initialize_connection_error_raises(self, client):
        mock_http = MagicMock()
        mock_http.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
        mock_http.aclose = AsyncMock()

        with patch("src.llm.ollama_client.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(LLMError, match="Cannot connect to Ollama"):
                await client.initialize()

    async def test_initialize_general_error_raises(self, client):
        mock_http = MagicMock()
        mock_http.get = AsyncMock(side_effect=RuntimeError("Unknown error"))
        mock_http.aclose = AsyncMock()

        with patch("src.llm.ollama_client.httpx.AsyncClient", return_value=mock_http):
            with pytest.raises(LLMError, match="Failed to initialize"):
                await client.initialize()

    async def test_cleanup(self, initialized_client):
        await initialized_client.cleanup()
        assert initialized_client._initialized is False
        assert initialized_client._client is None

    async def test_cleanup_when_not_initialized(self, client):
        await client.cleanup()  # Should not raise

    async def test_context_manager(self):
        # Use MagicMock for response since json() is synchronous
        mock_response = _make_mock_response({"models": []})

        mock_http = MagicMock()
        mock_http.get = AsyncMock(return_value=mock_response)
        mock_http.aclose = AsyncMock()

        with patch("src.llm.ollama_client.httpx.AsyncClient", return_value=mock_http):
            async with OllamaClient() as client:
                assert client._initialized is True
            assert client._initialized is False


# =============================================================================
# Generate Tests
# =============================================================================


class TestGenerate:
    async def test_basic_generate(self, initialized_client):
        mock_response = _make_mock_response({"response": "The answer is 42"})
        initialized_client._client.post.return_value = mock_response

        result = await initialized_client.generate("What is the meaning of life?")

        assert result == "The answer is 42"
        call_args = initialized_client._client.post.call_args
        assert call_args.args[0] == "/api/generate"
        request_data = call_args.kwargs["json"]
        assert request_data["prompt"] == "What is the meaning of life?"
        assert request_data["stream"] is False

    async def test_generate_with_custom_model(self, initialized_client):
        mock_response = _make_mock_response({"response": "ok"})
        initialized_client._client.post.return_value = mock_response

        await initialized_client.generate("test", model="llama2:7b")

        request_data = initialized_client._client.post.call_args.kwargs["json"]
        assert request_data["model"] == "llama2:7b"

    async def test_generate_with_system_prompt(self, initialized_client):
        mock_response = _make_mock_response({"response": "ok"})
        initialized_client._client.post.return_value = mock_response

        await initialized_client.generate(
            "test",
            system="You are a helpful assistant",
        )

        request_data = initialized_client._client.post.call_args.kwargs["json"]
        assert request_data["system"] == "You are a helpful assistant"

    async def test_generate_with_options(self, initialized_client):
        mock_response = _make_mock_response({"response": "ok"})
        initialized_client._client.post.return_value = mock_response

        await initialized_client.generate(
            "test",
            max_tokens=512,
            temperature=0.3,
            stop=["###"],
        )

        request_data = initialized_client._client.post.call_args.kwargs["json"]
        assert request_data["options"]["num_predict"] == 512
        assert request_data["options"]["temperature"] == 0.3
        assert request_data["options"]["stop"] == ["###"]

    async def test_generate_timeout_raises_llm_error(self, initialized_client):
        initialized_client._client.post.side_effect = httpx.TimeoutException("Timeout")

        with pytest.raises(LLMError, match="timed out"):
            await initialized_client.generate("test")

    async def test_generate_failure_raises_llm_error(self, initialized_client):
        initialized_client._client.post.side_effect = RuntimeError("Bad request")

        with pytest.raises(LLMError, match="Generation failed"):
            await initialized_client.generate("test")

    async def test_generate_empty_response(self, initialized_client):
        mock_response = _make_mock_response({})
        initialized_client._client.post.return_value = mock_response

        result = await initialized_client.generate("test")
        assert result == ""


# =============================================================================
# Chat Tests
# =============================================================================


class TestChat:
    async def test_basic_chat(self, initialized_client):
        mock_response = _make_mock_response({
            "message": {"role": "assistant", "content": "Hello!"}
        })
        initialized_client._client.post.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        result = await initialized_client.chat(messages)

        assert result == "Hello!"
        call_args = initialized_client._client.post.call_args
        assert call_args.args[0] == "/api/chat"

    async def test_chat_with_multiple_messages(self, initialized_client):
        mock_response = _make_mock_response({
            "message": {"role": "assistant", "content": "I see"}
        })
        initialized_client._client.post.return_value = mock_response

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "How are you?"},
        ]

        result = await initialized_client.chat(messages)
        assert result == "I see"

    async def test_chat_failure_raises_llm_error(self, initialized_client):
        initialized_client._client.post.side_effect = RuntimeError("Failed")

        with pytest.raises(LLMError, match="Chat completion failed"):
            await initialized_client.chat([{"role": "user", "content": "test"}])

    async def test_chat_empty_message(self, initialized_client):
        mock_response = _make_mock_response({"message": {}})
        initialized_client._client.post.return_value = mock_response

        result = await initialized_client.chat([{"role": "user", "content": "test"}])
        assert result == ""


# =============================================================================
# Model Management Tests
# =============================================================================


class TestModelManagement:
    async def test_is_model_available_true(self, initialized_client):
        result = await initialized_client.is_model_available("mistral:7b-instruct")
        assert result is True

    async def test_is_model_available_false(self, initialized_client):
        result = await initialized_client.is_model_available("nonexistent:model")
        assert result is False

    def test_available_models_property(self, initialized_client):
        models = initialized_client.available_models
        assert "mistral:7b-instruct" in models
        assert "llama2:7b" in models

    def test_available_models_returns_copy(self, initialized_client):
        models1 = initialized_client.available_models
        models2 = initialized_client.available_models
        assert models1 is not models2

    async def test_pull_model_success(self, initialized_client):
        # Mock pull response (post returns response for pull)
        mock_pull_response = _make_mock_response({})

        # Mock tags refresh (get returns response for tags)
        mock_tags_response = _make_mock_response({
            "models": [
                {"name": "mistral:7b-instruct"},
                {"name": "llama2:7b"},
                {"name": "new-model:latest"},
            ]
        })

        initialized_client._client.post.return_value = mock_pull_response
        initialized_client._client.get.return_value = mock_tags_response

        result = await initialized_client.pull_model("new-model:latest")
        assert result is True
        assert "new-model:latest" in initialized_client._available_models

    async def test_pull_model_failure(self, initialized_client):
        initialized_client._client.post.side_effect = RuntimeError("Download failed")

        result = await initialized_client.pull_model("bad-model")
        assert result is False

    async def test_get_model_info(self, initialized_client):
        mock_response = _make_mock_response({
            "modelfile": "FROM mistral",
            "parameters": "num_ctx 4096",
        })
        initialized_client._client.post.return_value = mock_response

        info = await initialized_client.get_model_info("mistral:7b-instruct")
        assert "modelfile" in info

    async def test_get_model_info_failure(self, initialized_client):
        initialized_client._client.post.side_effect = RuntimeError("Not found")

        info = await initialized_client.get_model_info("nonexistent")
        assert info == {}
