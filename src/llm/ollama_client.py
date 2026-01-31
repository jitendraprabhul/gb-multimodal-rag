"""
Ollama client for local LLM inference.

Provides:
- Text generation
- Chat completion
- Streaming support
- Model management
"""

import asyncio
from typing import Any, AsyncIterator

import httpx

from src.core.exceptions import LLMError
from src.core.logging import LoggerMixin


class OllamaClient(LoggerMixin):
    """
    Async client for Ollama API.

    Supports local LLM inference with Mistral, Llama,
    and other models served by Ollama.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "mistral:7b-instruct",
        timeout: int = 120,
        **config: Any,
    ) -> None:
        """
        Initialize Ollama client.

        Args:
            host: Ollama server URL
            model: Default model to use
            timeout: Request timeout in seconds
            **config: Additional configuration
        """
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.config = config

        self._client: httpx.AsyncClient | None = None
        self._initialized = False
        self._available_models: list[str] = []

    async def initialize(self) -> None:
        """Initialize HTTP client and verify connection."""
        if self._initialized:
            return

        self._client = httpx.AsyncClient(
            base_url=self.host,
            timeout=httpx.Timeout(self.timeout),
        )

        try:
            # Check connection and list models
            response = await self._client.get("/api/tags")
            response.raise_for_status()

            data = response.json()
            self._available_models = [
                m["name"] for m in data.get("models", [])
            ]

            self._initialized = True
            self.logger.info(
                "Ollama client initialized",
                host=self.host,
                models=self._available_models,
            )

            # Check if default model is available
            if self.model not in self._available_models:
                self.logger.warning(
                    f"Default model {self.model} not found. "
                    f"Available: {self._available_models}"
                )

        except httpx.ConnectError as e:
            raise LLMError(
                f"Cannot connect to Ollama at {self.host}. Is it running?",
                cause=e,
            )
        except Exception as e:
            raise LLMError(
                f"Failed to initialize Ollama client: {e}",
                cause=e,
            )

    async def cleanup(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: Input prompt
            model: Model to use (default: self.model)
            system: System prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if not self._initialized:
            await self.initialize()

        model = model or self.model

        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system:
            request_data["system"] = system

        if stop:
            request_data["options"]["stop"] = stop

        try:
            response = await self._client.post(
                "/api/generate",
                json=request_data,
            )
            response.raise_for_status()

            data = response.json()
            return data.get("response", "")

        except httpx.TimeoutException:
            raise LLMError(
                f"Ollama request timed out after {self.timeout}s",
                model=model,
                prompt_length=len(prompt),
            )
        except Exception as e:
            raise LLMError(
                f"Generation failed: {e}",
                model=model,
                prompt_length=len(prompt),
                cause=e,
            )

    async def generate_stream(
        self,
        prompt: str,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Generate text with streaming.

        Args:
            prompt: Input prompt
            model: Model to use
            system: System prompt
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Yields:
            Generated text chunks
        """
        if not self._initialized:
            await self.initialize()

        model = model or self.model

        request_data = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        if system:
            request_data["system"] = system

        try:
            async with self._client.stream(
                "POST",
                "/api/generate",
                json=request_data,
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if line:
                        import json

                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]

                        if data.get("done", False):
                            break

        except Exception as e:
            raise LLMError(
                f"Streaming generation failed: {e}",
                model=model,
                cause=e,
            )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> str:
        """
        Chat completion.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            **kwargs: Additional parameters

        Returns:
            Assistant response
        """
        if not self._initialized:
            await self.initialize()

        model = model or self.model

        request_data = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
        }

        try:
            response = await self._client.post(
                "/api/chat",
                json=request_data,
            )
            response.raise_for_status()

            data = response.json()
            message = data.get("message", {})
            return message.get("content", "")

        except Exception as e:
            raise LLMError(
                f"Chat completion failed: {e}",
                model=model,
                cause=e,
            )

    async def is_model_available(self, model: str) -> bool:
        """Check if a model is available."""
        if not self._initialized:
            await self.initialize()

        return model in self._available_models

    async def pull_model(self, model: str) -> bool:
        """
        Pull a model from Ollama registry.

        Args:
            model: Model name to pull

        Returns:
            True if successful
        """
        if not self._initialized:
            await self.initialize()

        try:
            self.logger.info(f"Pulling model: {model}")

            response = await self._client.post(
                "/api/pull",
                json={"name": model, "stream": False},
                timeout=httpx.Timeout(600.0),  # 10 minutes for download
            )
            response.raise_for_status()

            # Refresh available models
            response = await self._client.get("/api/tags")
            data = response.json()
            self._available_models = [
                m["name"] for m in data.get("models", [])
            ]

            return model in self._available_models

        except Exception as e:
            self.logger.error(f"Failed to pull model {model}: {e}")
            return False

    async def get_model_info(self, model: str) -> dict[str, Any]:
        """Get information about a model."""
        if not self._initialized:
            await self.initialize()

        try:
            response = await self._client.post(
                "/api/show",
                json={"name": model},
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            self.logger.warning(f"Failed to get model info: {e}")
            return {}

    @property
    def available_models(self) -> list[str]:
        """Get list of available models."""
        return self._available_models.copy()

    async def __aenter__(self) -> "OllamaClient":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.cleanup()
