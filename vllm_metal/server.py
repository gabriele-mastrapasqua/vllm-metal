# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible API server using MLX backend.

This provides a standalone server with the same API as vLLM,
but using MLX for inference on Apple Silicon.

Usage:
    python -m vllm_metal.server --model Qwen/Qwen3-0.6B --port 8000

Or via the CLI:
    vllm-metal serve Qwen/Qwen3-0.6B
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

import mlx.core as mx
import mlx_lm
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# Pydantic Models (OpenAI API compatible)
# ============================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = None
    stream: bool = False
    n: int = 1


class CompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int | None = 100
    stream: bool = False
    n: int = 1


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "vllm-metal"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ============================================================
# MLX Model Server
# ============================================================


class MLXModelServer:
    """Server that wraps MLX-LM for OpenAI-compatible inference."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model using MLX-LM."""
        logger.info(f"Loading model: {self.model_name}")
        start = time.time()
        self.model, self.tokenizer = mlx_lm.load(self.model_name)
        load_time = time.time() - start
        logger.info(f"Model loaded in {load_time:.2f}s")

        # Warmup
        logger.info("Warming up model...")
        _ = mlx_lm.generate(
            self.model, self.tokenizer, prompt="Hello", max_tokens=5, verbose=False
        )
        mx.eval([])
        logger.info("Model ready!")

    def _format_chat_prompt(self, messages: list[ChatMessage]) -> str:
        """Format chat messages into a prompt string."""
        # Simple chat format - can be improved for specific models
        formatted = ""
        for msg in messages:
            if msg.role == "system":
                formatted += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                formatted += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                formatted += f"Assistant: {msg.content}\n\n"
        formatted += "Assistant: "
        return formatted

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> tuple[str, int, int]:
        """Generate completion.

        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens)
        """
        prompt_tokens = len(self.tokenizer.encode(prompt))

        response = mlx_lm.generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
        )
        mx.eval([])

        # Extract just the new tokens
        generated = response[len(prompt) :]
        completion_tokens = len(self.tokenizer.encode(generated))

        return generated, prompt_tokens, completion_tokens

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> AsyncIterator[str]:
        """Stream generated tokens."""
        # Use mlx_lm streaming
        prev_text = prompt
        for response in mlx_lm.stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            new_text = response[len(prev_text) :]
            if new_text:
                yield new_text
                prev_text = response
            await asyncio.sleep(0)  # Allow other tasks to run


# ============================================================
# FastAPI Application
# ============================================================


def create_app(model_name: str) -> FastAPI:
    """Create FastAPI application with MLX backend."""

    app = FastAPI(
        title="vLLM Metal Server",
        description="OpenAI-compatible API server using MLX on Apple Silicon",
        version="0.1.0",
    )

    # Initialize model server
    server = MLXModelServer(model_name)

    @app.get("/health")
    async def health():
        return {"status": "healthy"}

    @app.get("/v1/models")
    async def list_models() -> ModelList:
        return ModelList(
            data=[
                ModelInfo(
                    id=model_name,
                    created=int(time.time()),
                )
            ]
        )

    @app.post("/v1/completions")
    async def create_completion(request: CompletionRequest):
        """Create a text completion."""
        prompts = (
            [request.prompt] if isinstance(request.prompt, str) else request.prompt
        )

        if request.stream:

            async def stream_response():
                for prompt in prompts:
                    async for token in server.generate_stream(
                        prompt,
                        max_tokens=request.max_tokens or 100,
                        temperature=request.temperature,
                        top_p=request.top_p,
                    ):
                        chunk = {
                            "id": f"cmpl-{uuid.uuid4().hex[:8]}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": request.model,
                            "choices": [
                                {"index": 0, "text": token, "finish_reason": None}
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        # Non-streaming
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for i, prompt in enumerate(prompts):
            text, prompt_tokens, completion_tokens = server.generate(
                prompt,
                max_tokens=request.max_tokens or 100,
                temperature=request.temperature,
                top_p=request.top_p,
            )
            choices.append(CompletionChoice(index=i, text=text, finish_reason="stop"))
            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=choices,
            usage=Usage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    @app.post("/v1/chat/completions")
    async def create_chat_completion(request: ChatCompletionRequest):
        """Create a chat completion."""
        prompt = server._format_chat_prompt(request.messages)
        max_tokens = request.max_tokens or 500

        if request.stream:

            async def stream_response():
                request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                async for token in server.generate_stream(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                ):
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

                # Final chunk
                final = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(stream_response(), media_type="text/event-stream")

        # Non-streaming
        text, prompt_tokens, completion_tokens = server.generate(
            prompt,
            max_tokens=max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="vLLM Metal Server")
    parser.add_argument("command", nargs="?", default="serve", help="Command (serve)")
    parser.add_argument("model", nargs="?", help="Model name (e.g., Qwen/Qwen3-0.6B)")
    parser.add_argument("--model", dest="model_flag", help="Model name")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")

    args = parser.parse_args()

    model_name = args.model or args.model_flag
    if not model_name:
        parser.error("Model name required. Usage: vllm-metal serve Qwen/Qwen3-0.6B")

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    vLLM Metal Server                             ║
║          OpenAI-compatible API on Apple Silicon                  ║
╚══════════════════════════════════════════════════════════════════╝

Model: {model_name}
Host:  {args.host}
Port:  {args.port}

Starting server...
""")

    app = create_app(model_name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
