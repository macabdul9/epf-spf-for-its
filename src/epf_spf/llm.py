"""OpenAI-compatible client for the generator (a vLLM server).

Request fields match what the previous implementation sent: a continuation turn
sets add_generation_prompt=False and continue_final_message=True so vLLM extends the
assistant message instead of opening a new turn.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI

from epf_spf.steps import StepFormat


@dataclass(frozen=True)
class GenerationConfig:
    sampling_temperature: float = 0.8
    max_step_tokens: int = 300
    min_tokens: int | None = None
    think_trigger: bool = False


class GeneratorClient:
    def __init__(
        self,
        endpoint: str,
        model: str,
        config: GenerationConfig,
        timeout_s: float = 1800.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.config = config
        self.client = AsyncOpenAI(
            base_url=endpoint, api_key="EMPTY", timeout=timeout_s, max_retries=max_retries
        )

    async def next_step(self, base_messages: list[dict], steps: list[str], fmt: StepFormat) -> str:
        messages = list(base_messages)
        extra_body: dict = {"include_stop_str_in_output": False}
        if steps:
            messages.append({"role": "assistant", "content": fmt.continuation(steps)})
            extra_body |= {"add_generation_prompt": False, "continue_final_message": True}
            if self.config.min_tokens is not None:
                extra_body["min_tokens"] = self.config.min_tokens
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stop=fmt.stop_strings,
            max_tokens=self.config.max_step_tokens,
            temperature=self.config.sampling_temperature,
            extra_body=extra_body,
        )
        return completion.choices[0].message.content or ""

    async def next_steps(
        self, base_messages: list[dict], steps_per_particle: list[list[str]], fmt: StepFormat
    ) -> list[str]:
        return list(
            await asyncio.gather(
                *(self.next_step(base_messages, steps, fmt) for steps in steps_per_particle)
            )
        )

    async def forced_answer_logprobs(
        self, base_messages: list[dict], assistant_prefix: str, max_tokens: int = 3
    ) -> list[list[tuple[str, float]]]:
        """Greedy continuation of `assistant_prefix`; per generated position, the top-20
        (token, logprob) pairs. Used by the answer probe."""
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=[*base_messages, {"role": "assistant", "content": assistant_prefix}],
            max_tokens=max_tokens,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
            extra_body={"add_generation_prompt": False, "continue_final_message": True},
        )
        logprobs = completion.choices[0].logprobs
        if logprobs is None or not logprobs.content:
            return []
        return [
            [(e.token, e.logprob) for e in (pos.top_logprobs or [])] for pos in logprobs.content
        ]

    async def close(self) -> None:
        await self.client.close()
