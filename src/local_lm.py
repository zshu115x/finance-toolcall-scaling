"""
local_lm.py — Custom AbstractLanguageModel wrapping a local HuggingFace model.

Implements the its_hub AbstractLanguageModel interface so all scaling algorithms
(SelfConsistency, BestOfN) work against a locally-loaded transformers model
without needing a vLLM / Ollama server.

Design notes:
- asyncio.to_thread bridges synchronous pipeline() into the async interface
  that LMOrchestrator expects from agenerate_single().
- LoRA adapters are loaded via peft.PeftModel so the fine-tuned model can be
  evaluated with the same code path as the base model.
- Temperature is applied via do_sample=True when temperature > 0; greedy decode
  when temperature == 0 (budget=1 baseline).
"""

import asyncio
import json
import logging
from typing import Optional

from its_hub import AbstractLanguageModel
from its_hub.api.types import ChatMessage

logger = logging.getLogger(__name__)


class LocalHFLanguageModel(AbstractLanguageModel):
    """
    Wraps a HuggingFace causal LM for use with its_hub algorithms.

    Supports optional LoRA adapter loading so the same class serves both
    the base model and fine-tuned model in the experiment.
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        max_new_tokens: int = 384,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        tools: Optional[list[dict]] = None,
        torch_dtype: str = "float32",
    ):
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.tools = tools
        self.torch_dtype = torch_dtype
        self._model = None
        self._tokenizer = None

    def _load(self):
        """Lazy-load model and tokenizer on first use."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.model_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=getattr(torch, self.torch_dtype),
            device_map="auto",
            trust_remote_code=True,
        )

        if self.adapter_path:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter: {self.adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)
            self._model = self._model.merge_and_unload()  # merge for faster inference

        self._model.eval()
        logger.info("Model loaded.")

    def _format_messages(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert its_hub ChatMessage list to the dict format the tokenizer expects."""
        result = []
        if self.system_prompt and (not messages or messages[0].role != "system"):
            result.append({"role": "system", "content": self.system_prompt})
        for msg in messages:
            result.append({"role": msg.role, "content": msg.content or ""})
        return result

    def _run_inference(self, messages: list[ChatMessage]) -> dict:
        """Synchronous inference call — runs in a thread via asyncio.to_thread."""
        self._load()
        import torch

        chat_messages = self._format_messages(messages)
        text = self._tokenizer.apply_chat_template(
            chat_messages,
            tools=self.tools,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        do_sample = self.temperature > 0
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self._tokenizer.pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs["temperature"] = self.temperature

        with torch.no_grad():
            output_ids = self._model.generate(**gen_kwargs)

        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        generated = self._tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        # Try to extract a tool_calls field from JSON output.
        # Fine-tuned models may output a raw JSON tool call in their content;
        # parse it so the its_hub tool-voting machinery can work with it.
        tool_calls = _try_parse_tool_call(generated)
        if tool_calls:
            return {"role": "assistant", "content": None, "tool_calls": tool_calls}
        return {"role": "assistant", "content": generated}

    async def agenerate(
        self,
        messages: list[ChatMessage] | list[list[ChatMessage]],
        stop: Optional[str] = None,
        **kwargs,
    ) -> dict | list[dict]:
        """Batch or single generation (delegates to agenerate_single)."""
        if messages and isinstance(messages[0], list):
            return await asyncio.gather(
                *[self.agenerate_single(m, stop=stop, **kwargs) for m in messages]
            )
        return await self.agenerate_single(messages, stop=stop, **kwargs)

    async def agenerate_single(
        self,
        messages: list[ChatMessage],
        stop: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Single async generation via thread offload."""
        return await asyncio.to_thread(self._run_inference, messages)


# ── Tool-call parsing helpers ───────────────────────────────────────────────

_TOOL_CALL_TEMPLATE = {
    "id": "call_0",
    "type": "function",
    "function": {"name": "search_tool", "arguments": ""},
}


def _try_parse_tool_call(text: str) -> Optional[list[dict]]:
    """
    Try to extract a search_tool call from raw model output.

    The fine-tuned model is trained to output the tool call JSON directly.
    The base model may not — if parsing fails we return None and the caller
    treats the output as plain text content.

    Handles two formats:
      1. OpenAI tool_calls list:
         [{"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}]
      2. Plain object (the arguments dict):
         {"name": "search_tool", "arguments": {...}}
      3. Bare arguments dict (no "name" wrapper):
         {"query": "...", "company": "...", ...}
    """
    text = text.strip()

    # Strip Qwen2.5 <tool_call>...</tool_call> XML wrapper
    if "<tool_call>" in text:
        import re as _re
        m = _re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", text, _re.DOTALL)
        if m:
            text = m.group(1).strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting the first JSON object/array from mixed text
        import re
        match = re.search(r"(\[.*?\]|\{.*?\})", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    # Format 1: already a tool_calls list
    if isinstance(parsed, list) and parsed and "function" in parsed[0]:
        return parsed

    # Format 2: {"name": "search_tool", "arguments": {...}}
    if isinstance(parsed, dict) and parsed.get("name") == "search_tool":
        args = parsed.get("arguments", {})
        if isinstance(args, dict):
            args = json.dumps(args)
        tc = dict(_TOOL_CALL_TEMPLATE)
        tc["function"] = {"name": "search_tool", "arguments": args}
        return [tc]

    # Format 3: bare arguments dict (has "query" and "company" keys)
    if isinstance(parsed, dict) and "query" in parsed and "company" in parsed:
        tc = dict(_TOOL_CALL_TEMPLATE)
        tc["function"] = {"name": "search_tool", "arguments": json.dumps(parsed)}
        return [tc]

    return None


def extract_tool_args(response: dict) -> Optional[dict]:
    """
    Extract the search_tool arguments dict from an its_hub response.

    Works for both:
    - Responses with tool_calls (fine-tuned model or parsed output)
    - Plain content responses (fallback JSON parse)

    Returns None if no valid tool call can be extracted.
    """
    tool_calls = response.get("tool_calls") or []
    for tc in tool_calls:
        if isinstance(tc, dict) and tc.get("function", {}).get("name") == "search_tool":
            raw_args = tc["function"].get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    return json.loads(raw_args)
                except json.JSONDecodeError:
                    pass
            elif isinstance(raw_args, dict):
                return raw_args

    # Fallback: try to parse tool call from content
    content = response.get("content") or ""
    tool_calls_from_content = _try_parse_tool_call(content)
    if tool_calls_from_content:
        raw_args = tool_calls_from_content[0]["function"].get("arguments", "{}")
        if isinstance(raw_args, str):
            try:
                return json.loads(raw_args)
            except json.JSONDecodeError:
                pass
        elif isinstance(raw_args, dict):
            return raw_args

    return None
