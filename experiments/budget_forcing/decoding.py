"""
Budget Forcing — Decoding-time intervention để kiểm soát compute.
Implement theo paper: s1: Simple Test-Time Scaling (EMNLP 2025).

Hai chế độ:
  - enforce_maximum: Khi token count > budget → chèn EoT + "Final Answer:"
  - enforce_minimum: Khi model sinh EoT → suppress, append trigger phrase

Tham khảo: Section 3.1 của paper.
"""

from __future__ import annotations
from typing import Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel


# ── Constants ──────────────────────────────────────────────────────────────────

# Token / string mà model dùng để kết thúc phần suy nghĩ (thinking)
DEFAULT_EOT_STRINGS = [
    "<|im_end|>",
    "</think>",
    "####",            # Thường dùng trong GSM8K/DeepSeek
    "\n\nFinal Answer:",
    "\n\nAnswer:",
]

DEFAULT_TRIGGER = "Wait"          # Trigger mặc định (từ paper)
FINAL_ANSWER_PREFIX = "\n\nFinal Answer:"


# ── Core Class ─────────────────────────────────────────────────────────────────

class BudgetForcingDecoder:
    """
    Wrapper quanh model.generate() thêm logic Budget Forcing.

    Ví dụ sử dụng:
        decoder = BudgetForcingDecoder(model, tokenizer)

        # Tăng compute: append "Wait" 2 lần nếu model muốn dừng sớm
        output = decoder.generate(
            input_ids,
            min_thinking_tokens=None,   # không enforce min
            max_thinking_tokens=4096,
            n_wait=2,
            trigger="Wait",
        )
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        eot_strings: list[str] | None = None,
        device: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.eot_strings = eot_strings or DEFAULT_EOT_STRINGS

        # Pre-encode EoT tokens for fast lookup
        self.eot_token_ids: set[int] = set()
        for s in self.eot_strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            # We only support single-token EoT for per-token suppression
            if len(ids) == 1:
                self.eot_token_ids.add(ids[0])
            elif ids:
                # If multi-token, we still add the first token to catch it,
                # though this is a heuristic and might have false positives
                # for common tokens like '\n'. 
                # TODO: Implement proper multi-token sequence suppression
                if s not in ["\n\nFinal Answer:", "\n\nAnswer:"]:
                    self.eot_token_ids.add(ids[0])

        # Always include the model's native EOS token
        if tokenizer.eos_token_id is not None:
            self.eot_token_ids.add(tokenizer.eos_token_id)

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 2048,
        max_thinking_tokens: Optional[int] = None,
        n_wait: int = 0,
        trigger: str = DEFAULT_TRIGGER,
        temperature: float = 0.0,
        do_sample: bool = False,
    ) -> dict:
        """
        Generate với Budget Forcing.

        Args:
            input_ids:            Tensor [1, seq_len]
            max_new_tokens:       Tổng số token tối đa sinh ra
            max_thinking_tokens:  Nếu set, enforce MAXIMUM: dừng thinking sau N tokens
            n_wait:               Số lần append trigger (enforce MINIMUM)
            trigger:              Chuỗi append khi model cố dừng ("Wait", ...)
            temperature:          Sampling temperature (0 = greedy)
            do_sample:            True nếu dùng sampling

        Returns:
            dict với keys: 'output_ids', 'thinking_text', 'answer_text',
                           'thinking_tokens', 'n_waits_triggered'
        """
        self.model.eval()
        current_ids = input_ids.to(self.device)

        generated_tokens = []
        waits_triggered = 0
        thinking_ended = False
        thinking_token_count = 0

        with torch.no_grad():
            for step in range(max_new_tokens):
                # ── Forward pass ──────────────────────────────────────────────
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]  # [1, vocab_size]

                # ── Sample or greedy ──────────────────────────────────────────
                if do_sample and temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                token_id = next_token.item()

                # ── Enforce MAXIMUM: dừng thinking sớm ───────────────────────
                if (
                    max_thinking_tokens is not None
                    and not thinking_ended
                    and thinking_token_count >= max_thinking_tokens
                ):
                    # Chèn EoT + Final Answer prompt, bỏ qua token vừa sinh
                    fa_ids = self.tokenizer.encode(
                        FINAL_ANSWER_PREFIX, add_special_tokens=False
                    )
                    generated_tokens.extend(fa_ids)
                    current_ids = torch.cat(
                        [current_ids, torch.tensor([fa_ids], device=self.device)],
                        dim=1,
                    )
                    thinking_ended = True
                    continue  # tiếp tục sinh phần answer

                # ── Enforce MINIMUM: suppress EoT, append trigger ─────────────
                if (
                    not thinking_ended
                    and token_id in self.eot_token_ids
                    and waits_triggered < n_wait
                ):
                    # Suppress EoT token, append trigger phrase thay vào
                    trigger_ids = self.tokenizer.encode(
                        f"\n{trigger}", add_special_tokens=False
                    )
                    generated_tokens.extend(trigger_ids)
                    current_ids = torch.cat(
                        [current_ids, torch.tensor([trigger_ids], device=self.device)],
                        dim=1,
                    )
                    waits_triggered += 1
                    continue

                # ── Normal token ──────────────────────────────────────────────
                generated_tokens.append(token_id)
                current_ids = torch.cat(
                    [current_ids, next_token.to(self.device)], dim=1
                )

                if not thinking_ended:
                    thinking_token_count += 1
                    if token_id in self.eot_token_ids:
                        thinking_ended = True

                # Dừng khi gặp EOS
                if token_id == self.tokenizer.eos_token_id:
                    break

        # ── Decode kết quả ────────────────────────────────────────────────────
        full_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        thinking_text, answer_text = self._split_thinking_answer(full_text)

        return {
            "output_ids": generated_tokens,
            "full_text": full_text,
            "thinking_text": thinking_text,
            "answer_text": answer_text,
            "thinking_tokens": thinking_token_count,
            "n_waits_triggered": waits_triggered,
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _split_thinking_answer(self, text: str | list[str]) -> tuple[str, str]:
        """Tách thinking trace và final answer từ full generated text."""
        if isinstance(text, list):
            text = "".join(text)
        
        delimiters = ["</think>", FINAL_ANSWER_PREFIX, "\n\nAnswer:", "####"]
        for delimiter in delimiters:
            if delimiter in text:
                parts = text.split(delimiter, 1)
                return parts[0].strip(), parts[1].strip()
        
        # Nếu không tìm thấy delimiter, coi toàn bộ là thinking
        return text.strip(), ""
