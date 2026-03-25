"""
SemanticDensityAgent
--------------------
Input  : transcript text only  (plain string)
Type   : LLM-based
Library: OpenRouter (OpenAI-compatible API; any model can be used via config)

Score semantics
---------------
  0.0 → dense, articulate speech  (low cognitive load)
  1.0 → vague, repetitive speech  (high cognitive load)

Robustness features
-------------------
  - JSON parse errors are retried once with a stricter extraction prompt
  - Long transcripts are truncated at MAX_TRANSCRIPT_CHARS with a warning
  - A fallback score of 0.5 is returned if both attempts fail
  - LLM response is sanitised (strips markdown fences) before JSON parsing
"""

import os
import re
import json
import asyncio
import dotenv
from typing import Optional
from pathlib import Path
from openai import AsyncOpenAI

dotenv.load_dotenv()

OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"
DEFAULT_MODEL        = "meta-llama/llama-3.1-8b-instruct"
MAX_TRANSCRIPT_CHARS = 4000   # ~800 words — enough context without blowing tokens


# JSON extraction helpers 

def _strip_markdown(text: str) -> str:
    """Remove ```json … ``` fences that some models add."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _extract_json(raw: str) -> dict:
    """Try to parse JSON; also attempts to extract the first {...} block."""
    raw = _strip_markdown(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the first valid JSON object in the response
        match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


class SemanticDensityAgent:
    """
    Wraps an async LLM call in a class so config, caching, and
    persistence mirror the other agents in the pipeline.

    Uses OpenRouter's endpoint so any model
    (Claude, GPT-4o, Gemini, Mistral …) can be swapped via the
    'model' argument without touching the rest of the code.
    """

    def __init__(
        self,
        model:        str  = DEFAULT_MODEL,
        max_tokens:   int  = 200,
        invert_score: bool = True,   # True → high density = low load
    ):
        self.model          = model
        self.max_tokens     = max_tokens
        self.invert_score   = invert_score
        self.last_result: Optional[dict] = None

    # Truncation helper
    def _smart_truncate(self, text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
        """Take 30% from start, 70% from end — keeps context and peaks later in speech."""
        if len(text) <= max_chars:
            return text
        
        head_chars = int(max_chars * 0.30)
        tail_chars = max_chars - head_chars
        
        head = text[:head_chars]
        tail = text[-tail_chars:]
        
        return head + "\n...[truncated]...\n" + tail

    # Prompt builders 

    @staticmethod
    def _build_prompt(transcript_text: str, total_words: int = None, question_text: str = None) -> str:
        # Calculate word count if not provided
        if total_words is None:
            total_words = len(transcript_text.split())
        
        # Build context string
        context_line = f"Note: this is a {total_words}-word spoken answer"
        if question_text:
            context_line += f" to the question: '{question_text}'."
            context_line += "\nRate density relative to what's possible in a spoken answer of this length."
        else:
            context_line += ".\nRate density relative to what's possible in a spoken answer of this length."
        
        return f"""You are a cognitive-load researcher analysing speech transcripts.

Rate the SEMANTIC DENSITY of the transcript below on a scale from 0.0 to 1.0:
0.0 = very low density  (repetitive, vague, lots of filler, circular reasoning)
0.5 = moderate          (some structure, some repetition, partial ideas)
1.0 = very high density (precise, information-rich, tightly argued, complex ideas)

{context_line}

Transcript:
\"\"\"{transcript_text}\"\"\"

Rules:
- Reply with ONLY a valid JSON object — no markdown fences, no preamble.
- Format: {{"density_score": <float 0.0-1.0>, "reasoning": "<one concise sentence>"}}
- density_score must be a number between 0.0 and 1.0 (inclusive).
"""

    @staticmethod
    def _build_retry_prompt(transcript_text: str, total_words: int = None, question_text: str = None) -> str:
        """Stricter prompt used on the second attempt."""
        if total_words is None:
            total_words = len(transcript_text.split())
        
        context_line = f"This is a {total_words}-word spoken answer"
        if question_text:
            context_line += f" to: '{question_text}'"
        context_line += ". Rate density for this length."
        
        return f"""Analyse this speech transcript and output ONLY valid JSON.

{context_line}

Transcript: "{transcript_text[:500]}"

Output format (no other text):
{{"density_score": 0.5, "reasoning": "your one-sentence explanation"}}

Replace 0.5 with your actual score between 0.0 and 1.0.
"""

    # Main compute 

    async def compute(self, transcript_text: str, client: AsyncOpenAI, question_text: str = None) -> dict:
        """
        Parameters
        ----------
        transcript_text : raw transcript string
        client          : openai.AsyncOpenAI pointed at OpenRouter
        question_text   : (optional) the question that prompted this response

        Returns
        -------
        {density_score, reasoning, score, truncated, model_used}
        """
        if not isinstance(transcript_text, str) or not transcript_text.strip():
            raise ValueError("transcript_text must be a non-empty string")

        # Count words in original text before truncation
        total_words = len(transcript_text.split())

        # Smart truncation: take 30% from start, 70% from end
        original_len = len(transcript_text)
        transcript_text = self._smart_truncate(transcript_text)
        truncated = original_len > MAX_TRANSCRIPT_CHARS
        if truncated:
            print(
                f"  [semantic_density] WARNING: transcript truncated to "
                f"{MAX_TRANSCRIPT_CHARS} chars (was {original_len})."
            )

        # First attempt 
        parsed = await self._call_llm(client, self._build_prompt(transcript_text, total_words, question_text))

        # Retry on parse failure 
        if parsed is None:
            print("  [semantic_density] Parse failed — retrying with stricter prompt …")
            parsed = await self._call_llm(client, self._build_retry_prompt(transcript_text, total_words, question_text))

        # Fallback 
        if parsed is None:
            print("  [semantic_density] WARNING: using fallback score 0.5")
            parsed = {"density_score": 0.5, "reasoning": "Fallback — LLM did not return valid JSON."}

        density = float(parsed["density_score"])
        density = max(0.0, min(1.0, density))   # clamp just in case

        score = round(1.0 - density, 3) if self.invert_score else round(density, 3)

        result = {
            "density_score": round(density, 3),
            "reasoning":     parsed.get("reasoning", ""),
            "score":         score,          # 0 = articulate (low load), 1 = vague (high load)
            "truncated":     truncated,
            "model_used":    self.model,
        }
        self.last_result = result
        return result

    async def _call_llm(self, client: AsyncOpenAI, prompt: str, temperature: float = 0.0) -> Optional[dict]:
        """Call the LLM and attempt JSON extraction. Returns None on any failure."""
        try:
            response = await client.chat.completions.create(
                model      = self.model,
                max_tokens = self.max_tokens,
                messages   = [{"role": "user", "content": prompt}],
                temperature= temperature,
            )
            raw = response.choices[0].message.content.strip()
            return _extract_json(raw)
        except Exception as exc:
            print(f"  [semantic_density] LLM call / parse error: {exc}")
            return None

    # Persistence

    def save(self, path: str) -> None:
        data = {
            "model":        self.model,
            "max_tokens":   self.max_tokens,
            "invert_score": self.invert_score,
            "last_result":  self.last_result,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.model        = data["model"]
        self.max_tokens   = data["max_tokens"]
        self.invert_score = data["invert_score"]
        self.last_result  = data.get("last_result")


# Standalone runner 

async def main():
    DATA_DIR   = Path("baby-data/transcripts")
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH  = OUTPUT_DIR / "semantic_density_metrics.json"
    LABELS     = ["low", "medium", "high"]

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set.\n"
            "Add it to your .env file or export it: set OPENROUTER_API_KEY=sk-or-..."
        )

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)

    client = AsyncOpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    agent  = SemanticDensityAgent(model=model, invert_score=True)

    print("=== Semantic Density Analysis (OpenRouter) ===")
    print(f"    model : {model}\n")

    for label in LABELS:
        data            = json.load(open(DATA_DIR / f"{label}_transcript.json"))
        transcript_text = data["text"]

        result = await agent.compute(transcript_text, client)
        print(f"  [{label.upper()}]")
        print(f"    density_score : {result['density_score']}  (1 = information-rich)")
        print(f"    score         : {result['score']}  (0 = articulate, 1 = vague / high load)")
        print(f"    reasoning     : {result['reasoning']}")
        if result.get("truncated"):
            print(f"    [!] transcript was truncated before scoring")
        print()

    agent.save(SAVE_PATH)
    print(f"Saved config + last result → {SAVE_PATH}")

    agent2 = SemanticDensityAgent()
    agent2.load(SAVE_PATH)
    print(f"Re-loaded model      : {agent2.model}")
    print(f"Re-loaded last_result: {agent2.last_result}")


if __name__ == "__main__":
    asyncio.run(main())