"""
CoherenceAgent
--------------
Input  : question text + verbatim transcript
Type   : LLM-based
Library: OpenRouter (OpenAI-compatible API)

Scores three content dimensions of a spoken response:
  RELEVANCE    — did the speaker address what was asked?
  COMPLETENESS — how developed is the answer?
  COHERENCE    — is the reasoning logically connected?

Each dimension is scored 1–5 (null for very short transcripts).
The composite score is the mean of available dimensions, normalised to [0, 1].

Design decisions
----------------
  - "Score content only, not fluency" — acoustic agents already capture
    disfluency; this agent must emit an orthogonal signal to avoid double-counting.
  - Null for short/garbled transcripts — better than a hallucinated score.
  - Confidence field in the LLM response — mirrors confidence_score.py so
    the composite_coherence_score integrates naturally with the aggregator.
  - No explanation requested — schema-only JSON output is more stable.

Robustness features
-------------------
  - JSON parse errors are retried once with a stricter prompt
  - Long transcripts are truncated at MAX_TRANSCRIPT_CHARS with a warning
  - A fallback score of 0.5 is returned if both LLM attempts fail
  - LLM response is sanitised (strips markdown fences) before JSON parsing
"""

import os
import re
import json
import asyncio
import dotenv
from typing import Dict, Optional
from pathlib import Path
from openai import AsyncOpenAI

# Import z-score helper
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from z_score import compute_z_score

dotenv.load_dotenv()

OPENROUTER_BASE_URL  = "https://openrouter.ai/api/v1"
DEFAULT_MODEL        = "mistralai/mistral-small-3.1-24b-instruct"
MAX_TRANSCRIPT_CHARS = 4000   # ~800 words — enough context without blowing tokens

# Raw dimension scale
DIM_MIN, DIM_MAX = 1.0, 5.0


# JSON extraction helpers

def _strip_markdown(text: str) -> str:
    """Remove ```json … ``` fences that some models add."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def _extract_json(raw: str) -> dict:
    """Try to parse JSON; also attempts to extract the first {...} block."""
    raw = _strip_markdown(raw)
    
    # First attempt: direct JSON parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    
    # Second attempt: find {...} block
    match = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Third attempt: more complex nested structures
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    raise json.JSONDecodeError(f"Could not extract valid JSON from: {raw[:200]}", raw, 0)


def _normalise_dim(value: Optional[float]) -> Optional[float]:
    """Map a 1–5 dimension score onto [0, 1]. Returns None if value is None."""
    if value is None:
        return None
    clamped = max(DIM_MIN, min(DIM_MAX, float(value)))
    return round((clamped - DIM_MIN) / (DIM_MAX - DIM_MIN), 3)


def _composite(relevance: Optional[float], completeness: Optional[float], coherence: Optional[float]) -> float:
    """Mean of available (non-None) normalised scores; fallback 0.5."""
    values = [v for v in (relevance, completeness, coherence) if v is not None]
    return round(sum(values) / len(values), 3) if values else 0.5


class CoherenceAgent:
    """
    Wraps an async LLM call in a class so config, caching, and
    persistence mirror the other agents in the pipeline.

    Uses OpenRouter's endpoint for diverse model selection.
    Any model can be swapped via the 'model' argument.
    """

    def __init__(
        self,
        model:      str = DEFAULT_MODEL,
        max_tokens: int = 200,
    ):
        self.model      = model
        self.max_tokens = max_tokens
        self.last_result: Optional[dict] = None

    # Truncation helper

    def _smart_truncate(self, text: str, max_chars: int = MAX_TRANSCRIPT_CHARS) -> str:
        """Take 30 % from start, 70 % from end — keeps context and peaks later in speech."""
        if len(text) <= max_chars:
            return text
        head_chars = int(max_chars * 0.30)
        tail_chars = max_chars - head_chars
        return text[:head_chars] + "\n...[truncated]...\n" + text[-tail_chars:]

    # Prompt builders

    @staticmethod
    def _build_prompt(question: str, transcript: str) -> str:
        return f"""You are scoring a spoken response for cognitive load indicators.
You will be given a question and a verbatim speech transcript.

Score the response on these 3 dimensions (1–5 each):

RELEVANCE: Did the speaker address what was actually asked?
  1 = completely off-topic
  3 = partial address, significant drift
  5 = directly and fully addresses the question

COMPLETENESS: How developed is the answer?
  1 = abandoned or minimal—few details
  3 = partial answer, missing key elements
  5 = thorough and well-developed answer

COHERENCE: Is the response logically structured with clear meaning?
  1 = hard to follow, unclear words, disconnected ideas, grammatical errors
  3 = somewhat disorganized, repetitive phrasing, some unclear references
  5 = clear structure, ideas flow logically, easy to understand

WHAT TO PENALIZE (these indicate cognitive effort):
  - Repetitive phrases ("then I could see" repeated many times)
  - Unclear or ambiguous words (listener would struggle to understand)
  - Grammatical errors that obscure meaning ("on my in my room")
  - List-like delivery instead of narrative flow
  - Vague references ("it" referring to unclear antecedent)

Rules:
- Score content organization and clarity, not utterance fluency
- DO NOT score based on filler words, hesitations, or pauses alone
- DO penalize unclear/ambiguous speech that makes comprehension difficult
- Be strict: lower scores when listener effort is needed to understand

OUTPUT FORMAT (CRITICAL - MUST BE VALID JSON ONLY):
- Return ONLY a single JSON object, no markdown, no text before or after
- Use exactly these 4 keys: "relevance", "completeness", "coherence", "confidence"
- All values must be NUMBERS (not strings)
- Dimensions: integers from 1 to 5
- Confidence: decimal from 0.0 to 1.0
- No null or None values—use 3 and 0.5 if unsure
- Example: {{"relevance": 4, "completeness": 3, "coherence": 4, "confidence": 0.8}}

Question: {question}
Transcript: {transcript}"""

    @staticmethod
    def _build_retry_prompt(question: str, transcript: str) -> str:
        """Stricter prompt used on the second attempt."""
        return f"""TASK: Score a response on 3 dimensions. Be strict.

Dimensions (1–5 scale):
1. relevance - does it answer the question directly?
2. completeness - is it thorough and detailed?
3. coherence - is it clear, organized, easy to follow?

PENALIZE cognitive load indicators:
- Repetitive phrases (searching for words)
- Unclear/ambiguous words (hard to understand)
- Grammatical errors or vague references
- List-like delivery (no flow or narrative)

JSON OUTPUT FORMAT (REQUIRED):
- Output ONLY a JSON object, nothing else
- Keys: "relevance", "completeness", "coherence", "confidence"
- All values are NUMBERS (not text)
- Dimensions: 1, 2, 3, 4, or 5 (integers only)
- Confidence: 0.0 to 1.0 (decimal)
- EXAMPLE: {{"relevance": 3, "completeness": 4, "coherence": 3, "confidence": 0.7}}

Question: {question[:200]}
Transcript: {transcript[:500]}"""

    # Main compute

    async def compute(
        self,
        transcript_text: str,
        client: AsyncOpenAI,
        question_text: str = "",
        speaker_id: Optional[str] = None,
        baselines: Optional[Dict] = None,
    ) -> dict:
        """
        Parameters
        ----------
        transcript_text : raw transcript string
        client          : openai.AsyncOpenAI pointed at Groq / OpenRouter
        question_text   : the question that prompted this response (required for
                          meaningful scoring; empty string is accepted gracefully)
        speaker_id      : optional speaker identifier for z-score computation
        baselines       : optional {speaker_id: {agent: baseline, ...}}

        Returns
        -------
        {
            relevance, completeness, coherence,   # raw 1–5 or null
            llm_confidence,                       # LLM-reported certainty
            composite_score,                      # normalised mean [0, 1]
            raw_score,                            # alias for composite_score
            score,                                # z-score (or raw if no baseline)
            truncated,
            model_used,
        }
        """
        if not isinstance(transcript_text, str) or not transcript_text.strip():
            raise ValueError("transcript_text must be a non-empty string")

        question = question_text or "(no question provided)"

        # Smart truncation
        original_len    = len(transcript_text)
        transcript_text = self._smart_truncate(transcript_text)
        truncated       = original_len > MAX_TRANSCRIPT_CHARS
        if truncated:
            print(
                f"  [coherence] WARNING: transcript truncated to "
                f"{MAX_TRANSCRIPT_CHARS} chars (was {original_len})."
            )

        # First attempt
        parsed = await self._call_llm(
            client,
            self._build_prompt(question, transcript_text),
            temperature=0.15,
        )
               
        # Retry on parse failure
        if parsed is None:
            print("  [coherence] Parse failed — retrying with stricter prompt …")
            parsed = await self._call_llm(
                client,
                self._build_retry_prompt(question, transcript_text),
                temperature=0.15,
            )

        # Return None if parsing failed both times
        if parsed is None:
            print("  [coherence] ERROR: LLM response parsing failed after retry; skipping coherence scoring")
            return None

        # Validate required fields and values
        required_fields = {"relevance", "completeness", "coherence"}
        missing_fields = required_fields - set(parsed.keys())
        if missing_fields:
            print(f"  [coherence] ERROR: Missing required fields: {missing_fields}")
            print(f"    Parsed keys: {list(parsed.keys())}")
            print(f"    Full parsed object: {parsed}")
            return None

        # Extract dimension scores (may be null / None)
        def _safe(key: str) -> Optional[float]:
            v = parsed.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (ValueError, TypeError) as e:
                print(f"  [coherence] WARNING: Could not convert {key}={v!r} to float: {e}")
                return 3.0 if key != "confidence" else 0.5

        rel_raw  = _safe("relevance")
        comp_raw = _safe("completeness")
        coh_raw  = _safe("coherence")
        
        # Extract confidence with error handling
        try:
            llm_conf = float(parsed.get("confidence", 0.5))
        except (ValueError, TypeError):
            print(f"  [coherence] WARNING: Could not parse confidence value, using default 0.5")
            llm_conf = 0.5
        
        llm_conf = max(0.0, min(1.0, llm_conf))

        # Normalise each dimension to [0, 1]
        rel_norm  = _normalise_dim(rel_raw)
        comp_norm = _normalise_dim(comp_raw)
        coh_norm  = _normalise_dim(coh_raw)

        composite = _composite(rel_norm, comp_norm, coh_norm)

        # Compute per-dimension z-scores
        rel_z = rel_norm if rel_norm is None else rel_norm
        comp_z = comp_norm if comp_norm is None else comp_norm
        coh_z = coh_norm if coh_norm is None else coh_norm
        
        if speaker_id and baselines:
            speaker_baseline = baselines.get(speaker_id, {})
            
            # Try to extract per-dimension baselines (new format)
            # Fall back to composite baseline if per-dimension not available (old format)
            composite_baseline = speaker_baseline.get("coherence", 0.5)
            
            rel_baseline = speaker_baseline.get("relevance", composite_baseline)
            comp_baseline = speaker_baseline.get("completeness", composite_baseline)
            coh_baseline = speaker_baseline.get("coherence", composite_baseline)
            
            # Compute z-scores for each dimension
            if rel_norm is not None:
                rel_z = compute_z_score(rel_norm, rel_baseline, "coherence")  # using coherence range for all dims
            if comp_norm is not None:
                comp_z = compute_z_score(comp_norm, comp_baseline, "coherence")
            if coh_norm is not None:
                coh_z = compute_z_score(coh_norm, coh_baseline, "coherence")

        result = {
            # Dimension scores (raw 1–5 from LLM, null if insufficient data)
            "relevance":        rel_raw,
            "completeness":     comp_raw,
            "coherence":        coh_raw,
            # Normalised equivalents (for inspection)
            "relevance_norm":   rel_norm,
            "completeness_norm": comp_norm,
            "coherence_norm":   coh_norm,
            # Per-dimension z-scores (for independent signal processing)
            "relevance_z":      rel_z,
            "completeness_z":   comp_z,
            "coherence_z":      coh_z,
            # Composite (for reference/aggregation)
            "composite_score":  composite,
            "llm_confidence":   round(llm_conf, 3),
            "truncated":        truncated,
            "model_used":       self.model,
        }
        self.last_result = result
        return result

    async def _call_llm(self, client: AsyncOpenAI, prompt: str, temperature: float = 0.0) -> Optional[dict]:
        """Call the LLM and attempt JSON extraction. Returns None on any failure."""
        raw = None
        try:
            response = await client.chat.completions.create(
                model       = self.model,
                max_tokens  = self.max_tokens,
                messages    = [{"role": "user", "content": prompt}],
                temperature = temperature,
            )
            
            # Validate response structure
            if not response.choices or not response.choices[0]:
                print(f"  [coherence] ERROR: Empty response from LLM")
                return None
            
            message = response.choices[0].message
            if not message or not message.content:
                print(f"  [coherence] ERROR: LLM returned empty content")
                return None
            
            raw = message.content.strip()
            if not raw:
                print(f"  [coherence] ERROR: LLM response is empty after stripping")
                return None
            
            parsed = _extract_json(raw)
            return parsed
            
        except json.JSONDecodeError as exc:
            print(f"  [coherence] JSON parse error: {exc}")
            if raw:
                print(f"    raw response: {raw[:500]}")
            return None
        except Exception as exc:
            print(f"  [coherence] LLM call error: {type(exc).__name__}: {exc}")
            import traceback
            print(f"    full traceback: {traceback.format_exc()}")
            if raw:
                print(f"    raw response: {raw[:500]}")
            return None

    # Persistence

    def save(self, path: str) -> None:
        data = {
            "model":       self.model,
            "max_tokens":  self.max_tokens,
            "last_result": self.last_result,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.model       = data["model"]
        self.max_tokens  = data["max_tokens"]
        self.last_result = data.get("last_result")


# Standalone runner

async def main():
    # Import transcribe_wav locally to avoid circular imports
    from utils import transcribe_wav
    
    DATA_DIR   = Path("Data/baseline")
    AUDIO_FILE = DATA_DIR / "ayu.mp4"
    OUTPUT_DIR = Path("output")
    OUTPUT_DIR.mkdir(exist_ok=True)
    SAVE_PATH  = OUTPUT_DIR / "coherence_metrics.json"

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is not set.\n"
            "Add it to your .env file or export it: set OPENROUTER_API_KEY=your-openrouter-key"
        )

    model  = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    client = AsyncOpenAI(
        api_key=api_key, 
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/speech-cognitive-load/project",
            "X-Title": "Speech Cognitive Load"
        }
    )
    agent  = CoherenceAgent(model=model)

    print("=== Coherence Analysis (OpenRouter) ===")
    print(f"    model : {model}\n")

    # Check if audio file exists
    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_FILE}")

    print(f"[Transcribe] Loading {AUDIO_FILE.name}")
    # Transcribe audio file
    transcript_data = transcribe_wav(AUDIO_FILE)
    transcript_text = transcript_data.get("text", "")
    question_text   = "Describe your room."  # Default question for baseline

    print(f"\n[Coherence Analysis]")
    result = await agent.compute(
        transcript_text,
        client,
        question_text=question_text,
        speaker_id="ayu",
    )
    
    # Handle case where result is None (LLM parsing failed)
    if result is None:
        print(f"\n  [BASELINE - AYU]")
        print(f"    ERROR: Failed to score coherence (LLM parsing failed)")
        print()
        return
    
    print(f"\n  [BASELINE - AYU]")
    print(f"    relevance        : {result['relevance']}  (raw 1-5)")
    print(f"    completeness     : {result['completeness']}  (raw 1-5)")
    print(f"    coherence        : {result['coherence']}  (raw 1-5)")
    print(f"    composite_score  : {result['composite_score']}  (normalised 0-1)")
    print(f"    llm_confidence   : {result['llm_confidence']}")
    print(f"    score            : {result['score']}")
    if result.get("truncated"):
        print(f"    [!] transcript was truncated before scoring")
    print()

    agent.save(SAVE_PATH)
    print(f"Saved config + last result → {SAVE_PATH}")

    agent2 = CoherenceAgent()
    agent2.load(SAVE_PATH)
    print(f"Re-loaded model      : {agent2.model}")
    print(f"Re-loaded last_result keys: {list(agent2.last_result.keys())}")


if __name__ == "__main__":
    asyncio.run(main())
