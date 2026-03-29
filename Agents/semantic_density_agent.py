"""
SemanticDensityAgent
--------------------
Input  : transcript text only  (plain string)
Type   : LLM-based
Library: OpenRouter (OpenAI-compatible API; any model can be used via config)

Score semantics
---------------
  0.0 → speaker speaking with ease, fluent, low effort
  1.0 → speaker struggling, hesitant, high cognitive effort

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
from typing import Dict, Optional
from pathlib import Path
from openai import AsyncOpenAI

# Import z-score helper
from z_score import compute_z_score

dotenv.load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL   = "DeepSeek-R1-Distill-Llama-70B"        
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

    Uses Groq's endpoint for fast inference on open-source models
    (Mixtral, Llama, etc.). Any model can be swapped via the
    'model' argument without touching the rest of the code.
    """

    def __init__(
        self,
        model:        str  = DEFAULT_MODEL,
        max_tokens:   int  = 200,
        invert_score: bool = False,   # False since effort_score already represents load directly
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
    def _build_prompt(transcript_text: str, total_words: int = None, question_text: str = None, baseline_transcript: str = None) -> str:
        # Calculate word count if not provided
        if total_words is None:
            total_words = len(transcript_text.split())
        
        # If baseline transcript is provided, use comparison format
        if baseline_transcript:
            return f"""You are analyzing a speech transcript from a Hindi-English speaker.
Compare their test response to their baseline to assess cognitive effort.

Here is a baseline transcript of this speaker's natural relaxed speech:
\"\"\"{baseline_transcript}\"\"\"

Here is the test transcript to evaluate:
\"\"\"{transcript_text}\"\"\"

Compared to their baseline, rate how much MORE effort the speaker is expending in the test transcript.
0.5 = same as baseline (no change in effort)
<0.5 = LESS effort (more relaxed than baseline)
>0.5 = MORE effort (more hesitant/struggling than baseline)

Rules:
- Do NOT penalize grammar errors or code-switching alone
- Judge PRODUCTION effort relative to their baseline
- Be objective: use specific evidence from both transcripts
- Be willing to give extreme scores (0.1 or 0.9) when differences are clear

Respond with ONLY valid JSON:
{{"effort_score": <float 0.0-1.0>, "reasoning": "<one sentence comparing to baseline>"}}"""
        
        # Original prompt if no baseline provided
        return f"""You are analyzing a speech transcript from a Hindi-English speaker.
        Rate the cognitive effort visible in their speech production on a 0.0-1.0 scale.

        ANCHOR EXAMPLES:
        - 0.1-0.2 (very low effort): Fluent, confident, smooth flow, no searching for words,
        responses feel automatic. Example: describing their own bedroom or daily routine.
        
        - 0.4-0.6 (moderate effort): Some hesitations, occasional word searching, mild 
        repetition but generally coherent. Manageable topic for the speaker.
        
        - 0.8-0.9 (high effort): Frequent false starts, long pauses mid-sentence, visible 
        word-searching, incomplete thoughts, topic seems to be pushing their limits.

        Rules:
        - Do NOT penalize grammar errors or code-switching alone
        - Judge PRODUCTION effort, not content quality
        - Compare to typical spontaneous Hindi-English speech, not native English
        - Be willing to give extreme scores (0.1 or 0.9) when evidence is clear
        - Do not default to 0.5 or 0.6 when uncertain — instead reflect uncertainty 
        in your reasoning

        Transcript ({total_words} words):
        \"\"\"{transcript_text}\"\"\"

        Respond with ONLY valid JSON:
        {{"effort_score": <float 0.0-1.0>, "reasoning": "<one concise sentence>"}}"""

    @staticmethod
    def _build_retry_prompt(transcript_text: str, total_words: int = None, question_text: str = None, baseline_transcript: str = None) -> str:
        """Stricter prompt used on the second attempt."""
        if total_words is None:
            total_words = len(transcript_text.split())
        
        if baseline_transcript:
            return f"""Judge cognitive effort relative to baseline:

Baseline (relaxed speech):
\"\"\"{baseline_transcript[:300]}\"\"\"

Test transcript:
\"\"\"{transcript_text[:300]}\"\"\"

0.5 = same effort as baseline
<0.5 = less effort (more relaxed)
>0.5 = more effort (more struggling)

Output ONLY valid JSON:
{{"effort_score": 0.5, "reasoning": "one sentence"}}
"""
        
        return f"""Judge cognitive effort from this speech:

0.0 = speaker relaxed, fluent, low effort
0.5 = speaker shows typical effortfulness
1.0 = speaker struggling, hesitant, high cognitive load

Transcript ({total_words} words):
"{transcript_text[:500]}"

Output ONLY valid JSON:
{{"effort_score": 0.5, "reasoning": "one sentence"}}
"""

    # Main compute 

    async def compute(self, transcript_text: str, client: AsyncOpenAI, question_text: str = None, speaker_id: Optional[str] = None, baselines: Optional[Dict] = None, baseline_transcript: Optional[str] = None) -> dict:
        """
        Parameters
        ----------
        transcript_text : raw transcript string
        client          : openai.AsyncOpenAI pointed at OpenRouter
        question_text   : (optional) the question that prompted this response
        speaker_id      : (optional) speaker identifier for z-score computation
        baselines       : (optional) {speaker_id: {agent: baseline, ...}, "_population_std": {agent: std, ...}}
        baseline_transcript : (optional) baseline transcript for comparison

        Returns
        -------
        {effort_score, reasoning, raw_score, score (z-score), truncated, model_used}
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
        # Don't pass question_text to avoid biasing the LLM toward expected difficulty level
        parsed = await self._call_llm(client, self._build_prompt(transcript_text, total_words, baseline_transcript=baseline_transcript), temperature=0.15)

        # Retry on parse failure 
        if parsed is None:
            print("  [semantic_density] Parse failed — retrying with stricter prompt …")
            parsed = await self._call_llm(client, self._build_retry_prompt(transcript_text, total_words, baseline_transcript=baseline_transcript), temperature=0.15)

        # Fallback 
        if parsed is None:
            print("  [semantic_density] WARNING: using fallback score 0.5")
            parsed = {"effort_score": 0.5, "reasoning": "Fallback — LLM did not return valid JSON."}

        # Accept both effort_score (new) and density_score (old) for compatibility
        effort = float(parsed.get("effort_score", parsed.get("density_score", 0.5)))
        effort = max(0.0, min(1.0, effort))   # clamp just in case

        raw_score = round(1.0 - effort, 3) if self.invert_score else round(effort, 3)

        # Compute z-score if baselines available
        z_score = raw_score
        if speaker_id and baselines:
            speaker_baseline = baselines.get(speaker_id, {})
            baseline = speaker_baseline.get("semantic_density", raw_score)
            z_score = compute_z_score(raw_score, baseline, "semantic_density")

        result = {
            "effort_score": round(effort, 3),
            "reasoning":     parsed.get("reasoning", ""),
            "raw_score":     raw_score,
            "score":         z_score,          # z-score
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

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Add it to your .env file or export it: set GROQ_API_KEY=your-groq-key"
        )

    model = os.environ.get("GROQ_MODEL", DEFAULT_MODEL)

    client = AsyncOpenAI(api_key=api_key, base_url=GROQ_BASE_URL)
    agent  = SemanticDensityAgent(model=model, invert_score=True)

    print("=== Semantic Density Analysis (Groq) ===")
    print(f"    model : {model}\n")

    for label in LABELS:
        data            = json.load(open(DATA_DIR / f"{label}_transcript.json"))
        transcript_text = data["text"]
        # Extract question_text from data dict if available
        question_text   = data.get("question_text") or data.get("question") or None

        result = await agent.compute(transcript_text, client, question_text=question_text)
        print(f"  [{label.upper()}]")
        print(f"    density_score : {result['density_score']}  (1 = information-rich)")
        print(f"    score         : {result['score']}  (0 = articulate, 1 = vague / high load)")
        print(f"    reasoning     : {result['reasoning']}")
        if question_text:
            print(f"    question      : {question_text}")
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