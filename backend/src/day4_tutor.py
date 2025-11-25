from __future__ import annotations
import logging
import os
import json
import re
from typing import Dict, List, Optional
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("day4_tutor")
load_dotenv(".env.local")

# Path to the small course content
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONTENT_PATH = os.path.join(BASE_DIR, "..", "shared-data", "day4_tutor_content.json")

def load_content() -> List[Dict]:
    try:
        with open(CONTENT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading content: {e}")
        return []

CONTENT = load_content()

# Voice mapping per mode (Murf voices)
VOICE_MAP = {
    "learn": "en-US-matthew",
    "quiz": "en-US-alicia",
    "teach_back": "en-US-ken",
}

@function_tool
async def get_concept(ctx: RunContext, concept_id: Optional[str] = None) -> Dict:
    """Return a concept object from the content file by id or default to first."""
    if not CONTENT:
        return {}
    if not concept_id:
        return CONTENT[0]
    q = concept_id.strip().lower()
    for c in CONTENT:
        if c.get("id") == q or c.get("title", "").lower() == q:
            return c
    return CONTENT[0]

@function_tool
async def switch_mode(ctx: RunContext, mode: str) -> str:
    """Agent can call this to request a session change voice/mode."""
    m = mode.strip().lower()
    if m not in VOICE_MAP:
        return f"unknown_mode:{m}"
    logger.info(f"Mode switch requested: {m}")
    return f"switched:{m}"

class TeachTheTutorAgent(Agent):
    def __init__(self, initial_mode: str = "learn") -> None:
        summary_texts = [f"- {c.get('id')}: {c.get('title')}" for c in CONTENT]
        content_list = "\n".join(summary_texts)

        instructions = (
            "You are an Active Recall Coach called 'Teach-the-Tutor'.\n"
            "Your job: greet the user, ask for their preferred mode (learn, quiz, teach_back), "
            "and then run short, focused interactions using the course content provided.\n"
            "Be concise, supportive, and avoid any medical or diagnostic advice.\n"
            "Available concepts:\n"
            f"{content_list}\n"
            "Behavior rules:\n"
            "- learn: explain the concept in simple language (use the concept 'summary').\n"
            "- quiz: ask the sample_question and accept a short answer, then give a short reflection.\n"
            "- teach_back: ask the user to explain the concept back and give brief qualitative feedback.\n"
            "You may call get_concept(concept_id) to fetch a concept and switch_mode(mode) to change voice.\n"
            "Always close a short interaction with a recap and ask 'Does this sound right?'."
        )

        super().__init__(instructions=instructions, tools=[get_concept, switch_mode])

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    default_voice = VOICE_MAP.get("learn")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice=default_voice,
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=TeachTheTutorAgent(initial_mode="learn"),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
