import logging
import os
import json
from typing import Dict, Literal

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

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# ========== DAY 4 CONFIG ==========
CONTENT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "shared-data", "day4_tutor_content.json")
)

TUTOR_VOICES = {
    "learn": "en-US-matthew",   # Murf Falcon - Matthew
    "quiz": "en-US-alicia",     # Murf Falcon - Alicia
    "teach_back": "en-US-ken",  # Murf Falcon - Ken
}


def load_content() -> Dict[str, dict]:
    try:
        with open(CONTENT_PATH, "r", encoding="utf-8") as f:
            items = json.load(f)
        return {item["id"]: item for item in items}
    except Exception as e:
        logger.error(f"Failed to load content: {e}")
        return {}


@function_tool
async def set_tutor_mode(
    ctx: RunContext,
    mode: Literal["learn", "quiz", "teach_back"],
    concept_id: str,
) -> str:
    """
    Switch the tutor mode and concept.
    Modes: learn | quiz | teach_back
    """
    ctx.session.userdata["tutor_mode"] = mode
    ctx.session.userdata["concept_id"] = concept_id
    return f"mode_set:{mode}:{concept_id}"


class Assistant(Agent):
    def __init__(self, content: Dict[str, dict]):
        instructions = f"""
You are an Active Recall Coach (“Teach-the-Tutor”). 
Greet the learner, ask which concept they want (e.g. variables, loops) and which mode: learn, quiz, or teach_back.

Rules:
- Use the JSON content provided by the backend when explaining, quizzing, or prompting teach-back.
- Learn: explain the concept using its 'summary'. Keep it short and clear.
- Quiz: ask 1–3 short questions; start with the concept's 'sample_question'. One at a time. Give brief feedback.
- Teach_back: ask the learner to explain the concept. Give qualitative feedback comparing to 'summary'.
- The learner can switch modes any time (“switch to quiz on loops”). When they do, call the tool set_tutor_mode(mode, concept_id).
- Keep responses concise and spoken-friendly.
"""
        super().__init__(instructions=instructions, tools=[set_tutor_mode])
        self.content = content


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew",
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

    # ========== DAY 4 TUTOR LOGIC ==========
    content = load_content()
    session.userdata["tutor_mode"] = "learn"
    session.userdata["concept_id"] = "variables" if "variables" in content else next(iter(content.keys()), "")

    async def apply_voice_for_mode():
        mode = session.userdata.get("tutor_mode", "learn")
        voice = TUTOR_VOICES.get(mode, "en-US-matthew")
        await session.set_tts(
            murf.TTS(
                voice=voice,
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True,
            )
        )

    await apply_voice_for_mode()

    await session.start(
        agent=Assistant(content=content),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )

    @session.on("tool_result")
    async def _on_tool_result(ev):
        try:
            if isinstance(ev.result, str) and ev.result.startswith("mode_set:"):
                await apply_voice_for_mode()
        except Exception:
            pass

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
