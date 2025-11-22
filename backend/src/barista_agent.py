import logging
import json
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
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("barista_agent")

load_dotenv(".env.local")


class BaristaAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly coffee shop barista. 
            You take orders from users for coffee and ask questions to fill in the order.
            The order includes drinkType, size, milk, extras, and name. 
            Be polite, cheerful, and concise. Once all details are collected, save the order to a JSON file.""",
        )
        # Initialize empty order state
        self.order_state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": ""
        }

    def save_order(self):
        """Saves the current order to a JSON file in backend folder"""
        with open("order_summary.json", "w") as f:
            json.dump(self.order_state, f, indent=4)
        print("Order saved to order_summary.json")  # optional debug

    async def handle_message(self, message: str):
        """Fill in the order state by asking questions if any field is missing."""
        if not self.order_state["drinkType"]:
            self.order_state["drinkType"] = message
            return "What size would you like? (Small, Medium, Large)"
        if not self.order_state["size"]:
            self.order_state["size"] = message
            return "Which milk would you like? (Regular, Almond, Soy, Oat)"
        if not self.order_state["milk"]:
            self.order_state["milk"] = message
            return "Any extras? (e.g., sugar, syrup, whipped cream). Say 'none' if no extras."
        if not self.order_state["extras"]:
            extras_list = [e.strip() for e in message.split(",")] if message.lower() != "none" else []
            self.order_state["extras"] = extras_list
            return "Lastly, may I have your name for the order?"
        if not self.order_state["name"]:
            self.order_state["name"] = message
            # Save order to JSON
            self.save_order()
            return f"Thank you {self.order_state['name']}! Your order has been placed."

        return "Your order is complete!"


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
            text_pacing=True
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
        agent=BaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC())
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
