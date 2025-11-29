"""
Day 9 – E-Commerce Voice Agent (ACP-Inspired)
- Reads product catalog from products.json in backend directory.
- Creates/updates orders.json to persist placed orders.
- Tools:
    - list_products / show_product
    - add_to_cart / remove_from_cart / show_cart
    - place_order / last_order / order_history
"""

import os
import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional, Annotated
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import Field
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# -------------------------
# Setup
# -------------------------
load_dotenv(".env.local")

logger = logging.getLogger("ecommerce_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)

# -------------------------
# Data models
# -------------------------
@dataclass
class CartItem:
    id: str
    name: str
    price: float
    quantity: int = 1
    size: Optional[str] = None

@dataclass
class Userdata:
    cart: List[CartItem] = field(default_factory=list)
    last_order: Optional[dict] = None
    user_name: Optional[str] = None

# -------------------------
# File paths
# -------------------------
# -------------------------
# File paths (Fixed)
# -------------------------
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CATALOG_PATH = os.path.join(BACKEND_DIR, "products.json")
ORDERS_PATH = os.path.join(BACKEND_DIR, "orders.json")

if not os.path.exists(CATALOG_PATH):
    logger.warning(f"⚠️ products.json not found at {CATALOG_PATH}. Please ensure it exists.")
else:
    logger.info(f"✅ products.json found at {CATALOG_PATH}")


logger.info(f"BACKEND_DIR: {BACKEND_DIR}")
logger.info(f"CATALOG_PATH: {CATALOG_PATH}")
logger.info(f"Catalog exists: {os.path.exists(CATALOG_PATH)}")

# -------------------------
# Catalog + Orders Helpers
# -------------------------
def load_catalog() -> list:
    try:
        with open(CATALOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load catalog from {CATALOG_PATH}: {e}")
        return []

def load_orders() -> list:
    if not os.path.exists(ORDERS_PATH):
        return []
    try:
        with open(ORDERS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_orders(orders: list):
    try:
        with open(ORDERS_PATH, "w", encoding="utf-8") as f:
            json.dump(orders, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save orders: {e}")

# -------------------------
# Tools
# -------------------------
@function_tool
async def list_products(
    ctx: RunContext[Userdata],
    query: Optional[str] = None,
    max_price: Optional[float] = None,
) -> str:

    catalog = load_catalog()
    filtered = []
    for p in catalog:
        name = p.get("name", "").lower()
        if query and query.lower() not in name:
            continue
        if max_price and float(p.get("price", 0)) > max_price:
            continue
        filtered.append(p)

    if not filtered:
        return f"No products found for '{query}' under ₹{max_price or '∞'}."

    lines = []
    for p in filtered[:10]:
        lines.append(f"- {p['name']} (id: {p['id']}) — ₹{p['price']} | Category: {p.get('category', '')}")
    return "Here are some options:\n" + "\n".join(lines)

@function_tool
async def show_product(
    ctx: RunContext[Userdata],
    product_id: Annotated[str, Field(description="Product ID")],
) -> str:
    catalog = load_catalog()
    for p in catalog:
        if p["id"].lower() == product_id.lower():
            return f"{p['name']} — ₹{p['price']} | Category: {p.get('category','')} | {p.get('description','No description')}"
    return f"Couldn't find product with id '{product_id}'."

@function_tool
async def add_to_cart(
    ctx: RunContext[Userdata],
    product_id: Annotated[str, Field(description="Product ID to add")],
    quantity: Annotated[int, Field(description="Quantity to add", ge=1)] = 1,
) -> str:
    catalog = load_catalog()
    item = next((p for p in catalog if p["id"].lower() == product_id.lower()), None)
    if not item:
        return f"Product '{product_id}' not found."
    for ci in ctx.userdata.cart:
        if ci.id.lower() == product_id.lower():
            ci.quantity += quantity
            total = sum(c.price * c.quantity for c in ctx.userdata.cart)
            return f"Updated '{ci.name}' quantity to {ci.quantity}. Cart total: ₹{total:.2f}"
    ctx.userdata.cart.append(CartItem(id=item["id"], name=item["name"], price=float(item["price"]), quantity=quantity))
    total = sum(c.price * c.quantity for c in ctx.userdata.cart)
    return f"Added {quantity} x {item['name']} to your cart. Cart total: ₹{total:.2f}"

@function_tool
async def remove_from_cart(
    ctx: RunContext[Userdata],
    product_id: Annotated[str, Field(description="Product ID to remove")],
) -> str:
    before = len(ctx.userdata.cart)
    ctx.userdata.cart = [ci for ci in ctx.userdata.cart if ci.id.lower() != product_id.lower()]
    after = len(ctx.userdata.cart)
    if before == after:
        return f"Item '{product_id}' not found in cart."
    total = sum(c.price * c.quantity for c in ctx.userdata.cart)
    return f"Removed item '{product_id}'. Cart total: ₹{total:.2f}"

@function_tool
async def show_cart(ctx: RunContext[Userdata]) -> str:
    if not ctx.userdata.cart:
        return "Your cart is empty."
    lines = [f"- {ci.quantity} x {ci.name} @ ₹{ci.price:.2f} = ₹{ci.price * ci.quantity:.2f}" for ci in ctx.userdata.cart]
    total = sum(c.price * c.quantity for c in ctx.userdata.cart)
    return "Your cart:\n" + "\n".join(lines) + f"\nTotal: ₹{total:.2f}"

@function_tool
async def place_order(
    ctx: RunContext[Userdata],
    customer_name: Annotated[str, Field(description="Customer name")],
) -> str:
    if not ctx.userdata.cart:
        return "Your cart is empty."
    order_id = str(uuid.uuid4())[:8]
    timestamp = datetime.utcnow().isoformat() + "Z"
    total = sum(c.price * c.quantity for c in ctx.userdata.cart)
    items = [{"id": c.id, "name": c.name, "price": c.price, "quantity": c.quantity} for c in ctx.userdata.cart]

    orders = load_orders()
    order = {
        "order_id": order_id,
        "customer": customer_name,
        "timestamp": timestamp,
        "total": total,
        "currency": "INR",
        "items": items,
        "status": "confirmed",
    }
    orders.append(order)
    save_orders(orders)

    ctx.userdata.last_order = order
    ctx.userdata.cart.clear()

    return f"Order placed successfully! Order ID: {order_id}. Total ₹{total:.2f}. It’s being processed under express checkout."

@function_tool
async def last_order(ctx: RunContext[Userdata]) -> str:
    if not ctx.userdata.last_order:
        return "You haven't placed any orders yet."
    o = ctx.userdata.last_order
    items = ", ".join([i["name"] for i in o["items"]])
    return f"Your last order ({o['order_id']}) includes {items}. Total ₹{o['total']:.2f}. Status: {o['status']}."

@function_tool
async def order_history(ctx: RunContext[Userdata]) -> str:
    orders = load_orders()
    if not orders:
        return "No past orders found."
    lines = []
    for o in orders[-5:]:
        lines.append(f"- {o['order_id']} | ₹{o['total']:.2f} | {o['status']} | {o['timestamp']}")
    return "Recent orders:\n" + "\n".join(lines)

# -------------------------
# Agent Definition
# -------------------------
class EcommerceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
            You are 'Nova', a friendly AI shopping assistant for an online store.
            Tone: Helpful, modern, concise, and professional.
            You help users browse products, compare items, and place orders.

            Guidelines:
            - Use the catalog functions to list, describe, and add products.
            - Be polite, avoid repeating too much.
            - Mention prices in Indian Rupees (₹).
            - Confirm details when placing an order.
            - Orders are simulated only (no payments).
            """,
            tools=[
                list_products,
                show_product,
                add_to_cart,
                remove_from_cart,
                show_cart,
                place_order,
                last_order,
                order_history,
            ],
        )

# -------------------------
# Entrypoint
# -------------------------
def prewarm(proc: JobProcess):
    try:
        proc.userdata["vad"] = silero.VAD.load()
    except Exception:
        logger.warning("VAD prewarm failed; continuing without preloaded VAD.")

async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("\n🛒 Starting E-Commerce Voice Agent (ACP Inspired)")
    
    # Verify catalog loads
    catalog = load_catalog()
    logger.info(f"Catalog loaded with {len(catalog)} products")
    if not catalog:
        logger.warning("⚠️ WARNING: Catalog is empty or failed to load!")

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-natalie",
            style="Conversational",
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata.get("vad"),
        userdata=Userdata(),
    )

    await session.start(
        agent=EcommerceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVC()),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
