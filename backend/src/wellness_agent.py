# src/wellness_agent.py
import json, datetime, os

LOG_FILE = "wellness_log.json"

def load_logs():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            return json.load(f)
    return []

def save_entry(entry):
    logs = load_logs()
    logs.append(entry)
    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)

def generate_prompt():
    logs = load_logs()
    if logs:
        last = logs[-1]
        ref = f"Last time you said you felt {last['mood']} and planned {', '.join(last['goals'])}. "
    else:
        ref = ""
    return (
        ref
        + "Let's do today's quick check-in. How are you feeling and what's your energy like today?"
    )

def handle_checkin(mood, goals):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mood": mood,
        "goals": goals,
        "summary": f"Mood: {mood}, goals: {', '.join(goals)}",
    }
    save_entry(entry)
    return f"Got it — you're feeling {mood}. Your goals are {', '.join(goals)}. Let's make today manageable!"
