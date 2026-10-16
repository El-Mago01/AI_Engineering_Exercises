#You do not need to edit this file.

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
SUPPORT_DIR = BASE_DIR / ".guides" / "scripts"
if str(SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(SUPPORT_DIR))


from pet_chat import PetChat
from pet_store import PetStore

app = Flask(__name__)
store = PetStore(BASE_DIR / "data" / "digipet.sqlite3")
chat = PetChat(BASE_DIR)

def stats_payload(pet):
    """Convert stored pet stats into JSON-ready data."""
    return pet.to_dict()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/stats")
def api_stats():
    return jsonify(stats_payload(store.get_stats()))


@app.post("/api/feed")
def api_feed():
    pet = store.feed()
    return jsonify({
        "message": "Food added. DigiPet looks more settled.",
        "stats": stats_payload(pet),
    })


@app.post("/api/give-water")
def api_give_water():
    pet = store.give_water()
    return jsonify({
        "message": "Water added. Hydration improved.",
        "stats": stats_payload(pet),
    })


@app.post("/api/play-with")
def api_play_with():
    pet = store.play()
    return jsonify({
        "message": "Play complete. DigiPet looks more engaged.",
        "stats": stats_payload(pet),
    })


@app.post("/api/talk")
def api_talk():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return jsonify({"error": "message is required", "reply": ""}), 400

    stats = stats_payload(store.get_stats())
    try:
        reply = chat.reply(message, stats)
    except Exception:
        app.logger.exception("DigiPet reply failed")
        return jsonify({
            "error": "reply failed",
            "reply": "DigiPet had trouble answering. Try again.",
            "message": message,
            "stats": stats,
        }), 500

    return jsonify({
        "reply": reply,
        "message": message,
        "stats": stats,
    })


@app.post("/api/reset-chat")
def api_reset_chat():
    chat.reset_history()
    return jsonify({"ok": True})


def main():
    """Run the Flask app for Codio preview."""
    hostname = os.environ.get("CODIO_HOSTNAME", "localhost")
    if hostname == "localhost":
        print("Preview: http://localhost:5000")
    else:
        print(f"Preview: https://{hostname}-5000.codio.io")

    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
