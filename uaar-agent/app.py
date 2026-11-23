"""
UAAR Info Agent — Flask Web Application
Run:  python app.py
Open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
from agent import get_agent

app   = Flask(__name__)
agent = get_agent()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data         = request.get_json()
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"answer": "Please type a question.",
                        "intent": "empty", "nb_conf": 0.0, "cos_sim": 0.0})
    result      = agent.get_response(user_message)
    suggestions = agent.get_suggestions(user_message, top_n=3)
    return jsonify({
        "answer"     : result["answer"],
        "intent"     : result["intent"],
        "nb_conf"    : result["nb_conf"],
        "cos_sim"    : result["cos_sim"],
        "suggestions": suggestions,
    })


@app.route("/intents")
def intents():
    return jsonify({"intents": agent.get_intent_classes()})


if __name__ == "__main__":
    print("\n UAAR Info Agent Web App")
    print(" Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)