from flask import Flask, request, jsonify, render_template
from chatbot import get_bot_response   # importing the chatbot function

app = Flask(__name__)

# Serve the frontend 
@app.route("/")
def home():
    return render_template("index.html")

# Handling user input
@app.route("/get", methods=["POST"])
def get_bot_reply():
    data = request.get_json()
    user_input = data.get("message", "")
    print("🟢 Received from frontend:", user_input)   # Debug log

    try:
        response = get_bot_response(user_input)   # calling chatbot.py function
        print("🔵 Chatbot response:", response)   # Debug log
        return jsonify({"response": response})
    except Exception as e:
        import traceback
        print("❌ ERROR in get_bot_response:", e)
        traceback.print_exc()
        return jsonify({"response": "⚠️ Server error, check console."}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
