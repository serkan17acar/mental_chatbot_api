from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "")
    return jsonify({"response": f"You said: {user_input}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)