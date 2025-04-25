from flask import Flask, request, jsonify

app = Flask(__name__)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "")

    inputs = tokenizer.encode(user_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=50)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)