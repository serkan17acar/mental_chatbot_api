from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import torch
import random
import os

app = Flask(__name__)

hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN not found in environment variables.")

MODEL_NAME = "serkanacar/mental-disorder-augmented-model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=hf_token)
model.eval()

label_map = {
    0: "anxiety",
    1: "bipolar",
    2: "depression",
    3: "normal",
    4: "personality disorder",
    5: "stress",
    6: "suicidal"
}

label_intros = {
    "depression": "Depression may involve persistent feelings of sadness, hopelessness, and a loss of interest in activities. Here are some suggestions that might help you:",
    "anxiety": "Anxiety often includes constant worry, restlessness, and physical tension. Here are some techniques that might support you:",
    "suicidal": "Suicidal thoughts are serious and may reflect deep emotional pain. You're not alone. Please consider the suggestions below and seek professional help:",
    "stress": "Stress can feel overwhelming and can manifest physically and emotionally. Here are a few tips to help you relax:",
    "bipolar": "Bipolar disorder involves shifts between emotional highs and lows. Here are some ways to help regulate your mood:",
    "personality disorder": "Personality-related difficulties may affect how you think, feel, and relate to others. Consider these suggestions that may offer clarity and support:"
}

suggestions = {
    "normal": [
        "Keep up your healthy habits like sleep, nutrition, and movement.",
        "Stay mindful of the things that give your life meaning.",
        "Keeping a journal can help you stay connected to your emotions."
    ],
    "depression": [
        "Try setting small daily goals and acknowledge yourself when you complete them.",
        "Explore what your feelings might be trying to show you.",
        "Make a checklist of simple tasks like taking a shower or preparing breakfast.",
        "Try the 54321 grounding technique to reconnect with the present."
    ],
    "anxiety": [
        "Try a breathing exercise: inhale for 4 seconds, hold for 4, exhale for 4.",
        "Progressive muscle relaxation can help—tighten and release different muscle groups.",
        "Go for a walk and try to name things you see, hear, and feel."
    ],
    "suicidal": [
        "Please talk to someone close to you about how you're feeling.",
        "If you're thinking about suicide, please reach out to emergency services or call 112.",
        "These thoughts can feel overwhelming. Seeking help is a strong choice.",
        "You're not alone. Talking to a professional can really help."
    ],
    "stress": [
        "Notice where in your body you feel tension and relax those areas.",
        "Take a 5-minute break in nature or disconnect from screens.",
        "Use the 4-7-8 breathing technique. Focus on your breath for 5 minutes.",
        "Do light exercise or stretching to unwind."
    ],
    "bipolar": [
        "Track your mood daily. It can help you spot patterns.",
        "If you're feeling manic, reduce stimulants and try grounding exercises.",
        "Talk with someone you trust and consider professional guidance."
    ],
    "personality disorder": [
        "Keep a journal to reflect on emotional reactions.",
        "Recognize patterns in relationships and try to pause before reacting.",
        "Therapy can help navigate identity and emotional regulation challenges."
    ]
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "").strip()

    if not user_input:
        return jsonify({
            "messages": [
                {
                    "sender": "bot",
                    "text": "Hi! I'm mentAI. If you're dealing with a mental health concern or feeling emotionally overwhelmed, I'm here to help. Just start sharing, and I'll do my best to support you."
                }
            ]
        })

    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
        predicted_label = label_map.get(predicted_class_id, "normal")

    messages = []

    if predicted_label != "normal":
        intro = label_intros.get(predicted_label, "")
        if intro:
            messages.append({"sender": "bot", "text": intro})

    suggestion_list = suggestions.get(predicted_label, suggestions["normal"])
    random_suggestion = random.choice(suggestion_list)
    messages.append({"sender": "bot", "text": random_suggestion})

    return jsonify({
        "label": predicted_label,
        "messages": messages
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
