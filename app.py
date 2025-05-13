from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import random

app = Flask(__name__)

MODEL_NAME = "serkanacar/mental-disorder-augmented-model"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

label_intros = {
    "depression": "Depression may involve persistent feelings of sadness, hopelessness, and a loss of interest in activities. Here are some suggestions that might help you:",
    "anxiety": "Anxiety often includes constant worry, restlessness, and physical tension. Here are some techniques that might support you:",
    "suicide": "Suicidal thoughts are serious and may reflect deep emotional pain. You're not alone. Please consider the suggestions below and seek professional help:",
    "stress": "Stress can feel overwhelming and can manifest physically and emotionally. Here are a few tips to help you relax:",
    "bipolar": "Bipolar disorder involves shifts between emotional highs and lows. Here are some ways to help regulate your mood:",
    "personality": "Personality-related difficulties may affect how you think, feel, and relate to others. Consider these suggestions that may offer clarity and support:"
}

suggestions = {
    "normal": [
        "Keep up your healthy habits like sleep, nutrition, and movement.",
        "Stay mindful of the things that give your life meaning.",
        "Keeping a journal can help you stay connected to your emotions."
    ],
    "depression": [
        "Try setting small daily goals and acknowledge yourself when you complete them. If this doesn’t help, please consider seeing a professional.",
        "Explore what your feelings might be trying to show you. If needed, don’t hesitate to reach out to a therapist.",
        "Make a checklist of simple tasks like taking a shower or preparing breakfast. If this feels too hard, seek help from a mental health professional.",
        "Try the 54321 technique: Name 5 things you see, 4 surfaces you can touch, 3 sounds you hear, 2 scents you smell, and 1 taste. If this doesn't help, speak with a professional."
    ],
    "anxiety": [
        "Try a breathing exercise: inhale for 4 seconds, hold for 4, exhale for 4. Repeat for a minute.",
        "Progressive muscle relaxation can help—tighten and release different muscle groups.",
        "Ask yourself: 'What does this anxiety show I care about?' Reflect, then go for a walk. If it’s too much, talk to a professional."
    ],
    "suicidal": [
        "Please talk to someone close to you about how you're feeling. You're not alone.",
        "If you're thinking about suicide, please reach out to emergency services or call 112.",
        "These thoughts can feel overwhelming. Seeking help is a brave and strong choice.",
        "If you're serious about these thoughts, talk to someone you trust and get professional help as soon as possible."
    ],
    "stress": [
        "Notice where in your body you feel tension (like your jaw or shoulders) and relax those areas.",
        "Take a 5-minute break in nature or disconnect from screens for a moment.",
        "Try 2 minutes of silence to remind yourself that you’re in control.",
        "Use the 4-7-8 breathing technique. Focus on your breath for 5 minutes.",
        "Try a full body tension and release exercise from your feet up.",
        "Go for a walk or do a stretching/yoga session at home to unwind.",
        "Replace inner criticism with affirmations: 'I can do this,' or 'Just step outside today!'"
    ],
    "bipolar": [
        "If you're in a less manic state, reduce stimulants like caffeine or alcohol. In a manic state, ask yourself impulse-control questions and talk to a professional.",
        "Try 4-7-8 or 4-4-6 breathing. Diaphragmatic breathing can calm your nervous system. If it’s not enough, see a professional.",
        "Track your mood daily (e.g., rate 0–10). Journaling can help observe patterns. A therapist can help you explore this safely."
    ],
    "personality disorder": [
        "Track moments that trigger you and note what you feel. Reflect on whether your automatic thoughts are helpful. Therapy is highly recommended.",
        "Practice recognizing your boundaries and others’. Journaling when emotions feel intense may help.",
        "Psychotherapy is the most helpful path for managing these complex patterns. Please consider seeking support."
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
                    "text": "Hi! I'm mentAI. If you're dealing with a mental health concern or feeling emotionally overwhelmed, I'm here to help. Just start sharing, and I'll do my best to support you!"
                }
            ]
        })

    inputs = tokenizer.encode(user_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=5)
    predicted_label = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()

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
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
