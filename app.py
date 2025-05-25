from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
from huggingface_hub import login
import torch
import random
import os
from datetime import datetime, timedelta 

app = Flask(__name__)

hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN not found in environment variables.")

MODEL_NAME = "serkanacar/mental-disorder-augmented-model"
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
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
    "depression": "What you've written seems to indicate signs of depression. Depression is a mental state where emotions like hopelessness, loss of interest, and low energy are experienced intensely and over a long period. Here are a few suggestions that might help.",
    "anxiety": "What you’ve shared suggests you may be experiencing anxiety. Anxiety shows itself through constant worry, overthinking, restlessness, and physical tension. I have a few suggestions that I think might help.",
    "suicidal": "Your sentences suggest a sense of hopelessness, a lack of will to live, and thoughts of ending life. This is an emotionally heavy and urgent situation. I want to offer you some advice at this point.",
    "stress": "From what you’ve written, I can tell you’re under stress. Stress is a mental and physical pressure that, if persistent, can cause sleep issues, irritability, and burnout. I have a few suggestions that I think may help and be beneficial.",
    "bipolar": "I think you might be showing symptoms of bipolar disorder. Bipolar disorder involves extreme mood shifts between highs (mania) and lows (depression). I have a few suggestions that may help and could be beneficial to you.",
    "personality disorder": "I think you’re showing signs of a personality disorder. Such difficulties often manifest as persistent patterns in how one perceives themselves, others, and relationships. These patterns can make it difficult to manage emotions, communicate, or adapt socially. I have some suggestions that might help and could be beneficial."
}

suggestions = {
    "normal": [
        "As you know, I’m a chatbot that detects mental disorders. I can identify six different mental disorders. Based on my evaluation, I did not detect any signs of a mental disorder in you. Your emotional state appears balanced and healthy. This suggests that you are able to cope with stress, manage your thoughts, and adapt to daily life. It’s very valuable to maintain this balance.",
        "As you know, I’m a chatbot focused on detecting mental disorders. I can identify six different mental disorders. However, based on your current expressions, I didn’t observe any signs of a disorder. Your mood seems stable, and your emotional and mental balance appears to be intact. This indicates harmony with yourself. Keep taking care of yourself and continue the habits that support this inner balance."
    ],
    "depression": [
    [
        "You can try to set small daily tasks and acknowledge yourself for completing them. You can create a small goal list. On this list, write manageable tasks like “I’ll take a shower,” or “I’ll prepare breakfast for myself,” and try to complete them. In addition, try to engage in physical activities like walking or exercising. I want you to avoid being alone during this process. Avoiding social isolation and staying in contact with a friend will help you feel better.",
        "Depression is a common condition that many people experience at some point in life. It's natural to feel alone or inadequate during this process. These suggestions may help, but if they’re not enough, I recommend seeking help from a professional. Remember, you are not alone."
    ],
    [
        "I can suggest the 54321 technique. It’s a five-step exercise. First, identify five things you can see. Second, feel four different surfaces around you. Third, focus on three different sounds you can hear. Fourth, notice two distinct smells. Finally, try to detect one taste. I believe this technique can be beneficial. Additionally, try journaling to confront your emotions. Try to understand yourself, show yourself compassion, and try to re-engage with your interests—but don’t push yourself too hard.",
        "Depression is a common condition that many people experience at some point in life. It’s normal to feel tired or lonely. If the suggestions don’t help, I recommend seeking support from a professional. Remember, you're not alone in this process."
    ]
    ],
    "anxiety": [
        "Let’s try a breathing exercise together. Focus on your breath: inhale for 4 seconds, hold for 4 seconds, and exhale for 4 seconds. Continue this for 1–2 minutes. In addition to breathing, physical exercises like tensing and relaxing muscles in sequence may help. If these exercises aren’t effective, ask yourself: “What is this anxiety trying to tell me about what I value or fear losing?” Answering this honestly can help you identify the source. Afterwards, go for a walk to clear your mind.",
        "Anxiety can sometimes appear as intense attacks or in milder phases. Remember that caffeine and sugar may trigger anxiety, so try to reduce their intake. Also, mindfulness practices like meditation can help you relax. Overall, positive lifestyle changes can be effective in reducing your anxiety levels."
    ],
    "suicidal": [
        "Please try not to be alone right now. Reach out to someone you feel safe with and share your feelings. Coping with such thoughts is not a burden you should carry alone. Seeking help from a professional is not a weakness; it is a strong step toward life. You may be struggling a lot right now, but remember: this is temporary, and you are valuable in this world.",
        "If these thoughts are exhausting you, please seek help from a loved one or a health professional. These feelings are not something anyone should have to carry alone. Life may not be easy, but it is still valuable. Asking for help is not an admission of failure; it is a courageous step toward finding a solution. Please give yourself that chance. Know that many people have experienced similar feelings and have healed. Remember, you're not alone. Seeking professional help is not weakness; it's a powerful step toward life."
    ],
    "stress": [
        "Focus on the areas where you feel tension most (like the jaw, shoulders, abdomen) and try progressive muscle relaxation by tensing and releasing those areas. This helps physically relax your body. Then try the 4-7-8 breathing technique: inhale for 4 seconds, hold for 7, and exhale for 8. Continue this for 5 minutes with full attention to your breath. Doing this with calming music can be even more effective.",
        "Taking a short walk or changing your environment can significantly reduce stress. If you can’t go outside, step away from screens for 10 minutes and sit quietly by a window, just listening to your breath. You can also write down your thoughts to relieve mental load. While doing so, give yourself positive affirmations: “I’m struggling, but it’s temporary. I’m just resting now.",
        "Take short silence breaks throughout the day to rest your mind. Close your eyes, focus only on breathing, and observe your body. Also, write down your thoughts without judgment to recognize where stress challenges you. If movement helps you relax, try light stretching or simple yoga exercises to release muscle tension."
    ],
    "bipolar": [
        "If you're currently in a calmer manic phase, reducing stimulants and alcohol may help. If you feel you're in a manic episode, ask yourself questions before acting impulsively: “Why am I making this decision?” and “How will I feel afterward?” Track your mood daily (e.g., rate it from 0 to 10). Instead of suppressing intense emotions, write them down: “Which part of me is dominant today?” or “Is there a constant part of me amid these ups and downs?” Understanding these contrasts with the help of a therapist can help you build a more integrated connection with yourself.",
        "Mood swings can be intense in bipolar disorder, so maintaining structure is important. Try to keep a consistent sleep schedule and reduce stimulants. Track your mood daily and ask yourself questions like “What will be the outcome of this action?” before making decisions. Even when your energy is high, don’t forget to take breaks. Finding the part of you that stays constant during these emotional waves and evaluating the process with a therapist can help you establish balance."
    ],
    "personality disorder": [
        "Make sure to note the moments during the day when you feel triggered and what you’re feeling at those moments. This awareness can help you develop the habit of “pausing to reflect” before reacting impulsively. Later, review the automatic thoughts that arose and perform a benefit–harm analysis: “What did this give me, and what did it cost me?” This process will help you better understand your boundaries and those of others. Also, working with a mental health professional can help deepen your insight. Seeking help from a professional will support better self-understanding and healthier relationships.",
        "At the end of the day, start recording brief notes about moments when you noticed yourself reacting impulsively. Realizing which situations challenge you and identifying recurring patterns can help you develop emotional regulation skills. Try to observe the difficulties you face in social interactions and identify which behavioral patterns repeat. Ask yourself: “Is this behavior truly serving me?” Along with this awareness, working with a therapist to improve your boundary-setting skills is a valuable step."
    ]
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    user_input = data.get("text", "").strip()

    if not user_input:
        return jsonify({})

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

    suggestion_data = suggestions.get(predicted_label, suggestions["normal"])
    if predicted_label == "depression":
        selected_pair = random.choice(suggestion_data)
        for suggestion in selected_pair:
            messages.append({"sender": "bot", "text": suggestion})
    else:
        random_suggestion = random.choice(suggestion_data)
        messages.append({"sender": "bot", "text": random_suggestion})

    now = datetime.utcnow()
    for i, msg in enumerate(messages):
        msg["timestamp"] = (now + timedelta(seconds=i)).isoformat() + "Z"

    return jsonify({
        "label": predicted_label,
        "messages": messages
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
  
