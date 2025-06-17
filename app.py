from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import pickle
import numpy as np
from textblob import TextBlob
import rapidfuzz
import re
import uvicorn
import random

# Initialize FastAPI app
app = FastAPI(title="LEA Bot Assistant", description="A banking chatbot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic model for request body
class ChatRequest(BaseModel):
    user_input: str

# Load data and models
try:
    # Load banking.csv
    df = pd.read_csv("banking.csv")
    df = df.dropna(subset=["question", "response"])
    train_size = int(0.8 * len(df))
    train_questions = df[:train_size]["question"].tolist()
    train_responses = df[:train_size]["response"].tolist()
    test_questions = df[train_size:]["question"].tolist()
    test_responses = df[train_size:]["response"].tolist()

    # Load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # Load fine-tuned model
    with open("fine_tuned_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load question embeddings
    with open("question_embeddings.pkl", "rb") as f:
        question_embeddings = pickle.load(f)

except FileNotFoundError as e:
    raise Exception(f"Error: File not found - {e}")
except Exception as e:
    raise Exception(f"Error loading files: {e}")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Memory & context
user_memory = {"name": None, "location": None}
last_topic = None

# Variations
greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "howdy", "greetings", "what's up", "yo", "hiya"]
thanks = ["thank you", "thanks", "thx", "appreciate", "much obliged", "cheers", "thankful", "ty", "gracias", "merci"]
affirmatives = ["yes", "yeah", "sure", "ok", "alright", "definitely", "of course", "yea", "yup", "certainly"]
off_topic_keywords = ["food", "love", "music", "football", "weather", "play", "movie", "hiking", "robot", "interact"]

# Domain-specific keywords and responses
domain_keywords = {
    "profit": "Profit and loss topics are best discussed with a financial advisor. I'm happy to assist with banking needs like savings or cards."
}

# Dynamic responses
greet_replies = [
    "Hey {name}, how can I help you today?", "Nice to see you, {name}. What can I do for you?",
    "Hi {name}! Ready to explore some banking options?", "Hello {name}, let's take care of your banking needs.",
    "Welcome back, {name}! How can I assist?", "Hello there, {name}. What would you like help with today?",
    "Hi {name}, what banking task are we tackling today?", "Hey {name}, how can I support you?",
    "Hey {name}, let's sort your banking needs.", "Greetings {name}, happy to help!"
]

thank_replies = [
    "You're welcome!", "Happy to help!", "Anytime!", "No problem, {name}!",
    "Glad to assist!", "Always here to help!", "You're most welcome!", "With pleasure!",
    "No worries!", "Sure thing, {name}!"
]

off_topic_replies = [
    "Haha, that’s fun! I focus on banking. Want to check your balance or report a card issue?",
    "Interesting! I specialize in banking. Would you like help with loans or ATM info?",
    "Cool! I'm trained for banking help. Try asking about your account or a transaction.",
    "That's a fun topic! I'm focused on banking though. Need help with cards or savings?",
    "I love that! Let’s talk banking—need help with something like fraud or PINs?",
    "Nice one! My zone is banking. Want help opening an account?",
    "Haha, I feel you! I handle account inquiries and loans best.",
    "That’s outside my expertise. Want to know your balance instead?",
    "Sounds exciting! But I’m more into balances and transfers!",
    "I wish I could help with that! But I’m your banking assistant"
]

fallback_replies = [
    "I'm here for your banking needs—like loans, fraud, or PIN help. What do you need?",
    "Sorry, I didn’t quite get that. Would you like help with opening an account?",
    "Interesting! I focus on banking. Try asking about ATM, transfers, or cards.",
    "Hmm, I’m not sure. But I’m great at deposits, balances, and reports!",
    "I’m better at banking questions. Want to check your balance or card status?",
    "Could you rephrase that? I can help with things like savings or fraud issues.",
    "Let’s get back to banking. Do you want to know about transfers or cards?",
    "Sorry, didn’t follow. But I can help with loans, accounts, or deposits.",
    "That’s a bit unclear. Do you need help with your account or a transaction?",
    "Let’s stay on topic—banking help like ATM, PIN, or transfers coming up?"
]

# Helper functions
def correct_spelling(text):
    return str(TextBlob(text).correct())

def update_memory(text):
    global user_memory
    name_match = re.search(r"(my name is|i am|i'm|this is|call me|you can call me|it's|they call me|name's)\s+(\w+)", text.lower())
    name_correction = re.search(r"not (\w+)", text.lower())
    loc_match = re.search(r"(i live in|i'm from|i stay in)\s+([a-zA-Z\s]+)", text.lower())

    response = ""
    if name_match:
        user_memory["name"] = name_match.group(2).capitalize()
        response += f"Nice to meet you, {user_memory['name']}! "
    elif name_correction and user_memory["name"]:
        corrected_name = name_correction.group(1).capitalize()
        user_memory["name"] = corrected_name
        response += f"Got it! I’ve updated your name to {corrected_name}. "

    if loc_match:
        user_memory["location"] = loc_match.group(2).strip().capitalize()
        response += f"{user_memory['location']} sounds like a great place. "

    if response:
        response += "How can I assist you with banking today?"
        return response
    return None

def personalize(text):
    return text.replace("{name}", user_memory["name"] if user_memory["name"] else "there")

def detect_intent(text):
    text = text.lower()
    for g in greetings:
        if rapidfuzz.fuzz.partial_ratio(g, text) > 90:
            return "greet"
    for t in thanks:
        if rapidfuzz.fuzz.partial_ratio(t, text) > 90:
            return "thanks"
    for w in off_topic_keywords:
        if w in text:
            return "off_topic"
    if "remember" in text and "name" in text:
        return "check_name"
    if any(word in text for word in affirmatives):
        return "confirm"
    if "talk like a human" in text or "interact like a human" in text:
        return "human_mode"
    if "more about" in text or "tell me more" in text:
        return "clarify"
    return "question"

# API Endpoints
@app.get("/")
def read_root():
    return {"message": "Welcome to LEA Bot Assistant! Visit /docs for the API documentation."}

@app.post("/chat")
async def chat(request: ChatRequest):
    global last_topic
    user_input = request.user_input.strip()

    if not user_input:
        return {"response": "Please provide some input to continue!"}

    corrected = correct_spelling(user_input)
    memory_reply = update_memory(corrected)
    if memory_reply:
        return {"response": memory_reply}

    for keyword in domain_keywords:
        if keyword in corrected.lower():
            return {"response": domain_keywords[keyword]}

    intent = detect_intent(corrected)

    if intent == "greet":
        return {"response": personalize(random.choice(greet_replies))}
    if intent == "thanks":
        return {"response": personalize(random.choice(thank_replies))}
    if intent == "off_topic":
        return {"response": random.choice(off_topic_replies)}
    if intent == "check_name":
        if user_memory["name"]:
            return {"response": f"Yes! You told me your name is {user_memory['name']}."}
        return {"response": "I don’t think you’ve told me your name yet. Try saying 'My name is ...'."}
    if intent == "human_mode":
        return {"response": "Absolutely! I’m here to chat in a friendly way and support your banking needs 😊"}
    if intent == "confirm":
        if last_topic:
            return {"response": f"Continuing from our last topic: **{last_topic}**. What exactly would you like to know?"}
        return {"response": "Sure! Can you share more about what you'd like help with?"}
    if intent == "clarify":
        if last_topic:
            return {"response": f"Here's more on your last topic: **{last_topic}**. Do you want to open an account or understand account types?"}
        return {"response": "Can you clarify which topic you'd like to dive into more?"}

    # Semantic similarity
    inputs = tokenizer(corrected, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        user_emb = model(**inputs).last_hidden_state.mean(dim=1).cpu()
    scores = [torch.nn.functional.cosine_similarity(user_emb, q_emb.unsqueeze(0)).item() for q_emb in question_embeddings]
    best_idx = scores.index(max(scores))
    best_score = max(scores)

    if best_score > 0.4:
        last_topic = train_questions[best_idx]
        return {"response": personalize(train_responses[best_idx])}
    return {"response": random.choice(fallback_replies)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
