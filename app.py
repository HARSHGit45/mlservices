from flask import Flask, request, jsonify
import pickle
import re
import os
import numpy as np
import gym
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model & vectorizer
ML_MODEL_DIR = "ml_models"
vectorizer_path = os.path.join(ML_MODEL_DIR, "vectorizer.pkl")
classifier_path = os.path.join(ML_MODEL_DIR, "spending_classifier.pkl")

with open(classifier_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

@app.route('/classify_spending', methods=['POST'])
def classify_spending():
    try:
        description = request.json.get("description", "")
        if not description:
            return jsonify({"error": "Description is required"}), 400

        text_vectorized = vectorizer.transform([description])
        prediction = model.predict(text_vectorized)
        return jsonify({"category": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Reinforcement Learning Budget Allocation
class BudgetEnv(gym.Env):
    def __init__(self, prev_budget, total_income, fixed_rent):
        super(BudgetEnv, self).__init__()
        self.total_income = total_income
        self.fixed_rent = fixed_rent  
        self.prev_budget = np.array(prev_budget)
        self.expense_categories = len(prev_budget)

        self.action_space = gym.spaces.Box(low=-0.2, high=0.2, shape=(self.expense_categories,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.expense_categories,), dtype=np.float32)

        self.state = np.array(prev_budget)

    def step(self, action):
        new_budget = self.state + (self.state * action)  
        new_budget = np.maximum(new_budget, 0)  
        total_expense = self.fixed_rent + np.sum(new_budget)  
        savings = self.total_income - total_expense  
        reward = savings if savings >= 0 else -100
        return new_budget, reward, True, {}

    def reset(self):
        return self.state


def train_rl_agent(prev_budget, total_income, fixed_rent):
    env = BudgetEnv(prev_budget, total_income, fixed_rent)
    state = env.reset()
    best_allocation = state
    best_reward = float('-inf')

    for _ in range(1000):
        action = np.random.uniform(-0.2, 0.2, size=(len(prev_budget),))
        new_budget, reward, _, _ = env.step(action)
        if reward > best_reward and np.sum(new_budget) + fixed_rent <= total_income:
            best_reward = reward
            best_allocation = new_budget

    savings = total_income - (fixed_rent + np.sum(best_allocation))
    return list(best_allocation), round(savings, 2)

@app.route('/allocate_budget', methods=['POST'])
def allocate_budget():
    try:
        data = request.json
        total_income = float(data["income"])
        fixed_rent = float(data["rent"])
        prev_budget = [float(data["food"]), float(data["clothing"]), float(data["education"]), float(data["misc"])]
        new_budget, savings = train_rl_agent(prev_budget, total_income, fixed_rent)
        
        return jsonify({
            "message": "New budget allocated dynamically while maximizing savings",
            "total_income": total_income,
            "previous_budget": {
                "rent": fixed_rent,
                "food": prev_budget[0],
                "clothing": prev_budget[1],
                "education": prev_budget[2],
                "misc": prev_budget[3]
            },
            "new_budget": {
                "rent": fixed_rent,
                "food": round(new_budget[0], 2),
                "clothing": round(new_budget[1], 2),
                "education": round(new_budget[2], 2),
                "misc": round(new_budget[3], 2)
            },
            "savings": savings
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("server running")
    app.run(debug=True)
