"""
predict_system.py
-------------------
Interactive command-line system that:
  1. Takes a student's academic/demographic inputs
  2. Predicts their final marks using the trained Gradient Boosting model
  3. Shows a confidence interval for the prediction
  4. Uses the AI chatbot to generate personalized academic recommendations
"""

import os
import pickle
import pandas as pd

from chatbot import AcademicChatbot

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "rb") as f:
        preprocessor = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "residual_std.pkl"), "rb") as f:
        resid_std = pickle.load(f)
    return model, preprocessor, resid_std


def ask(prompt, cast=float, choices=None):
    while True:
        raw = input(prompt).strip()
        if choices and raw not in choices:
            print(f"  Please enter one of: {choices}")
            continue
        try:
            return cast(raw) if not choices else raw
        except ValueError:
            print("  Please enter a valid number.")


def predict_for_student(student: dict, model, preprocessor, resid_std):
    df = pd.DataFrame([student])
    X = preprocessor.transform(df)
    pred = float(model.predict(X)[0])
    pred = max(0, min(100, pred))

    low = max(0, pred - 1.96 * resid_std)
    high = min(100, pred + 1.96 * resid_std)
    return pred, low, high


def run_cli():
    print("=" * 60)
    print(" STUDENT MARKS PREDICTION + AI ACADEMIC CHATBOT")
    print("=" * 60)

    model, preprocessor, resid_std = load_artifacts()
    bot = AcademicChatbot()

    print("\nEnter student details:\n")
    student = {
        "reading_score": ask("Reading score (0-100): "),
        "writing_score": ask("Writing score (0-100): "),
        "gender": ask("Gender [male/female]: ", cast=str, choices=["male", "female"]),
        "race_ethnicity": ask(
            "Group [group A/group B/group C/group D/group E]: ",
            cast=str,
            choices=["group A", "group B", "group C", "group D", "group E"],
        ),
        "parental_level_of_education": ask(
            "Parental education [some high school/high school/some college/"
            "associate's degree/bachelor's degree/master's degree]: ",
            cast=str,
            choices=[
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ],
        ),
        "lunch": ask("Lunch type [standard/free/reduced]: ", cast=str, choices=["standard", "free/reduced"]),
        "test_preparation_course": ask(
            "Completed test preparation course? [completed/none]: ",
            cast=str,
            choices=["completed", "none"],
        ),
    }

    pred, low, high = predict_for_student(student, model, preprocessor, resid_std)

    print("\n" + "-" * 60)
    print(f"Predicted Final Marks: {pred:.1f} / 100")
    print(f"Estimated range (95% confidence): {low:.1f} - {high:.1f}")
    print("-" * 60)

    print("\nPersonalized Recommendations:")
    for i, tip in enumerate(bot.recommendations_from_profile(student), 1):
        print(f"  {i}. {tip}")

    print("\nAsk the academic chatbot anything (type 'exit' to quit):")
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye! All the best for your exams.")
            break
        response = bot.get_response(query)
        print(f"Bot: {response['answer']}")


if __name__ == "__main__":
    run_cli()
