"""
main.py
--------
Entry point for the Student Marks Prediction & AI Chatbot project.

Usage:
    python main.py             -> runs the interactive prediction + chatbot CLI
    python src/train_model.py  -> trains/retrains the model on the Kaggle dataset

Dataset: data/kaggle_students_performance.csv
("Students Performance in Exams", Kaggle - spscientist)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from predict_system import run_cli

if __name__ == "__main__":
    run_cli()
