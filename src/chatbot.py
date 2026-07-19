"""
chatbot.py
-----------
A lightweight AI academic chatbot that uses TF-IDF vectorization and
cosine similarity to match a student's free-text query against a
knowledge base of study-related Q&A, returning the most relevant
recommendation.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


KNOWLEDGE_BASE = [
    {
        "tags": ["study hours", "time management", "how much should i study", "studying schedule", "daily routine"],
        "question": "How many hours should I study daily?",
        "answer": (
            "Aim for consistent, focused study sessions rather than long unfocused ones. "
            "3-4 hours of deep, distraction-free study (using techniques like the "
            "Pomodoro method) is often more effective than 6+ hours of passive reading."
        ),
    },
    {
        "tags": ["attendance", "class", "skipping"],
        "question": "Does attendance really affect my marks?",
        "answer": (
            "Yes. Attendance correlates strongly with performance because you catch "
            "context, hints about exams, and doubts get cleared in real time. "
            "Aim for at least 85% attendance."
        ),
    },
    {
        "tags": ["sleep", "rest", "tired", "sleeping", "sleepy", "exhausted", "fatigue", "drowsy"],
        "question": "How does sleep affect academic performance?",
        "answer": (
            "Sleep directly impacts memory consolidation and focus. 7-8 hours of sleep "
            "is ideal. Sacrificing sleep to cram often backfires by hurting recall "
            "during exams."
        ),
    },
    {
        "tags": ["assignments", "homework", "submission", "deadlines"],
        "question": "Why are assignments important?",
        "answer": (
            "Assignments reinforce concepts through active recall and often directly "
            "map to exam patterns. Completing them on time also builds discipline and "
            "usually contributes to internal/continuous assessment marks."
        ),
    },
    {
        "tags": ["previous marks", "past performance", "weak subject", "backlog"],
        "question": "I did badly last time, how do I improve?",
        "answer": (
            "Start by identifying which specific topics caused the drop, not just the "
            "overall score. Revisit fundamentals in those topics, practice previous "
            "years' questions, and consider a peer study group for accountability."
        ),
    },
    {
        "tags": ["extracurricular", "balance", "sports", "activities"],
        "question": "How much time should I give to extracurriculars?",
        "answer": (
            "Extracurriculars are valuable for well-rounded growth, but keep them "
            "time-boxed (1-2 hours/day) during exam-heavy periods so they support, "
            "rather than compete with, your study schedule."
        ),
    },
    {
        "tags": ["group study", "study group", "peer learning"],
        "question": "Is studying in a group helpful?",
        "answer": (
            "Group study can help through explaining concepts to each other, spotting "
            "gaps in understanding, and staying motivated - as long as the group stays "
            "focused and doesn't turn into a distraction."
        ),
    },
    {
        "tags": ["exam", "revision", "last minute", "exam prep", "before exam", "final week"],
        "question": "How should I prepare in the final week before exams?",
        "answer": (
            "Shift from learning new material to active revision: solve past papers, "
            "do timed mock tests, and revisit your weak topics using summary notes "
            "rather than re-reading full textbooks."
        ),
    },
    {
        "tags": ["test preparation course", "test prep", "coaching", "prep course"],
        "question": "Does a test preparation course actually help?",
        "answer": (
            "Yes - structured test prep exposes you to the exact question formats "
            "and timing constraints you'll face, which reduces exam-day surprises. "
            "Even a short prep course tends to correlate with meaningfully higher "
            "scores, since it turns passive knowledge into exam-ready practice."
        ),
    },
    {
        "tags": ["motivation", "procrastination", "focus"],
        "question": "I keep procrastinating, what should I do?",
        "answer": (
            "Break study goals into small, specific tasks (e.g., '30 minutes on "
            "chapter 3 problems') rather than vague ones. Removing phone "
            "distractions and using a visible progress tracker also helps a lot."
        ),
    },
    {
        "tags": ["internet", "online resources", "youtube", "courses"],
        "question": "Are online resources useful for improving marks?",
        "answer": (
            "Yes - platforms with structured courses and practice problems can "
            "supplement classroom learning well, especially for topics you find "
            "difficult. Use them to reinforce, not replace, your core coursework."
        ),
    },
]


class AcademicChatbot:
    def __init__(self, knowledge_base=None):
        self.kb = knowledge_base or KNOWLEDGE_BASE
        corpus = [
            " ".join(entry["tags"]) + " " + entry["question"] for entry in self.kb
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.kb_matrix = self.vectorizer.fit_transform(corpus)

    def get_response(self, query: str, threshold: float = 0.20):
        """Returns the best-matching answer for a query, or a fallback message."""
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.kb_matrix).flatten()
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score < threshold:
            return {
                "answer": (
                    "I don't have a specific tip for that yet. Try asking about "
                    "study hours, attendance, sleep, assignments, revision strategy, "
                    "or motivation/procrastination."
                ),
                "confidence": best_score,
                "matched_question": None,
            }

        return {
            "answer": self.kb[best_idx]["answer"],
            "confidence": best_score,
            "matched_question": self.kb[best_idx]["question"],
        }

    def recommendations_from_profile(self, student_features: dict):
        """
        Rule-based + chatbot-assisted personalized recommendations generated
        from a student's profile (used by predict_system.py). Only conditions
        on actionable, behavioral fields (test prep, prior scores) - not on
        demographic attributes like gender, race/ethnicity, or lunch type,
        since giving different "advice" based on those would be inappropriate.
        """
        tips = []

        if student_features.get("test_preparation_course") == "none":
            tips.append(self.get_response("test preparation course")["answer"])

        reading = student_features.get("reading_score", 100)
        writing = student_features.get("writing_score", 100)

        if reading < 60:
            tips.append(
                "Your reading score suggests comprehension practice would help - "
                "try summarizing passages in your own words after reading them, "
                "which also improves recall for other subjects."
            )
        if writing < 60:
            tips.append(
                "Your writing score suggests structured practice would help - "
                "outline your answer before writing it out, and review past "
                "graded work to spot recurring mistakes."
            )
        if abs(reading - writing) > 15:
            tips.append(
                "There's a noticeable gap between your reading and writing "
                "scores - the weaker of the two is worth extra focused practice "
                "since it's likely dragging down your overall average."
            )

        if not tips:
            tips.append(
                "Your current profile looks solid - keep up the consistency, "
                "and focus on active revision (practice tests) closer to exams."
            )
        return tips


if __name__ == "__main__":
    bot = AcademicChatbot()
    test_queries = [
        "how many hours should I study every day?",
        "I feel sleepy all the time, does it affect my grades?",
        "what should i eat before exam",
    ]
    for q in test_queries:
        res = bot.get_response(q)
        print(f"Q: {q}\nA: {res['answer']}\n(confidence: {res['confidence']:.2f})\n")
