from dotenv import load_dotenv

load_dotenv()

from agents.interviewer_agent import ask_interview_question
from agents.evaluator_agent import evaluate_answer

print("\n=== VLSI Mentor AI ===\n")

while True:

    topic = input(
        "\nTopic (SCAN/ATPG/EDT/MBIST/JTAG) or 'quit': "
    )

    if topic.lower() == "quit":
        break
    
    choice = input(
    "\nDifficulty (1.Beginner/2.Intermediate/3.Advanced): "
    )
    
    difficulty_map = {
    "1": "Beginner",
    "2": "Intermediate",
    "3": "Advanced"
    }

    difficulty = difficulty_map.get(choice)
    
    print("\nGenerating Question...\n")

    question = ask_interview_question(
    topic,
    difficulty
    )

    print("=" * 60)
    print("INTERVIEWER:")
    print(question)
    print("=" * 60)

    answer = input("\nYour Answer:\n\n")

    print("\nEvaluating...\n")

    result = evaluate_answer(
        question,
        answer
    )

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(result)
    print("=" * 60)