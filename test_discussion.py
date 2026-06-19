from agents.discussion_agent import run_discussion

result = run_discussion(
    "Why are lockup latches needed?"
)

print("\n")
print("=" * 80)
print("FINAL ANSWER")
print("=" * 80)

print(result["final"])