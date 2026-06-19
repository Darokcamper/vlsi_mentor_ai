from graphs.discussion_graph import run_graph

result = run_graph(
    "Why are lockup latches needed?"
)

print("\nFINAL ANSWER\n")
print(result["final_answer"])