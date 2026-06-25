import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from graphs.discussion_graph import run_graph

class TestDiscussionGraph(unittest.TestCase):
    
    def test_graph_flow(self):
        """Runs the LangGraph multi-agent discussion workflow and verifies result structure."""
        try:
            result = run_graph("Why are lockup latches needed in scan shift?")
            self.assertIsNotNone(result)
            self.assertIn("participants", result)
            self.assertIn("discussion_history", result)
            self.assertIn("critic_feedback", result)
            self.assertIn("critic_score", result)
            self.assertIn("final_answer", result)
            
            self.assertTrue(len(result["participants"]) > 0)
            self.assertTrue(len(result["discussion_history"]) > 0)
            self.assertTrue(len(result["final_answer"]) > 10)
        except Exception as e:
            self.fail(f"LangGraph execution failed: {e}")

if __name__ == "__main__":
    unittest.main()
