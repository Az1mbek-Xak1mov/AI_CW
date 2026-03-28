import os
import sys

# Add parent directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import StudentSuccessCopilot
from planner.study_schedule_planner import Task, astar_schedule, schedule_to_text


class ChatInterface:
    def __init__(self):
        print("Initializing Chatbot Interface...")
        self.copilot = StudentSuccessCopilot()

    def _get_int_input(self, prompt: str, min_val: int = None, max_val: int = None) -> int:
        while True:
            try:
                val = int(input(prompt).strip())
                if min_val is not None and val < min_val:
                    print(f"Please enter a number at least {min_val}.")
                    continue
                if max_val is not None and val > max_val:
                    print(f"Please enter a number at most {max_val}.")
                    continue
                return val
            except ValueError:
                print("Invalid input. Please enter a valid number.")

    def start_chat(self):
        print("\n" + "=" * 60)
        print("       Welcome to the Student Success Copilot Chat!")
        print("=" * 60)
        print("\nI'm here to help you analyze your academic workload and create a balanced schedule.\n")

        # ---------------------------------------------------------
        # Information Gathering
        # ---------------------------------------------------------
        profile = {}
        profile['workload_hours'] = self._get_int_input("How many hours of workload do you have this week? ")
        profile['available_hours'] = self._get_int_input("How many hours are you available to study this week? ")
        profile['stress_level'] = self._get_int_input("On a scale of 1-10, what is your current stress level? ", 1, 10)
        profile['confidence_level'] = self._get_int_input("On a scale of 1-10, how confident are you this week? ", 1, 10)
        profile['missed_sessions'] = self._get_int_input("How many classes or study sessions have you missed? ", 0)
        
        # Additional fields for the Rule Engine dependencies
        profile['sleep_hours'] = self._get_int_input("How many hours of sleep are you getting per night (e.g., 6)? ", 0, 24)
        
        while True:
            prox = input("Are your deadlines 'close' or 'far'? ").strip().lower()
            if prox in ('close', 'far'):
                profile['deadline_proximity'] = prox
                break
            print("Please enter 'close' or 'far'.")

        # ---------------------------------------------------------
        # Task Collection
        # ---------------------------------------------------------
        print("\nGreat! Now let's list your upcoming tasks.")
        tasks = []
        while True:
            name = input("\nEnter a task name (or type 'done' to finish): ").strip()
            if name.lower() == 'done':
                break
            if not name:
                continue
            
            workload = self._get_int_input(f"How many hours will '{name}' take? ", 1)
            deadline = self._get_int_input(f"In how many days is '{name}' due? ", 1)
            tasks.append(Task(name, workload, deadline))

        print("\nThank you! Analyzing your profile...\n")
        
        # ---------------------------------------------------------
        # Processing & Interactive Loop
        # ---------------------------------------------------------
        # Map fields so both ML predictor and Rule Engine recognize them
        eval_profile = profile.copy()
        eval_profile['stress'] = profile.get('stress_level', 0)
        eval_profile['confidence'] = profile.get('confidence_level', 0)
        
        # Step A: ML Prediction
        risk_labels = {0: "Low", 1: "Medium", 2: "High"}
        try:
            risk_score = self.copilot.predictor.predict_student(eval_profile)
            predicted_risk = risk_labels.get(risk_score, "Unknown")
        except Exception as e:
            predicted_risk = f"Error: {e}"

        print(f"[A] ML Predicted Risk Level: {predicted_risk}")

        # Step B: Rule Engine Deductions
        inferred_facts, trace = self.copilot.rules.evaluate_state(eval_profile)
        print("\n[B] Explanation (Rule Engine Deductions):")
        if not trace:
            print("  - No critical risk factors identified.")
        else:
            for step in trace:
                print(f"  - {step}")

        # Step C: Backward Chaining / Interactive Follow-ups
        combined_facts = {**eval_profile, **inferred_facts}
        goals_to_check = ['needs_tutor', 'needs_counselor']
        
        print("\n[C] Checking for Missing Information (Backward Chaining):")
        follow_up_asked = False
        
        for goal in goals_to_check:
            success, question = self.copilot.rules.backward_chain(goal, combined_facts)
            if not success and question:
                # Ask the inferred question interactively
                follow_up_asked = True
                print(f"\nFollow-up System Question: {question}")
                ans = input("Your Answer: ").strip()
                # (Optional) could parse the answer dynamically here to feed back into rules
                print(">> Recommendation Update: Noted your response. Please prioritize accordingly.")
                
        if not follow_up_asked:
            print("  - Information sufficient. No follow-up questions at this time.")

        # Step D: Weekly Schedule Configuration
        print("\n" + "=" * 60)
        print("              [D] Your Weekly Schedule (A*)")
        print("=" * 60)
        
        if not tasks:
            print("  No tasks to schedule.")
        else:
            schedule_state = astar_schedule(tasks, num_days=7, daily_capacity=3)
            schedule_text = schedule_to_text(schedule_state)
            for line in schedule_text.split('\n'):
                print(f"  {line}")


if __name__ == '__main__':
    interface = ChatInterface()
    interface.start_chat()
