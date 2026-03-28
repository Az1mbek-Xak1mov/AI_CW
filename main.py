import io
import sys
from contextlib import redirect_stdout

from ml.risk_predictor import RiskPredictor
from rules.student_copilot_rules import StudentCopilotRules
from planner.study_schedule_planner import Task, astar_schedule, schedule_to_text

class StudentSuccessCopilot:
    def __init__(self):
        print("Initializing Student Success Copilot...")
        # Step A: Initialize ML Predictor
        self.predictor = RiskPredictor()
        
        # Suppress the extensive print output from the training phase for a cleaner console
        print("Training Risk Predictor model...")
        with redirect_stdout(io.StringIO()):
            self.predictor.train_and_evaluate()
            
        # Initialize Rule Engine and Planner (Planner logic is module-level)
        self.rules = StudentCopilotRules()
        
    def analyze_student(self, student_profile: dict, tasks: list):
        print("\n" + "="*60)
        print("                 STUDENT ANALYSIS REPORT")
        print("="*60)
        
        # Step A: ML Prediction
        risk_labels = {0: "Low", 1: "Medium", 2: "High"}
        try:
            risk_score = self.predictor.predict_student(student_profile)
            predicted_risk = risk_labels.get(risk_score, "Unknown")
        except Exception as e:
            predicted_risk = f"Error predicting risk: {e}"
            
        print(f"[A] Predicted Risk Level: {predicted_risk}")
        
        # Step B: Forward Chaining for Explanation
        # We merge some keys since ML and Rules used slightly different names (e.g., stress vs stress_level)
        eval_profile = student_profile.copy()
        eval_profile['stress'] = student_profile.get('stress_level', 0)
        eval_profile['confidence'] = student_profile.get('confidence_level', 0)
        
        inferred_facts, trace = self.rules.evaluate_state(eval_profile)
        print("\n[B] Explanation of Risk (Rule Engine Deductions):")
        if not trace:
            print("  - No critical risk factors identified.")
        else:
            for step in trace:
                print(f"  - {step}")
                
        # Step C: Backward Chaining for Missing Data
        combined_facts = {**eval_profile, **inferred_facts}
        
        print("\n[C] Follow-up Questions (Backward Chaining):")
        goals_to_check = ['needs_tutor', 'needs_counselor']
        questions_asked = []
        for goal in goals_to_check:
            success, question = self.rules.backward_chain(goal, combined_facts)
            if not success and question:
                questions_asked.append(question)
        
        if questions_asked:
            for q in set(questions_asked):
                print(f"  - ACTION REQUIRED: {q}")
        else:
            print("  - Information sufficient. No follow-up questions at this time.")
            
        # Step D: Search Planner Generation using A*
        print("\n[D] Recommended Weekly Schedule (A* Search):")
        
        # Generate the schedule (cap daily tasks to 3 for workload balance)
        schedule_state = astar_schedule(tasks, num_days=7, daily_capacity=3)
        schedule_text = schedule_to_text(schedule_state)
        
        for line in schedule_text.split('\n'):
            print(f"  {line}")
            
        print("="*60 + "\n")


if __name__ == "__main__":
    copilot = StudentSuccessCopilot()
    
    # ---------------------------------------------------------
    # Scenario 1: A 'Normal' Student
    # ---------------------------------------------------------
    sim_1_profile = {
        'workload_hours': 10,
        'available_hours': 40,
        'stress_level': 2,
        'confidence_level': 9,
        'missed_sessions': 0,
        'sleep_hours': 8,
        'deadline_proximity': 'far'
    }
    
    sim_1_tasks = [
        Task("Read Chapter 1", workload_hours=2, deadline_days=5),
        Task("Easy Math HW", workload_hours=3, deadline_days=6),
        Task("Review Notes", workload_hours=1, deadline_days=7)
    ]
    
    print("\n>>> RUNNING DEMO SCENARIO 1: 'Normal' Student <<<")
    copilot.analyze_student(sim_1_profile, sim_1_tasks)
    
    # ---------------------------------------------------------
    # Scenario 2: An 'At-Risk' Student
    # ---------------------------------------------------------
    # High stress, low confidence, missed sessions, tight deadlines.
    sim_2_profile = {
        'workload_hours': 60,
        'available_hours': 15,
        'stress_level': 9,
        'confidence_level': 2,
        'missed_sessions': 4,
        'sleep_hours': 4, # triggers burnout risk
        'deadline_proximity': 'close'
        # Intentionally omitting quiz_average_below_70 to test backward chaining asking for it
    }
    
    sim_2_tasks = [
        Task("Final Project Draft", workload_hours=5, deadline_days=2),
        Task("Study for Midterm", workload_hours=4, deadline_days=3),
        Task("Catch up on lectures", workload_hours=6, deadline_days=4)
    ]
    
    print("\n>>> RUNNING DEMO SCENARIO 2: 'At-Risk' Student <<<")
    copilot.analyze_student(sim_2_profile, sim_2_tasks)
