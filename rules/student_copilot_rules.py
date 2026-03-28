from typing import Dict, Any, List, Tuple, Callable

class Rule:
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool], conclusion: str, certainty: float):
        self.name = name
        self.condition = condition
        self.conclusion = conclusion
        self.certainty = certainty

class BackwardRule:
    def __init__(self, conclusion: str, premises: List[str]):
        self.conclusion = conclusion
        self.premises = premises

class StudentCopilotRules:
    def __init__(self):
        self.forward_rules = [
            Rule(
                "High Academic Risk",
                lambda f: f.get('missed_sessions', 0) >= 2 and f.get('confidence', 10) < 5,
                "high_academic_risk",
                0.8
            ),
            Rule(
                "Burnout Risk",
                lambda f: f.get('stress', 0) > 7 and f.get('sleep_hours', 8) < 6,
                "burnout_risk",
                0.9
            ),
            Rule(
                "Needs Intervention",
                lambda f: f.get('high_academic_risk', 0) > 0.5 or f.get('burnout_risk', 0) > 0.5,
                "needs_intervention",
                1.0
            ),
            Rule(
                "Deadline Panic",
                lambda f: f.get('deadline_proximity') == 'close' and f.get('stress', 0) >= 6,
                "deadline_panic",
                0.75
            ),
            Rule(
                "Low Engagement",
                lambda f: f.get('missed_sessions', 0) >= 3,
                "low_engagement",
                0.85
            )
        ]

        self.backward_rules = [
            BackwardRule("needs_tutor", ["high_academic_risk", "quiz_average_low"]),
            BackwardRule("quiz_average_low", ["quiz_average_below_70"]),
            BackwardRule("needs_counselor", ["burnout_risk", "requests_help"])
        ]
        
        self.questions = {
            "quiz_average_low": "What is your current quiz average?",
            "quiz_average_below_70": "Is your quiz average below 70?",
            "requests_help": "Are you open to speaking with a counselor?"
        }

    def evaluate_state(self, facts: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
        inferred = {}
        trace = []
        changed = True
        
        # Combine user facts with inferred facts for condition checking
        current_state = facts.copy()
        
        while changed:
            changed = False
            for rule in self.forward_rules:
                if rule.conclusion not in inferred:
                    try:
                        if rule.condition(current_state):
                            inferred[rule.conclusion] = rule.certainty
                            current_state[rule.conclusion] = rule.certainty
                            trace.append(f"Fired Rule: {rule.name} -> Deduced '{rule.conclusion}' with certainty {rule.certainty}")
                            changed = True
                    except Exception:
                        pass
        return inferred, trace

    def backward_chain(self, goal: str, facts: Dict[str, Any], rules: List[BackwardRule] = None) -> Tuple[bool, str | None]:
        if rules is None:
            rules = self.backward_rules
            
        # If the goal is already a known fact
        if goal in facts or (goal in [r.conclusion for r in self.forward_rules] and facts.get(goal, 0) > 0):
            return True, None
            
        # Find rules that can prove the goal
        relevant_rules = [r for r in rules if r.conclusion == goal]
        
        if not relevant_rules:
            # It's a base fact we need to ask the user
            return False, self.questions.get(goal, f"What is the status of {goal}?")
            
        for rule in relevant_rules:
            rule_satisfied = True
            for premise in rule.premises:
                # Check if premise is known
                if premise in facts and facts[premise]:
                    continue
                
                # Try to prove the premise recursively
                success, question = self.backward_chain(premise, facts, rules)
                if not success:
                    rule_satisfied = False
                    if question:
                        return False, question
                    break
            
            if rule_satisfied:
                return True, None
                
        return False, None


def test_expert_system():
    system = StudentCopilotRules()
    
    print("--- Forward Chaining Trace ---")
    initial_facts = {
        'stress': 8,
        'confidence': 3,
        'missed_sessions': 2,
        'sleep_hours': 5,
        'deadline_proximity': 'close'
    }
    print(f"Initial facts: {initial_facts}\n")
    
    inferred_facts, trace = system.evaluate_state(initial_facts)
    for step in trace:
        print(step)
        
    print(f"\nAll Deductions: {inferred_facts}")
    
    print("\n\n--- Backward Chaining Example ---")
    # To prove needs_tutor, we need 'high_academic_risk' (which we have from forward chaining)
    # and 'quiz_average_low' (which we don't have)
    
    combined_facts = {**initial_facts, **inferred_facts}
    goal = 'needs_tutor'
    
    print(f"Goal: {goal}")
    success, question = system.backward_chain(goal, combined_facts)
    
    if success:
        print(f"Goal '{goal}' successfully proven!")
    else:
        print(f"Cannot prove '{goal}' yet. Missing information.")
        print(f"System asks: {question}")
        
if __name__ == '__main__':
    test_expert_system()
