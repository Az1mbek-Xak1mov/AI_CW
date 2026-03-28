# Explainable Student Success Copilot

## One-sentence pitch
Build a student-facing AI assistant that creates a personalised study plan, detects risk early (e.g., falling behind), and explains its recommendations using a hybrid of search + rules + ML, with responsible use of generative AI.

## Overview
This repository contains a hybrid AI project to evaluate student study risk and generate weekly plans based on workload, deadlines, progress, and student feedback.

### Project structure
- `ml/`
  - `risk_predictor.py`: RandomForest binary classification model (risk level 0/1/2)
- `rules/`
  - `student_copilot_rules.py`: Forward/backward chaining expert system generating risk signals and follow-up prompts
- `planner/`
  - `study_schedule_planner.py`: Search-based scheduler (BFS and A*) creating conflict-minimized weekly schedules
- `ui/`
  - `chatbot.py`: CLI interface collecting user input and invoking analysis pipeline
- `main.py`: orchestrator tying all components into an end-to-end workflow
- `main.ipynb`: notebook-based demo of the same pipeline

## What it does (minimum viable system)
- Inputs:
  - tasks (workload hours, deadlines)
  - availability constraints
  - self-reported stress/confidence
  - attendance, optional numeric fields
- Outputs:
  - weekly study schedule
  - risk level (low/medium/high)
  - explanation text from rule engine
  - interactive follow-up question loop when data is missing or ambiguous

## Running the system
1. Activate environment
```bash
source .venv/bin/activate
```
2. Run main script
```bash
python main.py
```
3. Invoke CLI chatbot
```bash
python ui/chatbot.py
```
4. Run notebook (instead of python directly):
```bash
jupyter notebook main.ipynb
```

## Notes
- The system is intentionally simple and designed as a demo.
- No external APIs are required; only standard Python libs and scikit-learn/pandas/numpy.

## Evaluations
- ML: accuracy + classification report on synthetic generated dataset
- Rule engine: confidence-weighted forward chaining, and backward-chaining follow-up question prompts
- Planner: BFS + A* schedule generation

## Risk/disclaimer
- This is an educational prototype; not clinical advice.
- Outcomes are synthetic and not guaranteed to fit real student behavior.

---

## Contents of repository
- `main.py` orchestrates prediction + rules + planning
- `main.ipynb` notebook demo
- `ml/risk_predictor.py` generates synthetic dataset and trains model
- `rules/student_copilot_rules.py` contains rule engine
- `planner/study_schedule_planner.py` contains scheduler
- `ui/chatbot.py` terminal UI
