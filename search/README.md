# Assignment 3: Scenario-Based Testing of an RL Agent

This project implements **Hill Climbing (Steepest Descent)** to discover failure scenarios (collisions) for a pre-trained **PPO agent** in a highway driving environment.  
It compares the effectiveness of Hill Climbing against a **Random Search** baseline.

---

## 🛠️ Setup and Installation

### Python Version
- The project was tested with **Python 3.11**.

### Dependencies
All required dependencies are listed in the provided `requirements.txt` file.  

## 🚀 How to Run

The main entry point of the project is `main.py`.

When executed with the default configuration, it:
- Runs **Random Search** for a fixed budget (default: 50 iterations)
- Runs **Hill Climbing** for a fixed budget (default: 50 iterations)
- Prints crash logs, vehicle counts, and configuration details to the console
- Saves videos of crash scenarios to the `videos/` directory

---

## ⚠️ Special Setup: Analysis Module

To keep the search logic clean and consistent across search methods, all statistical analysis is separated into a dedicated module named `analysis.py`.

### Crucial Implementation Detail

The function `analyze_results` from the analysis module **must be explicitly imported and called** from `main.py`.  
It should be invoked using the results returned by the `run_search` method.

### Expected Usage Flow

1. Run a search method (Random Search or Hill Climbing)
2. Collect the returned crash results and history
3. Pass these results to `analyze_results` for consistent evaluation

This design ensures that both search methods are evaluated using identical metrics and objectives.

---

## 📊 How to Reproduce Report Results

To reproduce the exact numerical results and qualitative behaviors discussed in the **Experimental Evaluation** section of the report, `main.py` must be run using specific random seeds.

### Global Configuration
- **Number of scenarios (iterations):** 50  
- **Neighbors per iteration (Hill Climbing only):** 10

### Reproduction Steps

1. Open `main.py`
2. Locate the `seed` parameter in the `run_search` calls
3. Set the seed to one of the values listed below
4. Run the script once per seed to observe the corresponding behavior

### Seeds and Observed Phenomena

| Seed | Phenomenon Observed |
|-----:|---------------------|
| 11 | **High vs. Low Density**: Random Search finds a high-density crash (51 cars), while Hill Climbing minimizes traffic to find a geometric crash (6.7 cars). |
| 37 | **Stress Testing**: Hill Climbing converges to a high-density scenario (31 cars). |
| 123 | **Geometric Exploitation**: Hill Climbing finds 22 distinct crashes in sparse traffic (10 cars), exploiting a weakness in Lane 4. |
| 42 | **Contrast Case**: Random Search “gets lucky” with a sparse crash (10 cars), while Hill Climbing evolves a dense traffic jam (34 cars). |
