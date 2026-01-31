"""
Assignment 3 — Scenario-Based Testing of an RL Agent (Hill Climbing)

You MUST implement:
    - compute_objectives_from_time_series
    - compute_fitness
    - mutate_config
    - hill_climb

DO NOT change function signatures.
You MAY add helper functions.

Goal
----
Find a scenario (environment configuration) that triggers a collision.
If you cannot trigger a collision, minimize the minimum distance between the ego
vehicle and any other vehicle across the episode.

Black-box requirement
---------------------
Your evaluation must rely only on observable behavior during execution:
- crashed flag from the environment
- time-series data returned by run_episode (positions, lane_id, etc.)
No internal policy/model details beyond calling policy(obs, info).
"""

import copy
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
import numpy as np

from envs.highway_env_utils import run_episode, record_video_episode

from search.base_search import ScenarioSearch


class HillClimbingSearch:
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def run_search(self, n_scenarios=50, seed=42, iterations=None, neighbors_per_iter=10):
        from search.hill_climbing import hill_climb
        iters = iterations or n_scenarios
        print(f"Running Hill Climbing (iters={iters}, neighbors={neighbors_per_iter})...")
        
        res = hill_climb(
            self.env_id,
            self.base_cfg,
            self.param_spec,
            self.policy,
            self.defaults,
            seed=seed,
            iterations=iters,
            neighbors_per_iter=neighbors_per_iter,
        )

        all_crashes = res.get("all_crashes", [])
        
        processed_seeds = set()
        final_crash_log = []

        for crash in all_crashes:
            s = crash["seed"]
            if s in processed_seeds:
                continue
            
            processed_seeds.add(s)
            
            # Add to final log
            final_crash_log.append({"cfg": crash["cfg"], "seed": s})
            
            # Record video
            print(f"Saving video for crash seed: {s}")
            record_video_episode(
                self.env_id, 
                crash["cfg"], 
                self.policy, 
                self.defaults, 
                s, 
                out_dir="videos42"
            )

        if not final_crash_log:
             best_fit = res.get("best_fitness", float("inf"))
             best_dist = res.get("best_objectives", {}).get("min_distance", "N/A")
             print(f"No crashes found. Best fitness: {best_fit:.4f}, Min Dist: {best_dist}")

        return final_crash_log


# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute your objective values from the recorded time-series.

    The time_series is a list of frames. Each frame typically contains:
      - frame["crashed"]: bool
      - frame["ego"]: dict or None, e.g. {"pos":[x,y], "lane_id":..., "length":..., "width":...}
      - frame["others"]: list of dicts with positions, lane_id, etc.

    Minimum requirements (suggested):
      - crash_count: 1 if any collision happened, else 0
      - min_distance: minimum distance between ego and any other vehicle over time (float)

    Return a dictionary, e.g.:
        {
          "crash_count": 0 or 1,
          "min_distance": float
        }

    NOTE: If you want, you can add more objectives (lane-specific distances, time-to-crash, etc.)
    but keep the keys above at least.
    """
    crash_count = 1 if any(frame.get("crashed", False) for frame in time_series) else 0
    min_distance = float('inf')

    for frame in time_series:
        ego = frame.get('ego', None)
        others = frame.get('others', [])
        if ego is not None and others:
            ego_pos = np.array(ego['pos'])
            for car in others:
                car_pos = np.array(car['pos'])
                distance = np.linalg.norm(ego_pos - car_pos)
                if distance < min_distance:
                    min_distance = distance

    return {"crash_count": crash_count, "min_distance": min_distance}


def compute_fitness(objectives: Dict[str, Any]) -> float:
    """
    Convert objectives into ONE scalar fitness value to MINIMIZE.

    Requirement:
    - Any crashing scenario must be strictly better than any non-crashing scenario.

    Examples:
    - If crash_count==1: fitness = -1 (best)
    - Else: fitness = min_distance (smaller is better)

    You can design a more refined scalarization if desired.
    """
    fitness = -1.0 if objectives.get('crash_count', 0) > 0 else objectives.get('min_distance', float('inf'))
    return fitness


# ============================================================
# 2) MUTATION / NEIGHBOR GENERATION
# ============================================================

def mutate_config(
    cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    rng: np.random.Generator
) -> Dict[str, Any]:
    """
    Generate ONE neighbor configuration by mutating the current scenario.

    Inputs:
      - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
      - param_spec: search space bounds, types (int/float), min/max
      - rng: random generator

    Requirements:
      - Do NOT modify cfg in-place (return a copy).
      - Keep mutated values within [min, max] from param_spec.
      - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

    Students can implement:
      - single-parameter mutation (recommended baseline)
      - multiple-parameter mutation
      - adaptive step sizes, etc.
    """
    # TODO (students)
    cfg_copy = copy.deepcopy(cfg)
    param_to_mutate = rng.choice(list(param_spec.keys()))
    param_info = param_spec[param_to_mutate]
    param_type = param_info['type']

    if param_type == 'int':
        new_value = rng.integers(param_info['min'], param_info['max'] + 1)
    elif param_type =='float':
        new_value = rng.uniform(param_info['min'], param_info['max'])
    
    cfg_copy[param_to_mutate] = new_value

    if param_to_mutate == 'lanes_count' and 'initial_lane_id' in cfg_copy:
        lanes_count = cfg_copy['lanes_count']
        cfg_copy['initial_lane_id'] = int(np.clip(cfg_copy['initial_lane_id'], 0, lanes_count - 1))
    
    return cfg_copy


# ============================================================
# 3) HILL CLIMBING SEARCH
# ============================================================

def hill_climb(
    env_id: str,
    base_cfg: Dict[str, Any],
    param_spec: Dict[str, Any],
    policy,
    defaults: Dict[str, Any],
    seed: int = 0,
    iterations: int = 100,
    neighbors_per_iter: int = 10,
) -> Dict[str, Any]:
    """
    Hill climbing loop.

    You should:
      1) Start from an initial scenario (base_cfg or random sample).
      2) Evaluate it by running:
            crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
         Then compute objectives + fitness.
      3) For each iteration:
            - Generate neighbors_per_iter neighbors using mutate_config
            - Evaluate each neighbor
            - Select the best neighbor
            - Accept it if it improves fitness (or implement another acceptance rule)
            - Optionally stop early if a crash is found
      4) Return the best scenario found and enough info to reproduce.

    Return dict MUST contain at least:
        {
          "best_cfg": Dict[str, Any],
          "best_objectives": Dict[str, Any],
          "best_fitness": float,
          "best_seed_base": int,
          "history": List[float]
        }

    Optional but useful:
        - "best_time_series": ts
        - "evaluations": int
    """
    rng = np.random.default_rng(seed)

    # TODO (students): choose initialization (base_cfg or random scenario)
    current_cfg = copy.deepcopy(base_cfg)

    # Evaluate initial solution (seed_base used for reproducibility)
    seed_base = int(rng.integers(1e9))
    crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
    if crashed and len(ts) > 0:
        print("Crash detected in hill climbing evaluation.")
        ts[-1]["crashed"] = True
    obj = compute_objectives_from_time_series(ts)
    cur_fit = compute_fitness(obj)

    best_res = {
        "best_cfg": copy.deepcopy(current_cfg),
        "best_objectives": obj,
        "best_fitness": cur_fit,
        "best_seed_base": seed_base,
        "history": [cur_fit],
        "all_crashes": []
    }
    
    # if cur_fit <= -1.0:
    #     return best_res

    history = [cur_fit]
    pbar = tqdm(range(iterations), desc="Hill Climbing")

    # TODO (students): implement HC loop
    # - generate neighbors
    # - evaluate
    # - pick best
    # - accept if improved
    # - early stop on crash (optional)

    

    for _ in pbar:
        iteration_candidates = []
        for j in range(neighbors_per_iter):
            neighbor_cfg = mutate_config(current_cfg, param_spec, rng)
            seed_base = int(rng.integers(1e9))
            crashed, ts = run_episode(env_id, neighbor_cfg, policy, defaults, seed_base)
            if crashed and len(ts) > 0:
                ts[-1]["crashed"] = True
            obj = compute_objectives_from_time_series(ts)
            fit = compute_fitness(obj)

            candidate = {
                "cfg": neighbor_cfg,
                "objs": obj,
                "fit": fit,
                "seed": seed_base
            }
            iteration_candidates.append(candidate)

            if crashed:
                print("Crash detected in hill climb evaluation.")
                crash_entry = {
                    "cfg": copy.deepcopy(neighbor_cfg), 
                    "seed": seed_base,
                    "fitness": fit
                }
                best_res["all_crashes"].append(crash_entry)
                best_cfg = copy.deepcopy(neighbor_cfg)
                best_obj = dict(obj)
                best_fit = float(fit)
                best_seed_base = seed_base
                history.append(best_fit)
                break
            best_neighbor = min(iteration_candidates, key=lambda x: x["fit"])

            if best_neighbor['fit'] < cur_fit:
                current_cfg = best_neighbor["cfg"]
                cur_fit = best_neighbor["fit"]  

                if cur_fit < best_res["best_fitness"]:
                    best_res["best_cfg"] = copy.deepcopy(current_cfg)
                    best_res["best_objectives"] = best_neighbor["objs"]
                    best_res["best_fitness"] = cur_fit
                    best_res["best_seed_base"] = best_neighbor["seed"]

            best_res["history"].append(best_res["best_fitness"])

            pbar.set_postfix({"Best Fit": f"{best_res['best_fitness']:.4f}"})


    return best_res