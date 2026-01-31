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

import numpy as np
from tqdm import trange, tqdm

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
        print(f"Running Hill Climbing with enhanced exploration...")
        iters = iterations or n_scenarios
        
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
        
        crash_log = []
        if res.get("best_fitness", float("inf")) <= -1.0:
            print(f"💥 Collision found! seed={res.get('best_seed_base')}")
            crash_log.append({"cfg": copy.deepcopy(res.get("best_cfg")), "seed": int(res.get("best_seed_base"))})
            record_video_episode(self.env_id, res.get("best_cfg"), self.policy, self.defaults, int(res.get("best_seed_base")), out_dir="videos")
        else:
            best_fitness = res.get('best_fitness', float("inf"))
            best_dist = res.get("best_objectives", {}).get("min_distance", "N/A")
            print(f"No crash found. Best fitness: {best_fitness:.4f}, Min distance: {best_dist}")
            print(f"Total evaluations: {res.get('evaluations', 'N/A')}")
        
        return crash_log

# ============================================================
# 1) OBJECTIVES FROM TIME SERIES
# ============================================================

def compute_objectives_from_time_series(time_series: List[Dict[str, Any]], crashed_flag: bool = False) -> Dict[str, Any]:
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
    crash_count = 1 if crashed_flag or any(frame.get("crashed", False) for frame in time_series) else 0
    min_distance = float('inf')

    for frame in time_series:
        ego = frame.get('ego', None)
        others = frame.get('others', [])
        if ego is None or not others:
            continue
        
        ego_pos = np.array(ego['pos'])
        ego_length = ego.get('length', 5.0)
        ego_width = ego.get('width', 2.0)
        
        # Ego vehicle bounding box: [x_min, x_max, y_min, y_max]
        ego_x_min = ego_pos[0] - ego_length / 2.0
        ego_x_max = ego_pos[0] + ego_length / 2.0
        ego_y_min = ego_pos[1] - ego_width / 2.0
        ego_y_max = ego_pos[1] + ego_width / 2.0
        
        for car in others:
            car_pos = np.array(car['pos'])
            car_length = car.get('length', 5.0)
            car_width = car.get('width', 2.0)
            
            # Other vehicle bounding box
            car_x_min = car_pos[0] - car_length / 2.0
            car_x_max = car_pos[0] + car_length / 2.0
            car_y_min = car_pos[1] - car_width / 2.0
            car_y_max = car_pos[1] + car_width / 2.0
            
            # Check if rectangles overlap (collision)
            x_overlap = ego_x_max >= car_x_min and ego_x_min <= car_x_max
            y_overlap = ego_y_max >= car_y_min and ego_y_min <= car_y_max
            if x_overlap and y_overlap:
                # Collision detected
                crash_count = 1
                min_distance = 0.0
            else:
                # Calculate minimum distance between rectangles (not center-to-center)
                # Distance in x direction
                if ego_x_max < car_x_min:
                    dist_x = car_x_min - ego_x_max
                elif car_x_max < ego_x_min:
                    dist_x = ego_x_min - car_x_max
                else:
                    dist_x = 0.0
                
                # Distance in y direction
                if ego_y_max < car_y_min:
                    dist_y = car_y_min - ego_y_max
                elif car_y_max < ego_y_min:
                    dist_y = ego_y_min - car_y_max
                else:
                    dist_y = 0.0
                
                # Euclidean distance between closest edges
                distance = np.sqrt(dist_x**2 + dist_y**2)
                if distance < min_distance:
                    min_distance = distance

    if min_distance == float('inf'):
        min_distance = float(1e6)
    
    return {"crash_count": crash_count, "min_distance": float(min_distance)}


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
    rng: np.random.Generator,
    mutation_strength: float = 0.3
) -> Dict[str, Any]:
    """
    Improved mutation: Mutates MULTIPLE parameters to escape local optima.
    """
    cfg_copy = copy.deepcopy(cfg)
    
    # Mutate 1 to 3 parameters simultaneously (aggressive exploration)
    # The more parameters we have, the more we should be willing to change.
    num_params = len(param_spec)
    num_mutations = rng.integers(1, min(4, num_params + 1))
    
    params_to_mutate = rng.choice(list(param_spec.keys()), size=num_mutations, replace=False)

    for param_to_mutate in params_to_mutate:
        param_info = param_spec[param_to_mutate]
        param_type = param_info['type']
        param_min = param_info['min']
        param_max = param_info['max']

        if param_type == 'int':
            current = cfg_copy.get(param_to_mutate, (param_min + param_max) // 2)
            range_width = param_max - param_min
            # Ensure step is at least 1, scale by mutation_strength
            step = max(1, int(range_width * mutation_strength))
            
            # Random jump within +/- step
            change = rng.integers(-step, step + 1)
            new_value = current + change
            new_value = int(np.clip(new_value, param_min, param_max))
        
        elif param_type == 'float':
            current = cfg_copy.get(param_to_mutate, (param_min + param_max) / 2.0)
            range_width = param_max - param_min
            step = range_width * mutation_strength
            
            # Gaussian perturbation
            new_value = current + rng.normal(0, step)
            new_value = float(np.clip(new_value, param_min, param_max))
        
        cfg_copy[param_to_mutate] = new_value

    # Constraint check: Ensure initial_lane_id is valid for the new lanes_count
    if 'lanes_count' in cfg_copy and 'initial_lane_id' in cfg_copy:
        lanes_count = int(cfg_copy['lanes_count'])
        # If lanes decreased, clamp the lane ID
        cfg_copy['initial_lane_id'] = int(np.clip(cfg_copy['initial_lane_id'], 0, max(0, lanes_count - 1)))
    
    return cfg_copy


# def mutate_config(
#     cfg: Dict[str, Any],
#     param_spec: Dict[str, Any],
#     rng: np.random.Generator,
#     mutation_strength: float = 0.3
# ) -> Dict[str, Any]:
#     """
#     Generate ONE neighbor configuration by mutating the current scenario.

#     Inputs:
#       - cfg: current scenario dict (e.g., vehicles_count, initial_spacing, ego_spacing, initial_lane_id)
#       - param_spec: search space bounds, types (int/float), min/max
#       - rng: random generator
#       - mutation_strength: how aggressive the mutation is (0.0-1.0)

#     Requirements:
#       - Do NOT modify cfg in-place (return a copy).
#       - Keep mutated values within [min, max] from param_spec.
#       - If you mutate lanes_count, keep initial_lane_id valid (0..lanes_count-1).

#     Students can implement:
#       - single-parameter mutation (recommended baseline)
#       - multiple-parameter mutation
#       - adaptive step sizes, etc.
#     """
#     cfg_copy = copy.deepcopy(cfg)
#     param_to_mutate = rng.choice(list(param_spec.keys()))
#     param_info = param_spec[param_to_mutate]
#     param_type = param_info['type']
#     param_min = param_info['min']
#     param_max = param_info['max']

#     if param_type == 'int':
#         # More aggressive mutation: perturb around current value with wider range
#         current = cfg_copy.get(param_to_mutate, (param_min + param_max) // 2)
#         range_width = param_max - param_min
#         step = max(1, int(range_width * mutation_strength))
#         new_value = current + rng.integers(-step, step + 1)
#         new_value = int(np.clip(new_value, param_min, param_max))
#     elif param_type == 'float':
#         # Perturb around current value with adaptive range
#         current = cfg_copy.get(param_to_mutate, (param_min + param_max) / 2.0)
#         range_width = param_max - param_min
#         step = range_width * mutation_strength
#         new_value = current + rng.normal(0, step)
#         new_value = float(np.clip(new_value, param_min, param_max))
    
#     cfg_copy[param_to_mutate] = new_value

#     if param_to_mutate == 'lanes_count' and 'initial_lane_id' in cfg_copy:
#         lanes_count = int(cfg_copy['lanes_count'])
#         cfg_copy['initial_lane_id'] = int(np.clip(cfg_copy['initial_lane_id'], 0, max(0, lanes_count - 1)))
    
#     return cfg_copy


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
    
    rng = np.random.default_rng(seed)

    best_cfg = None
    best_obj = None
    best_fit = float("inf")
    best_seed_base = None
    best_ts = None
    history = []
    total_evals = 0

    # FIX 1: Increase Restarts. 
    # Try 5 distinct starting points to find different local optima.
    num_restarts = 5 
    
    # Budget allocation: Split iterations among restarts
    iters_per_restart = max(5, iterations // num_restarts)

    for restart in range(num_restarts):
        print(f"\n🔄 Restart {restart + 1}/{num_restarts}")
        
        # FIX 2: Warm Start (Population Initialization)
        # Generate 10 random configs and pick the most dangerous one to start this restart
        ss = ScenarioSearch(env_id, base_cfg, param_spec, policy, defaults)
        candidate_cfgs = [ss.sample_random_config(rng) for _ in range(10)]
        
        start_cfg = candidate_cfgs[0]
        start_fit = float('inf')
        
        # Quick pre-check (optional, but helps start strong)
        # If simulation is too slow, just pick candidate_cfgs[0] directly.
        # For now, let's just pick one random one to save time, OR 
        # uncomment the block below to use the "best of 3" start:
        """
        best_start_idx = 0
        for idx, c in enumerate(candidate_cfgs[:3]):
            _cr, _ts = run_episode(env_id, c, policy, defaults, seed)
            _obj = compute_objectives_from_time_series(_ts, _cr)
            _f = compute_fitness(_obj)
            if _f < start_fit:
                start_fit = _f
                start_cfg = c
        """
        # Default to simple random start to save eval budget
        start_cfg = candidate_cfgs[0] 

        # Evaluate initial solution
        seed_base = int(rng.integers(1e9))
        crashed, ts = run_episode(env_id, start_cfg, policy, defaults, seed_base)
        obj = compute_objectives_from_time_series(ts, crashed_flag=crashed)
        cur_fit = compute_fitness(obj)
        total_evals += 1
        
        current_cfg = copy.deepcopy(start_cfg) # Ensure we are working on the evaluated config

        restart_best_cfg = copy.deepcopy(current_cfg)
        restart_best_obj = dict(obj)
        restart_best_fit = float(cur_fit)
        restart_best_seed = seed_base
        restart_best_ts = ts # Save the time series

        history.append(restart_best_fit)
        print(f"  👉 Start Fitness: {cur_fit:.4f}")

        # Hill climbing loop
        plateau_count = 0
        max_plateau = 15  # FIX 3: Increased patience from 5 to 15

        for it in trange(iters_per_restart, desc=f"Restart {restart + 1} HC", leave=True):
            # Annealing: reduce mutation strength over time
            # Starts high (exploration) -> Ends low (exploitation)
            mutation_strength = 0.6 * (1.0 - it / iters_per_restart) + 0.1
            
            best_neighbor = None
            best_neighbor_fit = float("inf")
            best_neighbor_obj = None
            best_neighbor_ts = None
            best_neighbor_seed = None

            # Generate and evaluate neighbors
            for _ in range(neighbors_per_iter):
                neighbor_cfg = mutate_config(current_cfg, param_spec, rng, mutation_strength=mutation_strength)
                s = int(rng.integers(1e9))
                
                crashed_n, ts_n = run_episode(env_id, neighbor_cfg, policy, defaults, s)
                total_evals += 1
                
                obj_n = compute_objectives_from_time_series(ts_n, crashed_flag=crashed_n)
                fit_n = compute_fitness(obj_n)

                if fit_n < best_neighbor_fit:
                    best_neighbor_fit = fit_n
                    best_neighbor = neighbor_cfg
                    best_neighbor_obj = obj_n
                    best_neighbor_ts = ts_n
                    best_neighbor_seed = s
            
            # Acceptance Logic
            # FIX 4: Allow side-stepping on plateaus (<= instead of <)
            # This allows the search to drift on flat landscapes
            if best_neighbor is not None and best_neighbor_fit <= cur_fit:
                if best_neighbor_fit < cur_fit:
                    plateau_count = 0 # Reset if strictly better
                    print(f"    ⬇️ Improved local: {best_neighbor_fit:.4f}")
                else:
                    plateau_count += 1 # Count plateau steps
                
                current_cfg = copy.deepcopy(best_neighbor)
                cur_fit = best_neighbor_fit
            else:
                plateau_count += 1

            # Update global restart best
            if cur_fit < restart_best_fit:
                restart_best_fit = float(cur_fit)
                restart_best_cfg = copy.deepcopy(current_cfg)
                restart_best_obj = dict(best_neighbor_obj) if best_neighbor_obj else dict(restart_best_obj)
                restart_best_seed = int(best_neighbor_seed) if best_neighbor_seed else restart_best_seed
                restart_best_ts = best_neighbor_ts
                print(f"  ✅ Restart Best Updated: {restart_best_fit:.4f}")

            history.append(restart_best_fit)

            # Check for crash
            if restart_best_fit <= -1.0:
                print(f"💥 CRASH FOUND at restart {restart + 1}!")
                return {
                    "best_cfg": restart_best_cfg,
                    "best_objectives": restart_best_obj,
                    "best_fitness": restart_best_fit,
                    "best_seed_base": restart_best_seed,
                    "history": history,
                    "best_time_series": restart_best_ts,
                    "evaluations": total_evals,
                }

            if plateau_count > max_plateau:
                print(f"  🛑 Plateau reached ({max_plateau} iters). Stopping restart.")
                break

        # End of restart: Update global best
        if restart_best_fit < best_fit:
            best_fit = float(restart_best_fit)
            best_cfg = copy.deepcopy(restart_best_cfg)
            best_obj = dict(restart_best_obj)
            best_seed_base = int(restart_best_seed)
            best_ts = restart_best_ts
            print(f"  🏆 New Global Best: {best_fit:.4f}")

    return {
        "best_cfg": best_cfg,
        "best_objectives": best_obj,
        "best_fitness": best_fit,
        "best_seed_base": best_seed_base,
        "history": history,
        "best_time_series": best_ts,
        "evaluations": total_evals,
    }


# def hill_climb(
#     env_id: str,
#     base_cfg: Dict[str, Any],
#     param_spec: Dict[str, Any],
#     policy,
#     defaults: Dict[str, Any],
#     seed: int = 0,
#     iterations: int = 100,
#     neighbors_per_iter: int = 10,
# ) -> Dict[str, Any]:
#     """
#     Hill climbing loop with random restarts.

#     You should:
#       1) Start from an initial scenario (base_cfg or random sample).
#       2) Evaluate it by running:
#             crashed, ts = run_episode(env_id, cfg, policy, defaults, seed_base)
#          Then compute objectives + fitness.
#       3) For each iteration:
#             - Generate neighbors_per_iter neighbors using mutate_config
#             - Evaluate each neighbor
#             - Select the best neighbor
#             - Accept it if it improves fitness (or implement another acceptance rule)
#             - Optionally stop early if a crash is found
#       4) Return the best scenario found and enough info to reproduce.

#     Return dict MUST contain at least:
#         {
#           "best_cfg": Dict[str, Any],
#           "best_objectives": Dict[str, Any],
#           "best_fitness": float,
#           "best_seed_base": int,
#           "history": List[float]
#         }

#     Optional but useful:
#         - "best_time_series": ts
#         - "evaluations": int
#     """
#     rng = np.random.default_rng(seed)

#     best_cfg = None
#     best_obj = None
#     best_fit = float("inf")
#     best_seed_base = None
#     best_ts = None
#     history = []
#     total_evals = 0

#     # Multiple restarts for better exploration
#     num_restarts = 1
#     iters_per_restart = iterations // num_restarts

#     for restart in range(num_restarts):
#         print(f"\n🔄 Restart {restart + 1}/{num_restarts}")
#         # Initialize from random config (more effective than base_cfg)
#         ss = ScenarioSearch(env_id, base_cfg, param_spec, policy, defaults)
#         current_cfg = ss.sample_random_config(rng)

#         # Evaluate initial solution
#         seed_base = int(rng.integers(1e9))
#         crashed, ts = run_episode(env_id, current_cfg, policy, defaults, seed_base)
#         obj = compute_objectives_from_time_series(ts, crashed_flag=crashed)
#         cur_fit = compute_fitness(obj)
#         total_evals += 1

#         restart_best_cfg = copy.deepcopy(current_cfg)
#         restart_best_obj = dict(obj)
#         restart_best_fit = float(cur_fit)
#         restart_best_seed = seed_base

#         history.append(restart_best_fit)

#         # Hill climbing for this restart
#         plateau_count = 0
#         max_plateau = 5

#         for it in trange(iters_per_restart, desc=f"Restart {restart + 1} HC", leave=True):
#             # Adaptive mutation strength: decrease over time
#             mutation_strength = 0.5 * (1.0 - it / iters_per_restart) + 0.1
            
#             best_neighbor = None
#             best_neighbor_fit = float("inf")
#             best_neighbor_obj = None
#             best_neighbor_ts = None
#             best_neighbor_seed = None

#             for _ in tqdm(range(neighbors_per_iter), desc="Neighbors", leave=False, disable=True):
#                 neighbor_cfg = mutate_config(current_cfg, param_spec, rng, mutation_strength=mutation_strength)
#                 s = int(rng.integers(1e9))
#                 crashed_n, ts_n = run_episode(env_id, neighbor_cfg, policy, defaults, s)
#                 total_evals += 1
#                 obj_n = compute_objectives_from_time_series(ts_n, crashed_flag=crashed_n)
#                 fit_n = compute_fitness(obj_n)

#                 if fit_n < best_neighbor_fit:
#                     best_neighbor_fit = fit_n
#                     best_neighbor = neighbor_cfg
#                     best_neighbor_obj = obj_n
#                     best_neighbor_ts = ts_n
#                     best_neighbor_seed = s

#             # Accept if improved (greedy hill climbing)
#             if best_neighbor is not None and best_neighbor_fit < cur_fit:
#                 current_cfg = copy.deepcopy(best_neighbor)
#                 cur_fit = best_neighbor_fit
#                 plateau_count = 0
#             else:
#                 plateau_count += 1

#             # Update global best
#             if best_neighbor_fit < restart_best_fit:
#                 restart_best_fit = float(best_neighbor_fit)
#                 restart_best_cfg = copy.deepcopy(best_neighbor)
#                 restart_best_obj = dict(best_neighbor_obj)
#                 restart_best_seed = int(best_neighbor_seed)
#                 restart_best_ts = best_neighbor_ts
#                 print(f"  ✅ Improved! Fitness: {restart_best_fit:.4f}")

#             history.append(restart_best_fit)

#             # Early exit if crash found
#             if restart_best_fit <= -1.0:
#                 print(f"💥 CRASH FOUND at restart {restart + 1}!")
#                 best_cfg = restart_best_cfg
#                 best_obj = restart_best_obj
#                 best_fit = restart_best_fit
#                 best_seed_base = restart_best_seed
#                 best_ts = restart_best_ts
#                 result = {
#                     "best_cfg": best_cfg,
#                     "best_objectives": best_obj,
#                     "best_fitness": best_fit,
#                     "best_seed_base": best_seed_base,
#                     "history": history,
#                     "best_time_series": best_ts,
#                     "evaluations": total_evals,
#                 }
#                 return result

#             # Early exit if plateau too long
#             if plateau_count > max_plateau:
#                 break

#         # Update global best from this restart
#         if restart_best_fit < best_fit:
#             best_fit = float(restart_best_fit)
#             best_cfg = copy.deepcopy(restart_best_cfg)
#             best_obj = dict(restart_best_obj)
#             best_seed_base = int(restart_best_seed)
#             best_ts = restart_best_ts
#             print(f"  📊 New global best: {best_fit:.4f}")

#     result = {
#         "best_cfg": best_cfg,
#         "best_objectives": best_obj,
#         "best_fitness": best_fit,
#         "best_seed_base": best_seed_base,
#         "history": history,
#         "best_time_series": best_ts,
#         "evaluations": total_evals,
#     }
#     return result