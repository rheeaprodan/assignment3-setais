from tqdm import trange
import numpy as np
import copy
from typing import List, Dict, Any, Optional
from envs.highway_env_utils import record_video_episode, run_episode


class RandomSearch:
    def __init__(self, env_id, base_cfg, param_spec, policy, defaults):
        self.env_id = env_id
        self.base_cfg = base_cfg
        self.param_spec = param_spec
        self.policy = policy
        self.defaults = defaults

    def run_search(self, n_scenarios=50, n_eval=1, seed=42):
        print(f"Running Random Search for {n_scenarios} scenarios...")
        rng = np.random.default_rng(seed)
        crash_log = []
        global_min_dist = float('inf')

        for i in trange(n_scenarios, desc="Random search"):
            cfg = self.sample_random_config(rng)
            for j in range(n_eval):
                s = int(rng.integers(1e9))
                crashed, ts = run_episode(self.env_id, cfg, self.policy, self.defaults, s)

                objs = self.compute_objectives(ts)
                current_dist = objs['min_distance']
                
                if current_dist < global_min_dist:
                    global_min_dist = current_dist

                if crashed:
                    print(f"💥 Collision: scenario {i}, seed={s}")
                    crash_log.append({"cfg": copy.deepcopy(cfg), "seed": s})
                    record_video_episode(self.env_id, cfg, self.policy, self.defaults, s, out_dir="videos")
                    break
                else:
                    print(f"No Crash: scenario {i}, seed={s}")
                    # crash_log.append({"cfg": copy.deepcopy(cfg), "seed": s})
                    # record_video_episode(self.env_id, cfg, self.policy, self.defaults, s, out_dir="videos")

        print("\n" + "="*40)
        print(f"RESULTS: Random Search")
        print(f"Total Crashes Found: {len(crash_log)}")
        print(f"Lowest Minimum Distance: {global_min_dist:.2f} m")
        print("="*40 + "\n")
        return crash_log

    def compute_objectives(self, time_series: List[Dict[str, Any]]) -> Dict[str, Any]:
        min_distance = float('inf')
        for frame in time_series:
            ego = frame.get('ego')
            others = frame.get('others', [])
            if ego is not None and others:
                ego_pos = np.array(ego['pos'])
                for car in others:
                    car_pos = np.array(car['pos'])
                    dist = np.linalg.norm(ego_pos - car_pos)
                    if dist < min_distance:
                        min_distance = dist
        
        if min_distance == float('inf'):
            min_distance = 1000.0
            
        return {"min_distance": min_distance}

    def sample_random_config(self, rng):
        from search.base_search import ScenarioSearch
        return ScenarioSearch.sample_random_config(self, rng)