from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbingSearch
from collections import Counter
import numpy as np

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    # search = HillClimbingSearch(env_id, base_cfg, param_spec, policy, defaults)
    # crashes, history = search.run_search(n_scenarios=5, seed=13)

    # print(f"✅ Found {len(crashes)} crashes.")

    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    rs_crashes = search.run_search(n_scenarios=50, seed=11)

    print(f"✅ Found {len(rs_crashes)} crashes.")
    analyze_results("Random Search", rs_crashes)

    search = HillClimbingSearch(env_id, base_cfg, param_spec, policy, defaults)
    hc_crashes = search.run_search(n_scenarios=50, seed=11)
    analyze_results("Hill Climbing", hc_crashes)

    print(f"✅ Found {len(hc_crashes)} crashes.")
    #if crashes:
    #    print(crashes)


def analyze_results(method_name, crash_log):
    """
    Analyzes the crash log to extract scenario characteristics and patterns.
    """
    print(f"\n{'='*20} {method_name} ANALYSIS {'='*20}")
    
    if not crash_log:
        print("No crashes found to analyze.")
        print("="*60)
        return

    # 1. Safely Extract parameters (use defaults if key is missing)
    vehicles_counts = [c['cfg'].get('vehicles_count', 0) for c in crash_log]
    spacings = [c['cfg'].get('initial_spacing', 0.0) for c in crash_log]
    lanes = [c['cfg'].get('initial_lane_id', 'N/A') for c in crash_log]
    durations = [c['cfg'].get('duration', 0) for c in crash_log]

    # 2. Statistics (Pattern Detection)
    print(f"Total Crashes: {len(crash_log)}")
    
    # Vehicles
    if vehicles_counts:
        avg_v = np.mean(vehicles_counts)
        print(f"Avg Vehicle Count: {avg_v:.2f} (Range: {min(vehicles_counts)}-{max(vehicles_counts)})")
    
    # Spacing
    if spacings:
        avg_s = np.mean(spacings)
        print(f"Avg Initial Spacing: {avg_s:.2f} (Range: {min(spacings):.2f}-{max(spacings):.2f})")
    
    # Lane Distribution
    lane_counts = Counter(lanes)
    print(f"Crashes by Initial Lane: {dict(lane_counts)}")

    # 3. Most Critical Scenario (Last one found is usually the most 'evolved' or latest)
    critical = crash_log[-1] 
    crit_cfg = critical.get('cfg', {})
    
    print(f"\n[Most Critical Scenario Configuration]")
    print(f"  - Vehicles: {crit_cfg.get('vehicles_count', 'N/A')}")
    print(f"  - Spacing:  {crit_cfg.get('initial_spacing', 'N/A')}")
    print(f"  - Lane:     {crit_cfg.get('initial_lane_id', 'N/A')}")
    print(f"  - Duration: {crit_cfg.get('duration', 'N/A')}")
  
    

    print("="*60)

if __name__ == "__main__":
    main()