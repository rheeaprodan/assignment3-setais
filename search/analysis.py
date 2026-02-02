import numpy as np
from collections import Counter

def analyze_results(method_name, crash_log):
    """
    Analyzes the crash log to extract scenario characteristics and patterns.
    """
    print(f"\n{'='*20} {method_name} ANALYSIS {'='*20}")
    
    if not crash_log:
        print("No crashes found to analyze.")
        print("="*60)
        return

    # 1. Safely Extract parameters
    vehicles_counts = [c['cfg'].get('vehicles_count', 0) for c in crash_log]
    spacings = [c['cfg'].get('initial_spacing', 0.0) for c in crash_log]
    lanes = [c['cfg'].get('initial_lane_id', 'N/A') for c in crash_log]

    # 2. Statistics
    print(f"Total Crashes: {len(crash_log)}")
    
    if vehicles_counts:
        avg_v = np.mean(vehicles_counts)
        print(f"Avg Vehicle Count: {avg_v:.2f} (Range: {min(vehicles_counts)}-{max(vehicles_counts)})")
    
    if spacings:
        avg_s = np.mean(spacings)
        print(f"Avg Initial Spacing: {avg_s:.2f} (Range: {min(spacings):.2f}-{max(spacings):.2f})")
    
    lane_counts = Counter(lanes)
    print(f"Crashes by Initial Lane: {dict(lane_counts)}")

    # 3. Critical Scenario
    critical = crash_log[-1] 
    crit_cfg = critical.get('cfg', {})
    
    print(f"\n[Most Critical Scenario Configuration]")
    print(f"  - Vehicles: {crit_cfg.get('vehicles_count', 'N/A')}")
    print(f"  - Spacing:  {crit_cfg.get('initial_spacing', 'N/A')}")
    print(f"  - Lane:     {crit_cfg.get('initial_lane_id', 'N/A')}")
    print(f"  - Duration: {crit_cfg.get('duration', 'N/A')}")
    

    print("="*60)