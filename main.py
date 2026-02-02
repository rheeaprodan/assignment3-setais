from config.search_space import param_spec, base_cfg
from policies.pretrained_policy import load_pretrained_policy
from envs.highway_env_utils import make_env
from search.random_search import RandomSearch
from search.hill_climbing import HillClimbingSearch
from search.analysis import analyze_results

def main():
    env_id = "highway-fast-v0"
    policy = load_pretrained_policy("agents/model")
    env, defaults = make_env(env_id)

    search = RandomSearch(env_id, base_cfg, param_spec, policy, defaults)
    rs_crashes = search.run_search(n_scenarios=50, seed=42)
    print(f"✅ Found {len(rs_crashes)} crashes.")
    analyze_results("Random Search", rs_crashes)

    search = HillClimbingSearch(env_id, base_cfg, param_spec, policy, defaults)
    hc_crashes = search.run_search(n_scenarios=50, seed=42)
    analyze_results("Hill Climbing", hc_crashes)
    print(f"✅ Found {len(hc_crashes)} crashes.")


if __name__ == "__main__":
    main()