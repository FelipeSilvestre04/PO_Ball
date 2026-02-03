"""
Comparação: Global CI vs Benchmark (VALORES CORRETOS do benchmark_results_2.txt)
"""
import numpy as np
import os
import sys
import time

sys.path.append('c:/Users/felip/Documents/GitHub/PO_Ball/python')
from problems.sequencing.sequenciamento import Sequenciamento

INSTANCES_DIR = os.path.join(os.path.dirname(__file__), 'instances')

# ============================================================================
# GLOBAL CI (Best Greedy)
# ============================================================================
def global_ci(env):
    n, m = env.num_products, env.num_machines
    solution = [[] for _ in range(m)]
    loads = [0.0] * m
    remaining = set(range(n))
    
    while remaining:
        best_cost = float('inf')
        best_p, best_m, best_pos = -1, -1, -1
        best_time = 0.0
        
        for p in remaining:
            for mach in range(m):
                rate = env.production_rates[p, mach]
                if rate <= 1e-6:
                    continue
                
                prod_time = env.demands[p] / rate
                seq = solution[mach]
                
                for pos in range(len(seq) + 1):
                    prev = env.initial_state[mach] if pos == 0 else seq[pos-1]
                    next_p = seq[pos] if pos < len(seq) else -1
                    
                    cost_add = env.setup_costs[prev, p, mach]
                    cost_rem = 0.0
                    if next_p != -1:
                        cost_add += env.setup_costs[p, next_p, mach]
                        cost_rem = env.setup_costs[prev, next_p, mach]
                    delta_cost = cost_add - cost_rem
                    
                    time_add = env.setup_times[prev, p, mach]
                    time_rem = 0.0
                    if next_p != -1:
                        time_add += env.setup_times[p, next_p, mach]
                        time_rem = env.setup_times[prev, next_p, mach]
                    delta_time = time_add + prod_time - time_rem
                    
                    if loads[mach] + delta_time <= env.machine_capacities[mach]:
                        if delta_cost < best_cost:
                            best_cost = delta_cost
                            best_p, best_m, best_pos = p, mach, pos
                            best_time = delta_time
        
        if best_p == -1:
            break
        
        solution[best_m].insert(best_pos, best_p)
        loads[best_m] += best_time
        remaining.remove(best_p)
    
    return solution

def calc_cost(env, solution):
    total = 0.0
    for m, seq in enumerate(solution):
        prev = env.initial_state[m]
        for p in seq:
            total += env.setup_costs[prev, p, m]
            prev = p
    return total

# Benchmark Literature - VALORES CORRETOS do benchmark_results_2.txt
BENCHMARK = {
    '100_5_v2.txt': 427.57,
    '100_10_v2.txt': 613.52,
    '100_15_v2.txt': 575.26,
    '200_5_v2.txt': 572.67,
    '200_10_v2.txt': 781.75,
    '200_15_v2.txt': 784.63,
    '300_5_v2.txt': 622.44,
    '300_10_v2.txt': 890.69,
    '300_15_v2.txt': 952.65,
    '400_5_v2.txt': 572.80,
    '400_10_v2.txt': 934.67,
    '400_15_v2.txt': 1064.49,
    '500_5_v2.txt': 631.36,
    '500_10_v2.txt': 952.40,
    '500_15_v2.txt': 1110.36,
    '600_5_v2.txt': 677.01,
    '600_10_v2.txt': 1084.05,
    '600_15_v2.txt': 1251.28,
    '700_5_v2.txt': 702.32,
    '700_10_v2.txt': 979.60,
    '700_15_v2.txt': 1299.57,
}

if __name__ == "__main__":
    print("=" * 85)
    print("GLOBAL CI vs BENCHMARK (Valores CORRETOS do benchmark_results_2.txt)")
    print("=" * 85)
    
    instances = [f'{n}_{m}_v2.txt' for n in range(100, 701, 100) for m in [5, 10, 15]]
    
    print(f"{'Instância':<18} {'GlobalCI':>10} {'Alocados':>10} {'Benchmark':>10} {'Gap':>10} {'Tempo':>8}")
    print("-" * 85)
    
    results = []
    for inst in instances:
        try:
            env = Sequenciamento(inst)
            
            start = time.time()
            sol = global_ci(env)
            elapsed = time.time() - start
            
            cost = calc_cost(env, sol)
            n_alloc = sum(len(s) for s in sol)
            bench = BENCHMARK.get(inst, 0)
            gap = (cost - bench) / bench * 100 if bench > 0 else 0
            
            status = "✅" if n_alloc == env.num_products else "❌"
            print(f"{status} {inst:<16} {cost:>10.2f} {n_alloc:>6}/{env.num_products:<3} {bench:>10.2f} {gap:>9.1f}% {elapsed:>7.1f}s")
            
            results.append({'inst': inst, 'cost': cost, 'alloc': n_alloc, 'n': env.num_products, 'bench': bench, 'gap': gap})
        except Exception as e:
            print(f"❌ {inst:<16} ERRO: {e}")
    
    print()
    print("=" * 85)
    valid = [r for r in results if r['alloc'] == r['n']]
    print(f"Válidas: {len(valid)}/{len(results)}")
    if valid:
        avg_gap = np.mean([r['gap'] for r in valid])
        print(f"Gap médio (válidas): {avg_gap:.1f}%")
    
    # Count instances where Global CI beats benchmark
    beats = [r for r in results if r['cost'] < r['bench']]
    print(f"Instâncias onde Global CI < Benchmark: {len(beats)}")
    if beats:
        print("  Vencem:", [r['inst'] for r in beats])
