"""
Comparação Completa: Nearest Neighbor vs RKO vs Benchmark
Roda NN em todas as 21 instâncias e compara resultados
"""
import numpy as np
import os
import sys
import time

# Setup paths
sys.path.append('c:/Users/felip/Documents/GitHub/PO_Ball/python')
from problems.sequencing.sequenciamento import Sequenciamento

INSTANCES_DIR = os.path.join(os.path.dirname(__file__), 'instances')

# ============================================================================
# NEAREST NEIGHBOR HEURISTIC
# ============================================================================
def nearest_neighbor(env):
    """
    Heurística Nearest Neighbor: sempre escolhe o produto mais barato
    para adicionar ao final de qualquer máquina.
    """
    n, m = env.num_products, env.num_machines
    solution = [[] for _ in range(m)]
    loads = [0.0] * m
    current = list(env.initial_state)  # Último produto de cada máquina
    
    remaining = set(range(n))
    
    while remaining:
        best_cost = float('inf')
        best_p, best_m = -1, -1
        best_time = 0.0
        
        for p in remaining:
            for mach in range(m):
                rate = env.production_rates[p, mach]
                if rate <= 1e-6:
                    continue
                    
                prod_time = env.demands[p] / rate
                setup_time = env.setup_times[current[mach], p, mach]
                total_time = setup_time + prod_time
                
                if loads[mach] + total_time <= env.machine_capacities[mach]:
                    cost = env.setup_costs[current[mach], p, mach]
                    if cost < best_cost:
                        best_cost = cost
                        best_p, best_m = p, mach
                        best_time = total_time
        
        if best_p == -1:
            break
            
        solution[best_m].append(best_p)
        loads[best_m] += best_time
        current[best_m] = best_p
        remaining.remove(best_p)
    
    n_allocated = sum(len(s) for s in solution)
    return solution, n_allocated

def calc_cost(env, solution):
    """Calcula custo total da solução"""
    total = 0.0
    for m, seq in enumerate(solution):
        prev = env.initial_state[m]
        for p in seq:
            total += env.setup_costs[prev, p, m]
            prev = p
    return total

# ============================================================================
# RESULTADOS DO RKO 1200s (extraídos de execuções anteriores)
# ============================================================================
# Formato: (custo_melhor, tempo_usado)
RKO_RESULTS = {
    # Esses valores devem ser preenchidos com resultados reais do RKO
    # Por enquanto, usando valores aproximados baseados em testes anteriores
    '100_5_v2.txt': 153.0,
    '100_10_v2.txt': 108.0,
    '100_15_v2.txt': 75.0,
    '200_5_v2.txt': 288.0,
    '200_10_v2.txt': 219.0,
    '200_15_v2.txt': 102.0,
    '300_5_v2.txt': 442.0,
    '300_10_v2.txt': 290.0,
    '300_15_v2.txt': 187.0,
    '400_5_v2.txt': 895.0,  # Valor observado
    '400_10_v2.txt': 390.0,
    '400_15_v2.txt': 265.0,
    '500_5_v2.txt': 700.0,
    '500_10_v2.txt': 500.0,
    '500_15_v2.txt': 350.0,
    '600_5_v2.txt': 850.0,
    '600_10_v2.txt': 600.0,
    '600_15_v2.txt': 400.0,
    '700_5_v2.txt': 1000.0,
    '700_10_v2.txt': 700.0,
    '700_15_v2.txt': 500.0,
}

# ============================================================================
# BENCHMARK LITERATURA (Li & Milne + Gurobi)
# ============================================================================
BENCHMARK_RESULTS = {
    '100_5_v2.txt': 149.84,
    '100_10_v2.txt': 107.13,
    '100_15_v2.txt': 72.72,
    '200_5_v2.txt': 286.89,
    '200_10_v2.txt': 219.92,
    '200_15_v2.txt': 99.29,
    '300_5_v2.txt': 433.90,
    '300_10_v2.txt': 282.03,
    '300_15_v2.txt': 181.64,
    '400_5_v2.txt': 572.80,
    '400_10_v2.txt': 383.63,
    '400_15_v2.txt': 262.99,
    '500_5_v2.txt': 681.95,
    '500_10_v2.txt': 493.80,
    '500_15_v2.txt': 340.60,
    '600_5_v2.txt': 823.88,
    '600_10_v2.txt': 563.10,
    '600_15_v2.txt': 389.79,
    '700_5_v2.txt': 960.26,
    '700_10_v2.txt': 649.73,
    '700_15_v2.txt': 454.39,
}

# ============================================================================
# MAIN - RODA NN EM TODAS AS INSTÂNCIAS
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("COMPARAÇÃO: NEAREST NEIGHBOR vs RKO 1200s vs BENCHMARK (Li & Milne)")
    print("=" * 80)
    print()
    
    # Lista de instâncias
    instances = []
    for n in range(100, 701, 100):
        for m in [5, 10, 15]:
            instances.append(f'{n}_{m}_v2.txt')
    
    results = []
    
    print(f"{'Instância':<20} {'NN Custo':>12} {'NN Alocados':>12} {'Benchmark':>12} {'Gap NN/Bench':>12}")
    print("-" * 80)
    
    for inst in instances:
        inst_path = os.path.join(INSTANCES_DIR, inst)
        if not os.path.exists(inst_path):
            print(f"{inst:<20} ARQUIVO NÃO ENCONTRADO")
            continue
        
        try:
            # Carrega instância
            env = Sequenciamento(inst)
            
            # Roda Nearest Neighbor
            start = time.time()
            sol, n_allocated = nearest_neighbor(env)
            nn_time = time.time() - start
            
            nn_cost = calc_cost(env, sol)
            bench_cost = BENCHMARK_RESULTS.get(inst, 0)
            
            # Calcula gap
            if bench_cost > 0:
                gap = (nn_cost - bench_cost) / bench_cost * 100
            else:
                gap = 0
            
            results.append({
                'instance': inst,
                'nn_cost': nn_cost,
                'nn_allocated': n_allocated,
                'nn_time': nn_time,
                'benchmark': bench_cost,
                'gap': gap,
                'n_products': env.num_products
            })
            
            status = "✅" if n_allocated == env.num_products else "❌"
            print(f"{status} {inst:<18} {nn_cost:>12.2f} {n_allocated:>8}/{env.num_products:<3} {bench_cost:>12.2f} {gap:>11.1f}%")
            
        except Exception as e:
            print(f"{inst:<20} ERRO: {e}")
    
    # Sumário
    print()
    print("=" * 80)
    print("SUMÁRIO")
    print("=" * 80)
    
    valid_results = [r for r in results if r['nn_allocated'] == r['n_products']]
    
    if valid_results:
        avg_gap = np.mean([r['gap'] for r in valid_results])
        avg_time = np.mean([r['nn_time'] for r in valid_results]) * 1000  # ms
        
        print(f"Instâncias válidas: {len(valid_results)}/{len(results)}")
        print(f"Gap médio NN vs Benchmark: {avg_gap:.1f}%")
        print(f"Tempo médio NN: {avg_time:.1f}ms")
    
    # Salva resultados em CSV
    output_path = os.path.join(os.path.dirname(__file__), 'Results', 'nn_comparison.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("Instance,NN_Cost,NN_Allocated,NN_Time_ms,Benchmark,Gap_Percent\n")
        for r in results:
            f.write(f"{r['instance']},{r['nn_cost']:.2f},{r['nn_allocated']},{r['nn_time']*1000:.1f},{r['benchmark']:.2f},{r['gap']:.2f}\n")
    
    print(f"\nResultados salvos em: {output_path}")
