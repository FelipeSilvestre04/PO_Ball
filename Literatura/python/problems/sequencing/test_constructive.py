"""
Teste de Heurísticas Construtivas para Scheduling com Setup Dependente
Compara: Nearest Neighbor, Cheapest Insertion, LPT, SPT, Random
"""
import numpy as np
import sys
import os

# Adiciona o caminho do projeto
sys.path.append('c:/Users/felip/Documents/GitHub/PO_Ball/python')
from problems.sequencing.sequenciamento import Sequenciamento

def load_instance(filepath):
    """Carrega instância e retorna ambiente"""
    return Sequenciamento(filepath)

def validate_solution(env, solution, name):
    """Valida que a solução é factível"""
    n, m = env.num_products, env.num_machines
    
    # Verifica produtos alocados
    all_products = []
    for seq in solution:
        all_products.extend(seq)
    
    # Verifica duplicatas
    if len(all_products) != len(set(all_products)):
        print(f"  ⚠️  {name}: DUPLICATAS encontradas!")
        return False
    
    # Verifica produtos faltando
    missing = set(range(n)) - set(all_products)
    if missing:
        print(f"  ⚠️  {name}: {len(missing)} produtos NÃO alocados: {list(missing)[:10]}...")
    
    # Verifica capacidades
    violations = []
    for mach, seq in enumerate(solution):
        if not seq:
            continue
        load = 0.0
        prev = env.initial_state[mach]
        for p in seq:
            rate = env.production_rates[p][mach]
            if rate <= 1e-6:
                print(f"  ⚠️  {name}: Produto {p} inválido para máquina {mach} (rate=0)!")
                return False
            load += env.setup_times[prev][p][mach]
            load += env.demands[p] / rate
            prev = p
        
        cap = env.machine_capacities[mach]
        if load > cap + 1e-6:
            violations.append((mach, load, cap, load - cap))
    
    if violations:
        print(f"  ⚠️  {name}: {len(violations)} VIOLAÇÕES de capacidade!")
        for mach, load, cap, excess in violations:
            print(f"      M{mach}: {load:.2f} > {cap:.2f} (excesso: {excess:.2f})")
        return False
    
    return True

def calc_solution_cost(env, solution):
    """Calcula custo total de uma solução (lista de listas por máquina)"""
    total_cost = 0.0
    for m, seq in enumerate(solution):
        if not seq:
            continue
        prev = env.initial_state[m]
        for p in seq:
            total_cost += env.setup_costs[prev, p, m]
            prev = p
    return total_cost

def calc_load(env, m, seq):
    """Calcula carga (tempo) de uma máquina com sequência"""
    load = 0.0
    prev = env.initial_state[m]
    for p in seq:
        load += env.setup_times[prev, p, m]
        load += env.demands[p] / env.production_rates[p, m]
        prev = p
    return load

# ============================================================================
# HEURÍSTICA 1: Nearest Neighbor (NN)
# ============================================================================
def nearest_neighbor(env):
    """
    Para cada produto (ordem arbitrária), aloca na máquina onde o custo de 
    transição do último produto inserido é menor.
    """
    n, m = env.num_products, env.num_machines
    solution = [[] for _ in range(m)]
    loads = [0.0] * m
    current = list(env.initial_state)  # Último produto de cada máquina
    
    remaining = set(range(n))
    
    while remaining:
        best_cost = float('inf')
        best_p, best_m = -1, -1
        
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
        
        if best_p == -1:
            break
            
        solution[best_m].append(best_p)
        rate = env.production_rates[best_p, best_m]
        loads[best_m] += env.setup_times[current[best_m], best_p, best_m]
        loads[best_m] += env.demands[best_p] / rate
        current[best_m] = best_p
        remaining.remove(best_p)
    
    return solution

# ============================================================================
# HEURÍSTICA 2: Cheapest Insertion (CI) - O que usamos no RKO
# ============================================================================
def cheapest_insertion(env, order=None):
    """
    Para cada produto (na ordem dada), testa TODAS as posições em TODAS as 
    máquinas e insere onde o delta_cost é mínimo.
    """
    n, m = env.num_products, env.num_machines
    solution = [[] for _ in range(m)]
    loads = [0.0] * m
    
    if order is None:
        order = list(range(n))
    
    for p in order:
        best_cost = float('inf')
        best_m, best_pos = -1, -1
        best_time = 0.0
        
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
                time_add = env.setup_times[prev, p, mach]
                cost_rem, time_rem = 0.0, 0.0
                
                if next_p != -1:
                    cost_add += env.setup_costs[p, next_p, mach]
                    cost_rem = env.setup_costs[prev, next_p, mach]
                    time_add += env.setup_times[p, next_p, mach]
                    time_rem = env.setup_times[prev, next_p, mach]
                
                delta_cost = cost_add - cost_rem
                delta_time = time_add + prod_time - time_rem
                
                if loads[mach] + delta_time <= env.machine_capacities[mach]:
                    if delta_cost < best_cost:
                        best_cost = delta_cost
                        best_m, best_pos = mach, pos
                        best_time = delta_time
        
        if best_m != -1:
            solution[best_m].insert(best_pos, p)
            loads[best_m] += best_time
    
    return solution

# ============================================================================
# HEURÍSTICA 3: LPT (Longest Processing Time First)
# ============================================================================
def lpt_heuristic(env):
    """Ordena produtos por demanda decrescente, depois aplica Cheapest Insertion"""
    order = sorted(range(env.num_products), key=lambda p: -env.demands[p])
    return cheapest_insertion(env, order)

# ============================================================================
# HEURÍSTICA 4: SPT (Shortest Processing Time First)
# ============================================================================
def spt_heuristic(env):
    """Ordena produtos por demanda crescente, depois aplica Cheapest Insertion"""
    order = sorted(range(env.num_products), key=lambda p: env.demands[p])
    return cheapest_insertion(env, order)

# ============================================================================
# HEURÍSTICA 5: Hardest First (Maior avg_setup_cost)
# ============================================================================
def hardest_first(env):
    """Ordena produtos por incompatibilidade (avg setup) decrescente"""
    avg_costs = []
    for i in range(env.num_products):
        total = 0.0
        count = 0
        for j in range(env.num_products):
            if i == j:
                continue
            for m in range(env.num_machines):
                total += env.setup_costs[i, j, m] + env.setup_costs[j, i, m]
                count += 2
        avg_costs.append(total / count if count > 0 else 0)
    
    order = sorted(range(env.num_products), key=lambda p: -avg_costs[p])
    return cheapest_insertion(env, order)

# ============================================================================
# HEURÍSTICA 6: GLOBAL CI (Best of All: Product × Machine × Position)
# ============================================================================
def global_ci(env):
    """
    A cada passo, busca o MELHOR (produto, máquina, posição) globalmente.
    Combina a busca de produto do NN com a busca de posição do CI.
    Complexidade: O(n² × m × n) - mais lento, mas deveria ser o melhor greedy.
    """
    n, m = env.num_products, env.num_machines
    solution = [[] for _ in range(m)]
    loads = [0.0] * m
    
    remaining = set(range(n))
    
    while remaining:
        best_cost = float('inf')
        best_p, best_m, best_pos = -1, -1, -1
        best_time = 0.0
        
        # Busca global: testa TODOS os produtos × máquinas × posições
        for p in remaining:
            for mach in range(m):
                rate = env.production_rates[p, mach]
                if rate <= 1e-6:
                    continue
                
                prod_time = env.demands[p] / rate
                seq = solution[mach]
                
                # Testa TODAS as posições (0 a len(seq))
                for pos in range(len(seq) + 1):
                    prev = env.initial_state[mach] if pos == 0 else seq[pos-1]
                    next_p = seq[pos] if pos < len(seq) else -1
                    
                    # Delta custo
                    cost_add = env.setup_costs[prev, p, mach]
                    cost_rem = 0.0
                    if next_p != -1:
                        cost_add += env.setup_costs[p, next_p, mach]
                        cost_rem = env.setup_costs[prev, next_p, mach]
                    delta_cost = cost_add - cost_rem
                    
                    # Delta tempo
                    time_add = env.setup_times[prev, p, mach]
                    time_rem = 0.0
                    if next_p != -1:
                        time_add += env.setup_times[p, next_p, mach]
                        time_rem = env.setup_times[prev, next_p, mach]
                    delta_time = time_add + prod_time - time_rem
                    
                    # Verifica capacidade
                    if loads[mach] + delta_time <= env.machine_capacities[mach]:
                        if delta_cost < best_cost:
                            best_cost = delta_cost
                            best_p, best_m, best_pos = p, mach, pos
                            best_time = delta_time
        
        if best_p == -1:
            break
        
        # Insere na melhor posição
        solution[best_m].insert(best_pos, best_p)
        loads[best_m] += best_time
        remaining.remove(best_p)
    
    n_allocated = sum(len(s) for s in solution)
    return solution

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    instance_path = "400_5_v2.txt"  # Just filename, Sequenciamento adds instances/ folder
    
    print("=" * 60)
    print("TESTE DE HEURÍSTICAS CONSTRUTIVAS")
    print("=" * 60)
    
    env = load_instance(instance_path)
    print(f"Instância: {env.num_products} produtos, {env.num_machines} máquinas")
    print()
    
    heuristics = [
        ("Nearest Neighbor (NN)", nearest_neighbor),
        ("Cheapest Insertion (CI)", cheapest_insertion),
        ("LPT (Longer First)", lpt_heuristic),
        ("SPT (Shorter First)", spt_heuristic),
        ("Hardest First", hardest_first),
        ("Global CI (Best Greedy)", global_ci),
    ]
    
    results = []
    for name, func in heuristics:
        try:
            sol = func(env)
            cost = calc_solution_cost(env, sol)
            n_allocated = sum(len(s) for s in sol)
            
            # Validate solution
            valid = validate_solution(env, sol, name)
            status = "✅" if valid and n_allocated == env.num_products else "❌"
            
            results.append((name, cost, n_allocated, valid))
            print(f"{status} {name:28s} | Custo: {cost:10.2f} | Produtos: {n_allocated}/{env.num_products}")
        except Exception as e:
            print(f"❌ {name:28s} | ERRO: {e}")
    
    print()
    print("-" * 60)
    best = min(results, key=lambda x: x[1])
    print(f"MELHOR: {best[0]} com custo {best[1]:.2f}")
    print()
    print("Literatura (Gurobi): ~572.80")
