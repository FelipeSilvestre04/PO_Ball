"""
Script de Engenharia Reversa para encontrar a seed usada na geração das instâncias v2.

Estratégia:
1. Para cada seed candidata (0 a N):
   a. Gera a instância 100_5_v2.txt com essa seed
   b. Roda o benchmark (com seed fixa para os shuffles internos)
   c. Compara o custo com o esperado (428.41)
   d. Se bater, encontramos a seed!

Custo esperado para 100_5: 428.41 (do benchmark_results.txt)
"""

import numpy as np
import os
import sys
import time

current_directory = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# FUNÇÃO DE GERAÇÃO DE INSTÂNCIA (copiada de instancia.py, com seed)
# ============================================================================
def generate_li_milne_instance_with_seed(num_products, num_reactors, filename, seed):
    """Gera uma instância com uma seed específica."""
    np.random.seed(seed)
    
    # Production Rates
    numerator = np.random.uniform(5, 15, size=(num_products, num_reactors))
    denominator = np.random.uniform(10, 20, size=(num_products, num_reactors))
    p_ir = numerator / denominator
    
    # Demands
    d_i = 10 * np.random.uniform(1, 6, size=num_products)
    
    # Changeover Costs
    c_ijr = np.random.uniform(0, 100, size=(num_products, num_products, num_reactors))
    for i in range(num_products):
        for r in range(num_reactors):
            c_ijr[i][i][r] = 0.0

    # Changeover Times
    t_ijr = np.random.uniform(0, 5, size=(num_products, num_products, num_reactors))
    for i in range(num_products):
        for r in range(num_reactors):
            t_ijr[i][i][r] = 0.0
            
    # Initial Products
    initial_products = np.random.randint(0, num_products, size=num_reactors)
    
    # Capacities (CORREÇÃO v2)
    avg_unit_proc_time = np.mean(1.0 / p_ir, axis=1) 
    avg_prod_time_total = np.sum(d_i * avg_unit_proc_time)
    avg_setup_time_total = num_products * 2.5
    total_time_required = avg_prod_time_total + avg_setup_time_total
    cap_per_reactor = total_time_required / num_reactors
    T_r = np.ones(num_reactors) * cap_per_reactor

    # Escrita do arquivo
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        f.write(f"{num_products} {num_reactors}\n")
        f.write(" ".join(map(str, T_r)) + "\n")
        f.write(" ".join(map(str, initial_products)) + "\n")
        f.write(" ".join(map(str, d_i)) + "\n")
        
        for i in range(num_products):
            f.write(" ".join(map(str, p_ir[i])) + "\n")
            
        for r in range(num_reactors):
            f.write(f"Reactor_Cost_{r}\n")
            for i in range(num_products):
                f.write(" ".join(map(str, c_ijr[i, :, r])) + "\n")
                
        for r in range(num_reactors):
            f.write(f"Reactor_Time_{r}\n")
            for i in range(num_products):
                f.write(" ".join(map(str, t_ijr[i, :, r])) + "\n")


# ============================================================================
# SOLVER SIMPLIFICADO (apenas Passo 1, sem Gurobi, para teste rápido)
# ============================================================================
class QuickSolver:
    def __init__(self, instance_path, shuffle_seed=42):
        self.shuffle_seed = shuffle_seed
        self.read_instance(instance_path)

    def read_instance(self, instance_path):
        with open(instance_path, 'r') as f:
            tokens = f.read().split()
        
        iterator = iter(tokens)
        self.num_products = int(next(iterator))
        self.num_machines = int(next(iterator))
        self.machine_capacities = np.array([float(next(iterator)) for _ in range(self.num_machines)])
        self.initial_state = [int(next(iterator)) for _ in range(self.num_machines)]
        self.demands = np.array([float(next(iterator)) for _ in range(self.num_products)])
        
        self.production_rates = np.zeros((self.num_products, self.num_machines))
        for i in range(self.num_products):
            for r in range(self.num_machines):
                self.production_rates[i][r] = float(next(iterator))
        
        self.setup_costs = np.zeros((self.num_products, self.num_products, self.num_machines))
        for r in range(self.num_machines):
            next(iterator)
            for i in range(self.num_products):
                for j in range(self.num_products):
                    self.setup_costs[i][j][r] = float(next(iterator))
        
        self.setup_times = np.zeros((self.num_products, self.num_products, self.num_machines))
        for r in range(self.num_machines):
            next(iterator)
            for i in range(self.num_products):
                for j in range(self.num_machines):
                    self.setup_times[i][j][r] = float(next(iterator))

    def get_sorting_orders(self):
        products = list(range(self.num_products))
        orders = []
        
        # Random (com seed fixa)
        np.random.seed(self.shuffle_seed)
        p1 = list(products)
        np.random.shuffle(p1)
        orders.append(("Random", p1))
        
        # Demand Asc
        p2 = sorted(products, key=lambda i: self.demands[i])
        orders.append(("Demand Asc", p2))
        
        # Outros... (simplificado)
        return orders

    def step1_constructive(self, sorted_products):
        machine_seqs = [[] for _ in range(self.num_machines)]
        machine_loads = np.zeros(self.num_machines)
        total_cost = 0.0

        for p in sorted_products:
            best_machine = -1
            best_pos = -1
            min_delta_cost = float('inf')
            best_delta_time = 0.0

            for m in range(self.num_machines):
                rate = self.production_rates[p][m]
                if rate <= 1e-6: continue

                proc_time = self.demands[p] / rate
                current_seq = [self.initial_state[m]] + machine_seqs[m]
                
                for k in range(len(current_seq)):
                    prev = current_seq[k]
                    next_p = current_seq[k+1] if k + 1 < len(current_seq) else None

                    cost_add = self.setup_costs[prev][p][m]
                    cost_rem = 0.0
                    if next_p is not None:
                        cost_add += self.setup_costs[p][next_p][m]
                        cost_rem = self.setup_costs[prev][next_p][m]
                    delta_cost = cost_add - cost_rem

                    time_add = self.setup_times[prev][p][m]
                    time_rem = 0.0
                    if next_p is not None:
                        time_add += self.setup_times[p][next_p][m]
                        time_rem = self.setup_times[prev][next_p][m]
                    delta_time = time_add + proc_time - time_rem

                    if machine_loads[m] + delta_time <= self.machine_capacities[m]:
                        if delta_cost < min_delta_cost:
                            min_delta_cost = delta_cost
                            best_machine = m
                            best_pos = k
                            best_delta_time = delta_time

            if best_machine != -1:
                machine_seqs[best_machine].insert(best_pos, p)
                machine_loads[best_machine] += best_delta_time
                total_cost += min_delta_cost
            else:
                total_cost += 1e6

        return machine_seqs, total_cost

    def get_best_constructive_cost(self):
        """Retorna o melhor custo do Passo 1."""
        orders = self.get_sorting_orders()
        best_cost = float('inf')
        
        for name, sorted_prods in orders:
            seqs, cost = self.step1_constructive(sorted_prods)
            if cost < best_cost:
                best_cost = cost
        
        return best_cost


# ============================================================================
# MAIN: BRUTE-FORCE DE SEEDS usando FINGERPRINT
# ============================================================================
# Estratégia: Gerar instância com cada seed e comparar os primeiros valores
# numéricos com os valores das instâncias sem _v2 que temos (se forem diferentes)
# ou rodar o benchmark e comparar a sequência de saída.

def get_instance_fingerprint(filename):
    """Retorna um fingerprint único da instância (primeiros valores numéricos)."""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Fingerprint: capacidade + demandas (valores únicos da instância)
    capacities = [float(x) for x in lines[1].strip().split()]
    demands = [float(x) for x in lines[3].strip().split()]
    
    # Retornar os primeiros 5 valores de cada para comparação rápida
    return tuple(capacities[:3] + demands[:5])


if __name__ == "__main__":
    import sys
    
    output_file = os.path.join(current_directory, 'instances', '100_5_v2.txt')
    
    print("=== Busca de Seed para Instâncias v2 ===")
    print(f"Gerando instâncias e gravando fingerprints...")
    print()
    
    # Primeiro, vamos apenas gerar várias instâncias e mostrar os fingerprints
    # para ver como eles variam
    
    start_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    results_file = os.path.join(current_directory, 'Results', 'seed_search.txt')
    
    with open(results_file, 'w') as f:
        f.write("seed,cap0,cap1,cap2,dem0,dem1,dem2,dem3,dem4\n")
        
        for seed in range(start_seed, start_seed + num_seeds):
            generate_li_milne_instance_with_seed(100, 5, output_file, seed)
            fp = get_instance_fingerprint(output_file)
            
            f.write(f"{seed},{','.join(map(str, fp))}\n")
            
            if seed % 100 == 0:
                print(f"Seed {seed}: capacidade[0]={fp[0]:.2f}, demanda[0]={fp[3]:.2f}")
    
    print(f"\nResultados salvos em: {results_file}")
    print("Próximo passo: Comparar com as instâncias originais (sem v2) que temos")

