"""
Busca de seed para recuperar instâncias v2 originais.
Estratégia: Gerar instância 100_5_v2, rodar benchmark, comparar custo com 428.41
"""
import numpy as np
import os
import sys
import time

current_directory = os.path.dirname(os.path.abspath(__file__))

# Importar o benchmark
sys.path.insert(0, current_directory)
from benchmark import LiMilneSolver

# ============================================================================
# FUNÇÃO DE GERAÇÃO DE INSTÂNCIA
# ============================================================================
def generate_instance_with_seed(num_products, num_reactors, filename, seed):
    """Gera uma instância com uma seed específica."""
    np.random.seed(seed)
    
    numerator = np.random.uniform(5, 15, size=(num_products, num_reactors))
    denominator = np.random.uniform(10, 20, size=(num_products, num_reactors))
    p_ir = numerator / denominator
    
    d_i = 10 * np.random.uniform(1, 6, size=num_products)
    
    c_ijr = np.random.uniform(0, 100, size=(num_products, num_products, num_reactors))
    for i in range(num_products):
        for r in range(num_reactors):
            c_ijr[i][i][r] = 0.0

    t_ijr = np.random.uniform(0, 5, size=(num_products, num_products, num_reactors))
    for i in range(num_products):
        for r in range(num_reactors):
            t_ijr[i][i][r] = 0.0
            
    initial_products = np.random.randint(0, num_products, size=num_reactors)
    
    avg_unit_proc_time = np.mean(1.0 / p_ir, axis=1) 
    avg_prod_time_total = np.sum(d_i * avg_unit_proc_time)
    avg_setup_time_total = num_products * 2.5
    total_time_required = avg_prod_time_total + avg_setup_time_total
    cap_per_reactor = total_time_required / num_reactors
    T_r = np.ones(num_reactors) * cap_per_reactor

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
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Custo alvo do benchmark_results.txt
    TARGET_COST = 428.41
    TOLERANCE = 0.01  # Tolerância para comparação de floats
    
    instance_file = os.path.join(current_directory, 'instances', '100_5_v2.txt')
    results_file = os.path.join(current_directory, 'Results', 'seed_search_results.txt')
    
    # Range de seeds a testar
    start_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    print(f"=== Busca de Seed para Instâncias v2 ===")
    print(f"Custo alvo: {TARGET_COST}")
    print(f"Testando seeds {start_seed} a {start_seed + num_seeds - 1}")
    print(f"Arquivo: {instance_file}")
    print()
    
    found_seeds = []
    
    with open(results_file, 'a') as f:
        f.write(f"\n--- Busca iniciada: seeds {start_seed} a {start_seed + num_seeds - 1} ---\n")
        
        for seed in range(start_seed, start_seed + num_seeds):
            try:
                # Gerar instância
                generate_instance_with_seed(100, 5, instance_file, seed)
                
                # Rodar benchmark (silencioso)
                import io
                from contextlib import redirect_stdout
                
                with redirect_stdout(io.StringIO()):
                    solver = LiMilneSolver(instance_file, time_limit_gurobi=30)
                    seqs, cost = solver.solve()
                
                # Comparar custo
                diff = abs(cost - TARGET_COST)
                
                if diff < TOLERANCE:
                    print(f"*** ENCONTRADO! Seed {seed}: Custo = {cost:.2f} ***")
                    f.write(f"MATCH! Seed {seed}: Custo = {cost:.2f}\n")
                    found_seeds.append(seed)
                elif diff < 5.0:  # Diferença pequena, pode ser interessante
                    print(f"Seed {seed}: Custo = {cost:.2f} (diff = {diff:.2f})")
                    f.write(f"CLOSE: Seed {seed}: Custo = {cost:.2f} (diff = {diff:.2f})\n")
                
                if seed % 10 == 0:
                    print(f"Seed {seed}: Custo = {cost:.2f}")
                    
            except Exception as e:
                print(f"Seed {seed}: ERRO - {e}")
                continue
    
    print()
    if found_seeds:
        print(f"=== Seeds encontradas: {found_seeds} ===")
    else:
        print(f"Nenhuma seed encontrada no range {start_seed}-{start_seed + num_seeds - 1}")
        print(f"Tente aumentar o range ou verificar se o custo alvo está correto.")
