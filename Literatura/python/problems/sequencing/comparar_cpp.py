"""
Script para comparar a lógica Python vs C++
Testa uma permutação FIXA para garantir que a função objetivo é igual
"""
import numpy as np
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
python_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, python_directory)
sys.path.insert(0, current_directory)

from sequenciamento import Sequenciamento

def main():
    print("=" * 60)
    print("VERIFICAÇÃO: Python vs C++ (mesma permutação)")
    print("=" * 60)
    
    instance = '100_5_v2.txt'
    env = Sequenciamento(instance)
    
    # Usar uma permutação fixa (0, 1, 2, ..., n-1) - ordem natural
    permutacao_natural = np.arange(env.tam_solution)
    custo_natural = env.cost(permutacao_natural)
    print(f"\n1. Permutação natural [0, 1, 2, ..., n-1]:")
    print(f"   Custo: {custo_natural:.4f}")
    
    # Permutação reversa (n-1, n-2, ..., 0)
    permutacao_reversa = np.arange(env.tam_solution)[::-1]
    custo_reversa = env.cost(permutacao_reversa)
    print(f"\n2. Permutação reversa [n-1, n-2, ..., 0]:")
    print(f"   Custo: {custo_reversa:.4f}")
    
    # Permutação aleatória com seed fixa
    np.random.seed(42)
    chaves = np.random.rand(env.tam_solution)
    permutacao = env.decoder(chaves)
    custo = env.cost(permutacao)
    print(f"\n3. Permutação aleatória (seed=42):")
    print(f"   Custo: {custo:.4f}")
    print(f"   Primeiros 10 índices: {permutacao[:10].tolist()}")
    
    # Imprimir parâmetros para verificação
    print("\n" + "=" * 60)
    print("PARÂMETROS DA INSTÂNCIA")
    print("=" * 60)
    print(f"Produtos: {env.num_products}")
    print(f"Máquinas: {env.num_machines}")
    print(f"Capacidades: {env.machine_capacities}")
    print(f"Estado inicial: {env.initial_state}")
    print(f"\nsetup_costs[0][1][0] = {env.setup_costs[0][1][0]:.4f}")
    print(f"setup_costs[1][0][0] = {env.setup_costs[1][0][0]:.4f}")
    print(f"setup_times[0][1][0] = {env.setup_times[0][1][0]:.4f}")
    print(f"production_rates[0][0] = {env.production_rates[0][0]:.4f}")
    print(f"demands[0] = {env.demands[0]:.4f}")
    
    # Verificação detalhada: reconstruir a solução passo a passo
    print("\n" + "=" * 60)
    print("TRACE: Primeiras 5 alocações (permutação natural)")
    print("=" * 60)
    
    machine_sequences = [[] for _ in range(env.num_machines)]
    machine_loads = np.zeros(env.num_machines)
    total_cost = 0.0
    
    for i, prod in enumerate(permutacao_natural[:5]):
        best_m = -1
        best_pos = -1
        min_delta = float('inf')
        best_dt = 0.0
        
        for m in range(env.num_machines):
            rate = env.production_rates[prod][m]
            if rate <= 1e-6:
                continue
            prod_time = env.demands[prod] / rate
            seq = machine_sequences[m]
            
            for pos in range(len(seq) + 1):
                prev = env.initial_state[m] if pos == 0 else seq[pos-1]
                next_p = seq[pos] if pos < len(seq) else None
                
                delta_c = env.setup_costs[prev][prod][m]
                delta_t = env.setup_times[prev][prod][m] + prod_time
                if next_p is not None:
                    delta_c += env.setup_costs[prod][next_p][m] - env.setup_costs[prev][next_p][m]
                    delta_t += env.setup_times[prod][next_p][m] - env.setup_times[prev][next_p][m]
                
                if machine_loads[m] + delta_t <= env.machine_capacities[m]:
                    if delta_c < min_delta:
                        min_delta = delta_c
                        best_m = m
                        best_pos = pos
                        best_dt = delta_t
        
        if best_m != -1:
            machine_sequences[best_m].insert(best_pos, prod)
            machine_loads[best_m] += best_dt
            total_cost += min_delta
            print(f"  Prod {prod:2d} -> Máq {best_m}, Pos {best_pos}, delta={min_delta:.4f}, load={machine_loads[best_m]:.2f}")
        else:
            print(f"  Prod {prod:2d} -> PENALIDADE!")
    
    print(f"\nCusto parcial (5 prods): {total_cost:.4f}")

if __name__ == "__main__":
    main()
