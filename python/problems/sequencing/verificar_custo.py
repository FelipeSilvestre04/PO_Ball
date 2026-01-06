"""
Script para verificar se a função objetivo está sendo calculada corretamente.
Compara o custo calculado pelo decoder com o recálculo manual.
"""
import numpy as np
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
python_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, python_directory)
sys.path.insert(0, current_directory)

from sequenciamento import Sequenciamento
from sequenciamento_insercao import SequenciamentoInsercao

def recalcular_custo(env, sequences):
    """
    Recalcula o custo total de uma solução do ZERO.
    Igual à função _calculate_machine_cost do benchmark.py.
    """
    total_cost = 0.0
    
    for m in range(env.num_machines):
        seq = sequences[m]
        if not seq:
            continue
        
        # Começa do estado inicial
        prev = env.initial_state[m]
        for p in seq:
            total_cost += env.setup_costs[prev][p][m]
            prev = p
    
    return total_cost

def testar_decoder(env_class, nome, instance='100_5_v2.txt'):
    """
    Testa se o custo retornado pelo decoder bate com o recálculo.
    """
    print(f"\n{'='*60}")
    print(f"TESTE: {nome}")
    print('='*60)
    
    env = env_class(instance)
    
    # Gerar uma solução
    np.random.seed(42)  # Seed fixa para reprodutibilidade
    keys = np.random.rand(env.tam_solution)
    solution = env.decoder(keys)
    
    # Custo retornado pelo decoder
    custo_decoder = env.cost(solution)
    
    # Agora precisamos pegar as sequências geradas
    # Vamos rodar o cost de novo e capturar as sequências
    sorted_products = solution
    machine_sequences = [[] for _ in range(env.num_machines)]
    machine_loads = np.zeros(env.num_machines)
    total_delta_cost = 0.0
    penalty = 0.0
    
    for product_idx in sorted_products:
        best_machine = -1
        best_position = -1
        min_delta_cost = float('inf')
        best_delta_time = 0.0
        
        for m in range(env.num_machines):
            rate = env.production_rates[product_idx][m]
            if rate <= 1e-6:
                continue
            
            prod_time = env.demands[product_idx] / rate
            current_seq = machine_sequences[m]
            
            for pos in range(len(current_seq) + 1):
                if pos == 0:
                    prev_prod = env.initial_state[m]
                else:
                    prev_prod = current_seq[pos - 1]
                
                next_prod = current_seq[pos] if pos < len(current_seq) else None
                
                cost_add = env.setup_costs[prev_prod][product_idx][m]
                cost_rem = 0.0
                if next_prod is not None:
                    cost_add += env.setup_costs[product_idx][next_prod][m]
                    cost_rem = env.setup_costs[prev_prod][next_prod][m]
                delta_cost = cost_add - cost_rem
                
                time_add = env.setup_times[prev_prod][product_idx][m]
                time_rem = 0.0
                if next_prod is not None:
                    time_add += env.setup_times[product_idx][next_prod][m]
                    time_rem = env.setup_times[prev_prod][next_prod][m]
                delta_time = time_add + prod_time - time_rem
                
                if machine_loads[m] + delta_time <= env.machine_capacities[m]:
                    if delta_cost < min_delta_cost:
                        min_delta_cost = delta_cost
                        best_machine = m
                        best_position = pos
                        best_delta_time = delta_time
                    elif abs(delta_cost - min_delta_cost) < 1e-6:
                        if best_machine != -1:
                            curr_rate = env.production_rates[product_idx][best_machine]
                            if rate > curr_rate:
                                best_machine = m
                                best_position = pos
                                best_delta_time = delta_time
        
        if best_machine != -1:
            machine_sequences[best_machine].insert(best_position, product_idx)
            machine_loads[best_machine] += best_delta_time
            total_delta_cost += min_delta_cost
        else:
            penalty += 100000 + (env.demands[product_idx] * 1000)
    
    # Recalcular custo do zero
    custo_recalculado = recalcular_custo(env, machine_sequences)
    
    print(f"\n📊 Resultados:")
    print(f"  Custo (soma de deltas):  {total_delta_cost:.4f}")
    print(f"  Custo (recálculo total): {custo_recalculado:.4f}")
    print(f"  Custo retornado pelo cost(): {custo_decoder:.4f}")
    print(f"  Penalidade: {penalty:.4f}")
    
    diferenca = abs(total_delta_cost - custo_recalculado)
    print(f"\n🔍 Diferença (soma deltas vs recálculo): {diferenca:.6f}")
    
    if diferenca < 0.01:
        print("✅ OK - Custos batem!")
    else:
        print("❌ ERRO - Custos diferentes!")
        print("\nSequências geradas:")
        for m, seq in enumerate(machine_sequences):
            print(f"  Máq {m}: {seq[:10]}..." if len(seq) > 10 else f"  Máq {m}: {seq}")
    
    return diferenca < 0.01

if __name__ == "__main__":
    print("\n" + "="*60)
    print("VERIFICAÇÃO DA FUNÇÃO OBJETIVO")
    print("="*60)
    
    # Testar o decoder original (sequenciamento.py)
    ok1 = testar_decoder(Sequenciamento, "Sequenciamento (Best Fit)")
    
    # Testar o decoder com 2-opt (se existir)
    try:
        ok2 = testar_decoder(SequenciamentoInsercao, "SequenciamentoInsercao (Best Fit + 2-opt)")
    except Exception as e:
        print(f"\n⚠️ Erro ao testar SequenciamentoInsercao: {e}")
        ok2 = False
    
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    print(f"  Sequenciamento: {'✅ OK' if ok1 else '❌ FALHOU'}")
    print(f"  SequenciamentoInsercao: {'✅ OK' if ok2 else '❌ FALHOU'}")
