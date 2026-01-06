"""
Teste de Performance do RKO com diferentes Decoders (1 minuto cada)
===================================================================
1. Best Fit (sequenciamento.py)
2. Best Fit + Gurobi TSP (sequenciamento_hibrido.py)
3. Best Fit + 2-opt one-pass (sequenciamento_insercao.py)
"""
import numpy as np
import os
import sys
import time
from datetime import datetime

current_directory = os.path.dirname(os.path.abspath(__file__))
python_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, python_directory)
sys.path.insert(0, current_directory)

from rko.rko import RKO
from sequenciamento import Sequenciamento
from sequenciamento_hibrido import SequenciamentoHibrido
from sequenciamento_insercao import SequenciamentoInsercao

TEMPO_TESTE = 60  # segundos
INSTANCIA = '100_5_v2.txt'
RUNS = 1  # 1 run por decoder para teste rápido

def testar_rko(env_class, nome, **kwargs):
    """Roda o RKO por TEMPO_TESTE segundos e retorna o melhor resultado."""
    print(f"\n{'='*50}")
    print(f"Testando RKO com: {nome}")
    print(f"{'='*50}")
    
    env = env_class(INSTANCIA, **kwargs) if kwargs else env_class(INSTANCIA)
    
    # Configurar RKO
    results_dir = os.path.join(current_directory, 'Results')
    solver = RKO(env, True)  # False = sem Q-learning report
    
    start = time.time()
    
    # Rodar RKO
    final_cost, final_solution, time_to_best = solver.solve(
        time_total=TEMPO_TESTE,
        brkga=1, lns=1, vns=1, ils=1, sa=1, pso=1, ga=1, ms=1,
        runs=RUNS
    )
    
    elapsed = time.time() - start
    
    resultados = {
        'nome': nome,
        'melhor_custo': final_cost,
        'tempo_para_melhor': time_to_best,
        'tempo_total': elapsed
    }
    
    print(f"  Melhor custo: {resultados['melhor_custo']:.2f}")
    print(f"  Tempo para melhor: {resultados['tempo_para_melhor']:.2f}s")
    
    return resultados

def main():
    print("\n" + "="*60)
    print(f"TESTE RKO COM DIFERENTES DECODERS")
    print(f"Instância: {INSTANCIA}")
    print(f"Tempo por decoder: {TEMPO_TESTE} segundos")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    resultados = []
    
    # 1. Best Fit
    r1 = testar_rko(Sequenciamento, "Best Fit (original)")
    resultados.append(r1)
    
    # 2. Best Fit + Gurobi TSP (pode ser lento!)
    r2 = testar_rko(SequenciamentoHibrido, "Best Fit + Gurobi TSP", time_limit_tsp=1.0)
    resultados.append(r2)
    
    # 3. Best Fit + 2-opt one-pass
    r3 = testar_rko(SequenciamentoInsercao, "Best Fit + 2-opt (1 pass)")
    resultados.append(r3)
    
    # Salvar resultados
    output_file = os.path.join(current_directory, 'Results', 'rko_decoders.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TESTE RKO COM DIFERENTES DECODERS\n")
        f.write("="*70 + "\n")
        f.write(f"Instância: {INSTANCIA}\n")
        f.write(f"Tempo por decoder: {TEMPO_TESTE} segundos\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Decoder':<30} {'Melhor Custo':>15} {'Tempo p/ Melhor':>18}\n")
        f.write("-"*63 + "\n")
        
        for r in resultados:
            f.write(f"{r['nome']:<30} {r['melhor_custo']:>15.2f} {r['tempo_para_melhor']:>15.2f}s\n")
        
        f.write("\n" + "="*70 + "\n")
        
        melhor = min(resultados, key=lambda x: x['melhor_custo'])
        f.write(f"MELHOR: {melhor['nome']} (custo={melhor['melhor_custo']:.2f})\n")
    
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    print(f"{'Decoder':<30} {'Melhor Custo':>15}")
    print("-"*45)
    for r in resultados:
        print(f"{r['nome']:<30} {r['melhor_custo']:>15.2f}")
    
    print(f"\n✅ Resultados salvos em: {output_file}")

if __name__ == "__main__":
    main()
