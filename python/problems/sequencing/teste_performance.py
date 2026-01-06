"""
Teste de Performance dos Decoders (60 segundos cada)
=====================================================
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

from sequenciamento import Sequenciamento
from sequenciamento_hibrido import SequenciamentoHibrido
from sequenciamento_insercao import SequenciamentoInsercao

TEMPO_TESTE = 60  # segundos
INSTANCIA = '100_5_v2.txt'

def testar_decoder(env_class, nome, **kwargs):
    """Roda o decoder por TEMPO_TESTE segundos e coleta estatísticas."""
    print(f"\n{'='*50}")
    print(f"Testando: {nome}")
    print(f"{'='*50}")
    
    env = env_class(INSTANCIA, **kwargs) if kwargs else env_class(INSTANCIA)
    
    custos = []
    start = time.time()
    
    while time.time() - start < TEMPO_TESTE:
        keys = np.random.rand(env.tam_solution)
        sol = env.decoder(keys)
        custo = env.cost(sol)
        custos.append(custo)
    
    elapsed = time.time() - start
    
    resultados = {
        'nome': nome,
        'solucoes': len(custos),
        'tempo': elapsed,
        'sol_por_seg': len(custos) / elapsed,
        'melhor': min(custos),
        'pior': max(custos),
        'media': np.mean(custos),
        'std': np.std(custos)
    }
    
    print(f"  Soluções avaliadas: {resultados['solucoes']}")
    print(f"  Taxa: {resultados['sol_por_seg']:.1f} sol/s")
    print(f"  Melhor custo: {resultados['melhor']:.2f}")
    print(f"  Custo médio: {resultados['media']:.2f}")
    
    return resultados

def main():
    print("\n" + "="*60)
    print(f"TESTE DE PERFORMANCE DOS DECODERS")
    print(f"Instância: {INSTANCIA}")
    print(f"Tempo por decoder: {TEMPO_TESTE} segundos")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    resultados = []
    
    # 1. Best Fit
    r1 = testar_decoder(Sequenciamento, "Best Fit (original)")
    resultados.append(r1)
    
    # 2. Best Fit + Gurobi TSP
    r2 = testar_decoder(SequenciamentoHibrido, "Best Fit + Gurobi TSP", time_limit_tsp=1.0)
    resultados.append(r2)
    
    # 3. Best Fit + 2-opt one-pass
    r3 = testar_decoder(SequenciamentoInsercao, "Best Fit + 2-opt (1 pass)")
    resultados.append(r3)
    
    # Salvar resultados
    output_file = os.path.join(current_directory, 'Results', 'performance_decoders.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("TESTE DE PERFORMANCE DOS DECODERS\n")
        f.write("="*70 + "\n")
        f.write(f"Instância: {INSTANCIA}\n")
        f.write(f"Tempo por decoder: {TEMPO_TESTE} segundos\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Decoder':<30} {'Soluções':>10} {'Sol/s':>10} {'Melhor':>12} {'Média':>12} {'Std':>10}\n")
        f.write("-"*84 + "\n")
        
        for r in resultados:
            f.write(f"{r['nome']:<30} {r['solucoes']:>10} {r['sol_por_seg']:>10.1f} {r['melhor']:>12.2f} {r['media']:>12.2f} {r['std']:>10.2f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("ANÁLISE:\n")
        f.write("-"*70 + "\n")
        
        melhor_qualidade = min(resultados, key=lambda x: x['melhor'])
        mais_rapido = max(resultados, key=lambda x: x['sol_por_seg'])
        
        f.write(f"Melhor qualidade: {melhor_qualidade['nome']} (custo={melhor_qualidade['melhor']:.2f})\n")
        f.write(f"Mais rápido: {mais_rapido['nome']} ({mais_rapido['sol_por_seg']:.1f} sol/s)\n")
    
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    print(f"{'Decoder':<30} {'Sol/s':>10} {'Melhor':>12}")
    print("-"*52)
    for r in resultados:
        print(f"{r['nome']:<30} {r['sol_por_seg']:>10.1f} {r['melhor']:>12.2f}")
    
    print(f"\n✅ Resultados salvos em: {output_file}")

if __name__ == "__main__":
    main()
