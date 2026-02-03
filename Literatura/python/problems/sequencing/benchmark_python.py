"""
Benchmark: Python Best Fit decoder vs C++ 
Testa nas 3 instâncias com 100 itens: 100_5, 100_10, 100_15
60 segundos cada, reporta avaliações/segundo e melhor custo
"""
import numpy as np
import os
import sys
import time

current_directory = os.path.dirname(os.path.abspath(__file__))
python_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, python_directory)
sys.path.insert(0, current_directory)

from sequenciamento import Sequenciamento

TEMPO_TESTE = 60  # segundos
INSTANCIAS = ['100_5_v2.txt', '100_10_v2.txt', '100_15_v2.txt']

def benchmark_instancia(instance_file):
    """Roda benchmark de 60s para uma instância."""
    print(f"\n{'='*50}")
    print(f"Instância: {instance_file}")
    print('='*50)
    
    env = Sequenciamento(instance_file)
    
    best_cost = float('inf')
    evaluations = 0
    
    start = time.time()
    
    while time.time() - start < TEMPO_TESTE:
        # Gera chaves aleatórias
        keys = np.random.rand(env.tam_solution)
        # Decodifica
        solution = env.decoder(keys)
        # Avalia custo
        cost = env.cost(solution)
        
        if cost < best_cost:
            best_cost = cost
        
        evaluations += 1
    
    elapsed = time.time() - start
    evals_per_sec = evaluations / elapsed
    
    print(f"TEMPO TOTAL:      {elapsed:.2f} s")
    print(f"TOTAL AVALIACOES: {evaluations}")
    print(f"VELOCIDADE:       {int(evals_per_sec)} sol/s")
    print(f"MELHOR CUSTO:     {best_cost:.2f}")
    print('-'*50)
    
    return {
        'instance': instance_file,
        'evaluations': evaluations,
        'evals_per_sec': evals_per_sec,
        'best_cost': best_cost,
        'time': elapsed
    }

def main():
    print("\n" + "="*60)
    print("BENCHMARK PYTHON - Best Fit Decoder")
    print(f"Tempo por instância: {TEMPO_TESTE} segundos")
    print("="*60)
    
    results = []
    
    for instance in INSTANCIAS:
        result = benchmark_instancia(instance)
        results.append(result)
    
    # Resumo
    print("\n" + "="*60)
    print("RESUMO PYTHON")
    print("="*60)
    print(f"{'Instância':<20} {'Sol/s':>10} {'Melhor Custo':>15}")
    print("-"*45)
    for r in results:
        print(f"{r['instance']:<20} {int(r['evals_per_sec']):>10} {r['best_cost']:>15.2f}")
    
    # Salvar em arquivo
    output_file = os.path.join(current_directory, 'Results', 'benchmark_python.txt')
    with open(output_file, 'w') as f:
        f.write("BENCHMARK PYTHON - Best Fit Decoder\n")
        f.write(f"Tempo: {TEMPO_TESTE}s por instância\n")
        f.write("="*50 + "\n")
        for r in results:
            f.write(f"{r['instance']}: {int(r['evals_per_sec'])} sol/s, melhor={r['best_cost']:.2f}\n")
    
    print(f"\n✅ Resultados salvos em: {output_file}")

if __name__ == "__main__":
    main()
