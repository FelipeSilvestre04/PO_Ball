"""
Script para executar o benchmark C++ nas instâncias de sequenciamento.
Executa começando pelas instâncias MAIORES e com tempo de 1200 segundos cada.
"""
import subprocess
import os
import time
from datetime import datetime

# Configurações
TEMPO_POR_INSTANCIA = 1200  # 1200 segundos = 20 minutos

# Diretórios (ajuste conforme necessário)
current_directory = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_directory)))

# Caminho para o executável C++
CPP_EXE = os.path.join(project_root, 'cpp', 'Program', 'teste_decoder.exe')

# Caminho para as instâncias (ajuste para o caminho correto no seu sistema)
INSTANCES_DIR = os.path.join(current_directory, 'instances')

# Arquivo de resultados
RESULTS_FILE = os.path.join(current_directory, 'Results', 'benchmark_cpp_1200s.txt')

# Lista de instâncias ordenadas do MAIOR para o MENOR
# Formato: (num_produtos, num_maquinas)
INSTANCIAS = []
for num_produtos in [700, 600, 500, 400, 300, 200, 100]:  # Maiores primeiro
    for num_maquinas in [15, 10, 5]:  # Mais máquinas primeiro
        INSTANCIAS.append(f"{num_produtos}_{num_maquinas}_v2.txt")


def run_benchmark():
    """Executa o benchmark para todas as instâncias"""
    
    # Verifica se o executável existe
    if not os.path.exists(CPP_EXE):
        print(f"ERRO: Executável não encontrado: {CPP_EXE}")
        print("Por favor, compile o código C++ primeiro com:")
        print("  cd cpp/Program")
        print("  g++ -O3 -o teste_decoder.exe teste_decoder.cpp")
        return
    
    # Verifica se o diretório de instâncias existe
    if not os.path.exists(INSTANCES_DIR):
        print(f"AVISO: Diretório de instâncias não encontrado: {INSTANCES_DIR}")
        print("Por favor, ajuste a variável INSTANCES_DIR no script.")
        return
    
    # Cria diretório de resultados se não existir
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    print("=" * 70)
    print("BENCHMARK C++ - INSTÂNCIAS DE SEQUENCIAMENTO")
    print(f"Tempo por instância: {TEMPO_POR_INSTANCIA}s ({TEMPO_POR_INSTANCIA/60:.1f} min)")
    print(f"Total de instâncias: {len(INSTANCIAS)}")
    print(f"Tempo total estimado: {len(INSTANCIAS) * TEMPO_POR_INSTANCIA / 3600:.1f} horas")
    print("=" * 70)
    
    # Cabeçalho do arquivo de resultados
    with open(RESULTS_FILE, 'w') as f:
        f.write(f"# Benchmark C++ - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Tempo por instância: {TEMPO_POR_INSTANCIA}s\n")
        f.write("# Formato: instancia, melhor_custo, avaliações, tempo_real, velocidade(sol/s)\n")
        f.write("-" * 70 + "\n")
    
    for i, instance_name in enumerate(INSTANCIAS):
        instance_path = os.path.join(INSTANCES_DIR, instance_name)
        
        print(f"\n[{i+1}/{len(INSTANCIAS)}] Executando: {instance_name}")
        print(f"     Início: {datetime.now().strftime('%H:%M:%S')}")
        
        # Verifica se a instância existe
        if not os.path.exists(instance_path):
            print(f"     AVISO: Arquivo não encontrado, pulando...")
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{instance_name}, ARQUIVO_NAO_ENCONTRADO\n")
            continue
        
        # Executa o benchmark C++
        start_time = time.time()
        try:
            result = subprocess.run(
                [CPP_EXE, instance_path, str(TEMPO_POR_INSTANCIA)],
                capture_output=True,
                text=True,
                timeout=TEMPO_POR_INSTANCIA + 120  # Timeout com margem de 2 min
            )
            elapsed = time.time() - start_time
            
            # Parseia a saída
            output = result.stdout
            best_cost = "N/A"
            evaluations = "N/A"
            speed = "N/A"
            
            for line in output.split('\n'):
                if "MELHOR CUSTO:" in line:
                    best_cost = line.split(':')[1].strip()
                elif "TOTAL AVALIACOES:" in line:
                    evaluations = line.split(':')[1].strip()
                elif "VELOCIDADE:" in line:
                    speed = line.split(':')[1].strip()
            
            print(f"     Melhor custo: {best_cost}")
            print(f"     Avaliações: {evaluations}")
            print(f"     Velocidade: {speed}")
            
            # Salva resultados
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{instance_name}, {best_cost}, {evaluations}, {elapsed:.2f}s, {speed}\n")
                
        except subprocess.TimeoutExpired:
            print(f"     TIMEOUT após {TEMPO_POR_INSTANCIA + 120}s")
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{instance_name}, TIMEOUT\n")
                
        except Exception as e:
            print(f"     ERRO: {e}")
            with open(RESULTS_FILE, 'a') as f:
                f.write(f"{instance_name}, ERRO: {e}\n")
    
    print("\n" + "=" * 70)
    print("BENCHMARK CONCLUÍDO!")
    print(f"Resultados salvos em: {RESULTS_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
