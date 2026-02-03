"""
Script para executar o RKO C++ por RODADAS.
Em vez de executar todas as runs de uma instância antes de passar para a próxima,
executa uma run de cada instância por rodada:
  - Round 1: Run 1 de todas as instâncias
  - Round 2: Run 2 de todas as instâncias
  - ... e assim por diante

Resultados são salvos em:
  - Results/Results_Run_1.csv, Results_Run_2.csv, etc.
  - Results/Results_All_Runs.csv (consolidado)
"""
import subprocess
import os
import time
from datetime import datetime

# ===================== CONFIGURAÇÕES =====================
TEMPO_POR_INSTANCIA = 150   # segundos por instância
NUM_RUNS = 5                # número de rodadas

# Diretórios - Usando caminhos WSL
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # PO_Ball

# Diretórios do C++
PROGRAM_DIR = os.path.join(SCRIPT_DIR, 'Program')
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'Results')

# Caminho para as instâncias
INSTANCES_DIR = os.path.join(PROJECT_ROOT, 'python', 'problems', 'sequencing', 'instances')

# Lista de instâncias (ordenadas do menor para o maior)
INSTANCIAS = []
for num_produtos in [100, 200, 300, 400, 500, 600, 700]:
    for num_maquinas in [5, 10, 15]:
        INSTANCIAS.append(f"{num_produtos}_{num_maquinas}_v2.txt")

# =========================================================

def find_rko_executable():
    """Encontra o executável do RKO"""
    # Tenta diferentes nomes possíveis (ordem de preferência)
    possible_names = ['runTest', 'runTest.exe', 'RKO', 'RKO.exe', 'main', 'main.exe']
    
    for name in possible_names:
        path = os.path.join(PROGRAM_DIR, name)
        if os.path.exists(path):
            return path
    
    # Se não encontrar, retorna o nome padrão
    return os.path.join(PROGRAM_DIR, 'runTest')


def update_config_maxruns(maxruns=1):
    """Atualiza o MAXRUNS no arquivo de configuração para 1"""
    config_path = os.path.join(PROGRAM_DIR, 'config', 'config_tests.conf')
    
    if not os.path.exists(config_path):
        print(f"AVISO: Arquivo de configuração não encontrado: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    with open(config_path, 'w') as f:
        for line in lines:
            if line.strip().startswith('MAXRUNS'):
                f.write(f'MAXRUNS {maxruns}\n')
            else:
                f.write(line)
    
    return True


def run_rko_single(exe_path, instance_path, timeout):
    """Executa o RKO para uma única instância"""
    try:
        result = subprocess.run(
            [exe_path, instance_path, str(timeout)],
            capture_output=True,
            text=True,
            timeout=timeout + 60,  # Margem de 1 minuto
            cwd=PROGRAM_DIR
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "TIMEOUT"
    except Exception as e:
        return False, "", str(e)


def parse_rko_csv_last_line(csv_path):
    """Lê a última linha do Results_RKO.csv e extrai os dados"""
    if not os.path.exists(csv_path):
        return None
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return None
    
    # Pega a última linha não vazia
    last_line = None
    for line in reversed(lines):
        if line.strip():
            last_line = line.strip()
            break
    
    return last_line


def clear_rko_results():
    """Limpa o arquivo Results_RKO.csv para iniciar uma nova run"""
    csv_path = os.path.join(RESULTS_DIR, 'Results_RKO.csv')
    if os.path.exists(csv_path):
        # Faz backup
        backup_path = os.path.join(RESULTS_DIR, 'Results_RKO_backup.csv')
        os.rename(csv_path, backup_path)


def run_benchmark_by_rounds():
    """Executa o benchmark por rodadas"""
    
    # Encontra o executável
    exe_path = find_rko_executable()
    if not os.path.exists(exe_path):
        print(f"ERRO: Executável RKO não encontrado: {exe_path}")
        print("Por favor, compile o código C++ primeiro.")
        return
    
    # Verifica se as instâncias existem
    if not os.path.exists(INSTANCES_DIR):
        print(f"ERRO: Diretório de instâncias não encontrado: {INSTANCES_DIR}")
        return
    
    # Cria diretório de resultados se não existir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Atualiza o MAXRUNS para 1
    print("Configurando MAXRUNS=1...")
    update_config_maxruns(1)
    
    # Calcula tempo total estimado
    total_instances = len(INSTANCIAS)
    total_executions = total_instances * NUM_RUNS
    total_time_hours = (total_executions * TEMPO_POR_INSTANCIA) / 3600
    
    print("=" * 70)
    print("RKO BENCHMARK - EXECUÇÃO POR RODADAS")
    print("=" * 70)
    print(f"Instâncias: {total_instances}")
    print(f"Rodadas (runs): {NUM_RUNS}")
    print(f"Tempo por instância: {TEMPO_POR_INSTANCIA}s")
    print(f"Total de execuções: {total_executions}")
    print(f"Tempo estimado: {total_time_hours:.1f} horas")
    print("=" * 70)
    print(f"\nResultados serão salvos em: {RESULTS_DIR}")
    print("  - Results_Run_1.csv, Results_Run_2.csv, ...")
    print("  - Results_All_Runs.csv (consolidado)")
    print("=" * 70)
    
    # Arquivo consolidado
    all_runs_file = os.path.join(RESULTS_DIR, 'Results_All_Runs.csv')
    with open(all_runs_file, 'w') as f:
        f.write("instance,run,ofv,time_best,time_total\n")
    
    # Executa por rodadas
    for run in range(1, NUM_RUNS + 1):
        print(f"\n{'='*70}")
        print(f"RODADA {run}/{NUM_RUNS}")
        print(f"{'='*70}")
        
        # Arquivo de resultados desta rodada
        run_file = os.path.join(RESULTS_DIR, f'Results_Run_{run}.csv')
        with open(run_file, 'w') as f:
            f.write(f"# Rodada {run} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("instance,ofv,time_best,time_total\n")
        
        for i, instance_name in enumerate(INSTANCIAS):
            instance_path = os.path.join(INSTANCES_DIR, instance_name)
            
            print(f"\n  [{i+1}/{total_instances}] {instance_name}")
            print(f"      Início: {datetime.now().strftime('%H:%M:%S')}")
            
            if not os.path.exists(instance_path):
                print(f"      AVISO: Arquivo não encontrado, pulando...")
                continue
            
            # Limpa resultados anteriores do RKO
            rko_csv = os.path.join(RESULTS_DIR, 'Results_RKO.csv')
            if os.path.exists(rko_csv):
                os.remove(rko_csv)
            
            # Executa o RKO
            start = time.time()
            success, stdout, stderr = run_rko_single(exe_path, instance_path, TEMPO_POR_INSTANCIA)
            elapsed = time.time() - start
            
            if success:
                # Lê o resultado do arquivo CSV gerado
                result_line = parse_rko_csv_last_line(rko_csv)
                
                if result_line:
                    # Parse do resultado (formato: instance\tmetaheuristics\tnum_runs\tofv1\t...\tbest\tavg\ttime_best\ttime_total)
                    parts = result_line.split('\t')
                    if len(parts) >= 5:
                        ofv = parts[3]  # Primeiro OFV (única run)
                        time_best = parts[-2] if len(parts) > 2 else "N/A"
                        time_total = parts[-1] if len(parts) > 1 else "N/A"
                    else:
                        ofv = "PARSE_ERROR"
                        time_best = "N/A"
                        time_total = "N/A"
                else:
                    ofv = "NO_RESULT"
                    time_best = "N/A"
                    time_total = "N/A"
                
                print(f"      OFV: {ofv}")
                print(f"      Tempo: {elapsed:.1f}s")
                
                # Salva no arquivo da rodada
                with open(run_file, 'a') as f:
                    f.write(f"{instance_name},{ofv},{time_best},{time_total}\n")
                
                # Salva no arquivo consolidado
                with open(all_runs_file, 'a') as f:
                    f.write(f"{instance_name},{run},{ofv},{time_best},{time_total}\n")
            else:
                print(f"      ERRO: {stderr}")
                with open(run_file, 'a') as f:
                    f.write(f"{instance_name},ERROR,N/A,N/A\n")
                with open(all_runs_file, 'a') as f:
                    f.write(f"{instance_name},{run},ERROR,N/A,N/A\n")
        
        print(f"\n  Rodada {run} concluída!")
        print(f"  Resultados salvos em: {run_file}")
    
    print("\n" + "=" * 70)
    print("BENCHMARK CONCLUÍDO!")
    print("=" * 70)
    print(f"Resultados por rodada: Results_Run_1.csv ... Results_Run_{NUM_RUNS}.csv")
    print(f"Resultados consolidados: {all_runs_file}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark_by_rounds()
