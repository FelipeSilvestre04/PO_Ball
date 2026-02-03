import numpy as np
import os
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))

# Importação do módulo sequenciamento
from sequenciamento import Sequenciamento


def verify_cpp_solution(instance_path, cpp_sequences):
    """
    Recebe a sequência final (lista de listas) gerada pelo C++ e recalcula
    o custo e tempo no Python para validação cruzada.
    """
    filename = os.path.basename(instance_path)
    print(f"=== Verificando Sequência C++ para {filename} ===")
    
    # if not os.path.exists(instance_path):
    #     print(f"Erro: Arquivo de instância '{instance_path}' não encontrado.")
    #     return

    # 1. Carrega a instância (Dados: custos, tempos, demandas, etc.)
    env = Sequenciamento(instance_path)
    
    # Validação básica
    if len(cpp_sequences) != env.num_machines:
        print(f"ALERTA: A solução tem {len(cpp_sequences)} máquinas, mas a instância tem {env.num_machines}.")

    # 2. Recálculo Manual dos Custos e Tempos
    # Vamos reconstruir o estado da solução passo a passo
    machine_loads = np.zeros(env.num_machines)
    total_setup_cost = 0.0
    penalty = 0.0
    assigned_products = set()
    
    print("\n--- Detalhes do Recálculo ---")
    
    for m, seq in enumerate(cpp_sequences):
        if m >= env.num_machines: break
        
        # Produto 'anterior' inicial é o estado inicial da máquina (P0)
        prev_prod = env.initial_state[m]
        
        for product_idx in seq:
            assigned_products.add(product_idx)
            
            # A. Taxa de Produção
            rate = env.production_rates[product_idx][m]
            if rate <= 1e-6:
                print(f"[ERRO] Máquina {m} não consegue processar produto {product_idx} (Taxa ~ 0).")
                penalty += 1e6
                prod_time = 0
            else:
                prod_time = env.demands[product_idx] / rate
            
            # B. Setup (Tempo e Custo)
            # Matrizes são [origem][destino][maquina]
            setup_time = env.setup_times[prev_prod][product_idx][m]
            setup_cost = env.setup_costs[prev_prod][product_idx][m]
            
            # C. Acumula
            machine_loads[m] += setup_time + prod_time
            total_setup_cost += setup_cost
            
            # Atualiza o anterior para o próximo passo
            prev_prod = product_idx
            
        # D. Verifica Capacidade da Máquina
        if machine_loads[m] > env.machine_capacities[m]:
            excess = machine_loads[m] - env.machine_capacities[m]
            print(f"[VIOLAÇÃO] Máquina {m} estourou capacidade em {excess:.4f}.")
            # Se quiser aplicar penalidade de tempo igual ao decoder:
            # penalty += excess * 1000 
    
    # 3. Verifica se faltou algum produto
    expected_products = set(range(env.num_products))
    missing = expected_products - assigned_products
    if missing:
        print(f"[VIOLAÇÃO] Produtos faltando na solução: {missing}")
        penalty += len(missing) * 100000

    # 4. Monta o dicionário de solução para usar o print oficial
    solution_reconstructed = {
        "sequences": cpp_sequences,
        "total_cost": total_setup_cost + penalty,
        "penalty": penalty,
        "times": machine_loads
    }
    
    # 5. Exibe Resultado
    print("\n=== Resultado da Verificação (Python) ===")
    print(f">>> Custo Total de Setup: {total_setup_cost:.2f}")
    print(f">>> Penalidade: {penalty:.2f}")
    print(f">>> CUSTO FINAL: {(total_setup_cost + penalty):.2f}")
    print("\n--- Cargas das Máquinas ---")
    for m in range(len(cpp_sequences)):
        if m < env.num_machines:
            print(f"Máquina {m}: {machine_loads[m]:.2f} / {env.machine_capacities[m]:.2f}")
    
    if penalty > 0:
        print("\n❌ VEREDITO: INVÁLIDA (Penalidades detectadas)")
    else:
        print("\n✅ VEREDITO: VÁLIDA (Viável)")

if __name__ == "__main__":
    # 1. Caminho da Instância
    instance = "400_5_v2.txt"
    
    # Chaves de alta precisão (OFV: 942.23)
    keys = np.array([
        0.962, 0.090, 0.947, 0.012, 0.989, 0.009, 0.488, 0.225, 0.163, 0.436, 0.612, 0.696, 0.695, 0.619, 0.618, 0.840, 
        0.941, 0.638, 0.614, 0.855, 0.449, 0.989, 0.833, 0.610, 0.992, 0.739, 0.878, 0.616, 0.898, 0.681, 0.371, 0.663, 
        0.408, 0.963, 0.328, 0.558, 0.677, 0.152, 0.005, 0.372, 0.197, 0.032, 0.379, 0.338, 0.827, 0.343, 0.665, 0.501, 
        0.178, 0.302, 0.665, 0.859, 0.915, 0.898, 0.081, 0.997, 0.659, 0.651, 0.757, 0.985, 0.862, 0.117, 0.610, 0.627, 
        0.536, 0.238, 0.313, 0.013, 0.941, 0.251, 0.165, 0.402, 0.022, 0.139, 0.740, 0.858, 0.680, 0.514, 0.785, 0.602, 
        0.967, 0.106, 0.550, 0.032, 0.123, 0.087, 0.578, 0.455, 0.606, 0.947, 0.006, 0.886, 0.710, 0.436, 0.772, 0.777, 
        0.739, 0.923, 0.184, 0.046, 0.897, 0.032, 0.301, 0.808, 0.448, 0.451, 0.170, 0.587, 0.040, 0.920, 0.890, 0.918, 
        0.555, 0.355, 0.144, 0.318, 0.866, 0.391, 0.160, 0.835, 0.092, 0.608, 0.254, 0.485, 0.513, 0.314, 0.730, 0.283, 
        0.483, 0.168, 0.477, 0.136, 0.720, 0.477, 0.047, 0.608, 0.221, 0.326, 0.431, 0.317, 0.522, 0.666, 0.066, 0.202, 
        0.353, 0.392, 0.671, 0.692, 0.498, 0.202, 0.595, 0.048, 0.923, 0.686, 0.558, 0.001, 0.178, 0.949, 0.388, 0.747, 
        0.688, 0.165, 0.054, 0.260, 0.409, 0.236, 0.286, 0.006, 0.606, 0.736, 0.253, 0.701, 0.036, 0.812, 0.728, 0.705, 
        0.003, 0.580, 0.345, 0.393, 0.653, 0.800, 0.704, 0.842, 0.651, 0.747, 0.657, 0.941, 0.212, 0.181, 0.601, 0.049, 
        0.814, 0.279, 0.342, 0.234, 0.403, 0.873, 0.931, 0.175, 0.130, 0.795, 0.886, 0.357, 0.054, 0.185, 0.660, 0.175, 
        0.623, 0.600, 0.530, 0.234, 0.675, 0.698, 0.137, 0.514, 0.965, 0.949, 0.176, 0.021, 0.258, 0.278, 0.428, 0.860, 
        0.020, 0.823, 0.953, 0.675, 0.568, 0.173, 0.350, 0.409, 0.292, 0.999, 0.322, 0.220, 0.666, 0.505, 0.964, 0.431, 
        0.927, 0.862, 0.292, 0.699, 0.677, 0.368, 0.980, 0.579, 0.337, 0.855, 0.867, 0.821, 0.054, 0.249, 0.307, 0.523, 
        0.320, 0.156, 0.975, 0.186, 0.077, 0.835, 0.488, 0.745, 0.571, 0.306, 0.193, 0.555, 0.857, 0.357, 0.177, 0.897, 
        0.952, 0.470, 0.911, 0.327, 0.663, 0.118, 0.187, 0.574, 0.228, 0.204, 0.468, 0.135, 1.000, 0.142, 0.808, 0.628, 
        0.909, 0.542, 0.446, 0.352, 0.277, 0.968, 0.312, 0.457, 0.898, 0.400, 0.328, 0.950, 0.233, 0.413, 0.029, 0.100, 
        0.049, 0.948, 0.163, 0.511, 0.604, 0.792, 0.906, 0.472, 0.186, 0.580, 0.361, 0.210, 0.170, 0.392, 0.963, 0.230, 
        0.282, 0.154, 0.322, 0.910, 0.875, 0.246, 0.168, 0.662, 0.948, 0.675, 0.798, 0.153, 0.638, 0.134, 0.051, 0.045, 
        0.947, 0.418, 0.697, 0.700, 0.293, 0.794, 0.919, 0.220, 0.845, 0.354, 0.058, 0.595, 0.259, 0.994, 0.977, 0.763, 
        0.424, 0.510, 0.127, 0.942, 0.026, 0.722, 0.139, 0.016, 0.274, 0.120, 0.374, 0.496, 0.178, 0.601, 0.335, 0.178, 
        0.939, 0.122, 0.406, 0.332, 0.132, 0.169, 0.635, 0.566, 0.192, 0.147, 0.610, 0.938, 0.363, 0.936, 0.099, 0.042, 
        0.003, 0.911, 0.978, 0.696, 0.213, 0.118, 0.087, 0.475, 0.961, 0.385, 0.598, 0.256, 0.537, 0.661, 0.750, 0.968
    ])