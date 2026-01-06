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
    final_val = env.cost(solution_reconstructed, final_solution=True)
    
    print(f">>> Custo Calculado: {final_val:.2f}")
    if penalty > 0:
        print("❌ VEREDITO: INVÁLIDA (Penalidades detectadas)")
    else:
        print("✅ VEREDITO: VÁLIDA (Viável)")

if __name__ == "__main__":
    # 1. Caminho da Instância
    instance = "500_5_v2.txt"
    
    # 2. Sequência copiada do output do C++
    sequences_from_cpp = [[290, 431, 399, 306, 450, 224, 187, 122, 105, 258, 68, 209, 108, 403, 453, 58, 416, 280, 435, 104, 360, 140, 434, 283, 307, 336, 458, 218, 51, 310, 335, 322, 236, 323, 375, 221, 340, 483, 162, 344, 1, 4, 79, 99, 309, 461, 138, 208, 498, 220, 264, 40, 141, 466, 364, 468, 304, 182, 50, 354, 109, 181, 155, 144, 41, 239, 473, 432, 339, 29, 476, 248, 275, 449, 396, 491, 6, 5, 326, 37, 217, 253, 443, 479, 270, 388, 91, 428, 173, 257, 281, 90, 143, 351, 142, 334, 400, 353, 397, 101, 456, 440, 17], [225, 30, 53, 59, 433, 378, 381, 384, 413, 120, 255, 86, 146, 25, 125, 195, 246, 376, 395, 9, 164, 300, 289, 60, 20, 365, 123, 296, 85, 422, 113, 371, 8, 317, 445, 103, 34, 232, 116, 13, 347, 201, 362, 222, 7, 22, 319, 361, 46, 313, 342, 231, 227, 19, 278, 89, 198, 56, 421, 82, 149, 83, 406, 145, 183, 179, 439, 32, 496, 391, 454, 455, 49, 377, 316, 417, 117, 200, 48, 312, 193, 370, 380, 110, 497, 139, 197], [269, 314, 160, 57, 295, 203, 118, 357, 26, 401, 284, 441, 328, 61, 350, 245, 480, 98, 262, 70, 457, 202, 412, 212, 303, 135, 186, 234, 484, 414, 229, 320, 111, 475, 315, 332, 42, 261, 223, 77, 153, 14, 152, 73, 184, 137, 368, 75, 213, 256, 10, 321, 493, 330, 205, 114, 263, 311, 128, 80, 159, 31, 106, 459, 302, 437, 18, 338, 470, 419, 47, 273, 148, 241], [71, 15, 267, 477, 151, 372, 211, 464, 451, 228, 293, 36, 226, 279, 175, 249, 157, 23, 385, 45, 87, 100, 489, 271, 274, 352, 492, 389, 166, 194, 219, 243, 126, 373, 462, 444, 240, 24, 215, 426, 407, 469, 318, 341, 481, 54, 392, 35, 43, 288, 63, 404, 478, 0, 333, 93, 38, 405, 424, 11, 436, 324, 408, 250, 438, 237, 177, 430, 97, 62, 485, 199, 115, 390, 343, 299, 349, 27, 268, 286, 171, 465, 266, 247, 188, 158, 206, 259, 471, 242, 329, 121, 348, 346, 374, 33, 21, 282, 66, 276, 210, 84, 44, 356, 495, 3, 409, 482, 204], [410, 169, 251, 369, 358, 119, 415, 147, 355, 67, 452, 420, 134, 398, 382, 102, 163, 191, 383, 64, 230, 192, 216, 167, 254, 165, 305, 252, 331, 168, 185, 235, 2, 136, 55, 425, 379, 474, 460, 463, 359, 301, 95, 308, 81, 363, 131, 76, 429, 272, 486, 154, 427, 448, 172, 442, 16, 366, 260, 292, 150, 196, 423, 161, 367, 297, 189, 325, 494, 69, 132, 447, 190, 285, 174, 499, 265, 96, 238, 124, 467, 180, 12, 65, 244, 92, 446, 291, 176, 294, 337, 298, 72, 133, 129, 207, 39, 386, 214, 402, 178, 418, 327, 127, 472, 487, 170, 74, 345, 130, 107, 393, 94, 387, 112, 411, 394, 488, 277, 233, 287, 28, 88, 156, 78, 490, 52]]    
    verify_cpp_solution(instance, sequences_from_cpp)