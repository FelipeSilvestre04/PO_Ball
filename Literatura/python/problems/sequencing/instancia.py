import numpy as np
import os

current_directory = os.path.dirname(os.path.abspath(__file__))

def generate_li_milne_instance(num_products, num_reactors, filename):
    """
    Gera uma instância seguindo as regras de Li & Milne (2014) com CORREÇÃO MATEMÁTICA NA CAPACIDADE.
    """
    
    # 1. Parâmetros Básicos
    # Produtos: 0 a N-1
    # Reatores: 0 a M-1
    
    # 2. Production Rates (p_ir)
    # "dividing a random number from U[5,15] lbs by a random number from U[10,20] hours"
    numerator = np.random.uniform(5, 15, size=(num_products, num_reactors))
    denominator = np.random.uniform(10, 20, size=(num_products, num_reactors))
    p_ir = numerator / denominator
    
    # 3. Demands (d_i)
    # "multiplying the average of the U[5,15] (which is 10) by a random number from U[1,6]"
    d_i = 10 * np.random.uniform(1, 6, size=num_products)
    
    # 4. Changeover Costs (c_ijr) - U[0, 100]
    c_ijr = np.random.uniform(0, 100, size=(num_products, num_products, num_reactors))
    for i in range(num_products):
        for r in range(num_reactors):
            c_ijr[i][i][r] = 0.0

    # 5. Changeover Times (t_ijr) - U[0, 5]
    t_ijr = np.random.uniform(0, 5, size=(num_products, num_products, num_reactors))
    for i in range(num_products):
        for r in range(num_reactors):
            t_ijr[i][i][r] = 0.0
            
    # 6. Initial Products (P^0_r)
    initial_products = np.random.randint(0, num_products, size=num_reactors)
    
    # 7. Capacities (T_r) - CORREÇÃO CRÍTICA
    # O cálculo anterior usava np.mean(p_ir), o que subestima o tempo necessário (Desigualdade de Jensen).
    # O correto é calcular o tempo médio que cada produto leva (inverso da taxa) e somar.
    
    # Passo A: Calcular o tempo médio de processamento unitário para cada produto (média entre as máquinas)
    # Taxa (p) está em lbs/hora -> Tempo unitário é 1/p (horas/lb)
    avg_unit_proc_time = np.mean(1.0 / p_ir, axis=1) 
    
    # Passo B: Calcular o tempo total de produção esperado para a demanda
    avg_prod_time_total = np.sum(d_i * avg_unit_proc_time)
    
    # Passo C: Adicionar o tempo médio de setup (como já estava correto)
    avg_setup_time_total = num_products * 2.5 # Média de U[0,5] é 2.5
    
    total_time_required = avg_prod_time_total + avg_setup_time_total
    
    # Passo D: Dividir entre os reatores
    cap_per_reactor = total_time_required / num_reactors
    T_r = np.ones(num_reactors) * cap_per_reactor

    # --- ESCRITA DO ARQUIVO ---
    if not os.path.exists(os.path.dirname(filename)):
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

    print(f"Instância {filename} gerada com sucesso (Capacidade Corrigida).")

if __name__ == "__main__":
    # Seed fixa para reprodutibilidade das instâncias v2
    # IMPORTANTE: Não altere este valor para manter consistência com experimentos anteriores
    np.random.seed(79)
    
    # Caminho base relativo
    base_path = os.path.join(current_directory, 'instances')
    
    for num in range(100, 701, 100):
        for reactors in [5, 10, 15]:
            filename = os.path.join(base_path, f"{num}_{reactors}_v2.txt")
            generate_li_milne_instance(num, reactors, filename)