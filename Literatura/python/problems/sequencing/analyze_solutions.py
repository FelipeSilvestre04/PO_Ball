"""
Análise detalhada de soluções RKO vs Literatura
Compara atribuições de produtos às máquinas e identifica diferenças
"""
import numpy as np
import sys

# Importa o ambiente
sys.path.append('c:/Users/felip/Documents/GitHub/PO_Ball/python')
from problems.sequencing.sequenciamento import Sequenciamento


def decode_rko_solution(keys, env):
    """Decodifica a key do RKO usando o decoder Cheapest Insertion (igual ao C++)"""
    n = env.num_products
    m = env.num_machines
    
    # Ordena produtos por key (argsort)
    sorted_products = np.argsort(keys)
    
    # Estruturas para armazenar a solução
    machine_seqs = [[] for _ in range(m)]
    machine_loads = np.zeros(m)
    total_setup_cost = 0.0
    penalty = 0.0
    
    # Cheapest Insertion Decoder: para cada produto na ordem das keys
    for product_idx in sorted_products:
        best_machine = -1
        best_pos = -1
        min_delta_cost = float('inf')
        best_time_increase = 0.0
        
        # Testa todas as máquinas
        for machine in range(m):
            rate = env.production_rates[product_idx][machine]
            if rate <= 1e-6:
                continue
            
            prod_time = env.demands[product_idx] / rate
            current_seq = machine_seqs[machine]
            seq_len = len(current_seq)
            
            # Testa todas as posições (0 a seq_len)
            for pos in range(seq_len + 1):
                prev_prod = env.initial_state[machine] if pos == 0 else current_seq[pos - 1]
                next_prod = current_seq[pos] if pos < seq_len else -1
                
                # Delta Custo
                cost_add = env.setup_costs[prev_prod][product_idx][machine]
                cost_rem = 0.0
                
                if next_prod != -1:
                    cost_add += env.setup_costs[product_idx][next_prod][machine]
                    cost_rem = env.setup_costs[prev_prod][next_prod][machine]
                
                delta_cost = cost_add - cost_rem
                
                # Delta Tempo
                time_add = env.setup_times[prev_prod][product_idx][machine]
                time_rem = 0.0
                
                if next_prod != -1:
                    time_add += env.setup_times[product_idx][next_prod][machine]
                    time_rem = env.setup_times[prev_prod][next_prod][machine]
                
                delta_time = time_add + prod_time - time_rem
                
                # Verifica Capacidade
                if machine_loads[machine] + delta_time <= env.machine_capacities[machine]:
                    if delta_cost < min_delta_cost:
                        min_delta_cost = delta_cost
                        best_machine = machine
                        best_pos = pos
                        best_time_increase = delta_time
                    # Desempate por taxa de produção
                    elif abs(delta_cost - min_delta_cost) < 1e-6:
                        if best_machine != -1:
                            curr_rate = env.production_rates[product_idx][best_machine]
                            if rate > curr_rate:
                                best_machine = machine
                                best_pos = pos
                                best_time_increase = delta_time
        
        # Realiza Alocação
        if best_machine != -1:
            machine_seqs[best_machine].insert(best_pos, product_idx)
            machine_loads[best_machine] += best_time_increase
            total_setup_cost += min_delta_cost
        else:
            # Penalidade por não alocar (produto não coube em nenhuma máquina)
            penalty += 100000.0 + (env.demands[product_idx] * 1000.0)
    
    return machine_seqs, machine_loads


def analyze_solution(solution, env, name):
    """Analisa uma solução e retorna estatísticas detalhadas"""
    stats = {
        'name': name,
        'num_machines_used': sum(1 for seq in solution if len(seq) > 0),
        'machine_loads': [],
        'machine_products': [],
        'total_demand_per_machine': [],
        'avg_setup_per_machine': [],
        'total_setup_cost_per_machine': [],
        'total_setup_time_per_machine': [],
        'load_balance_std': 0,
        'max_load': 0,
        'min_load': 0,
        'total_cost': 0,
        'total_setup_cost': 0,
    }
    
    total_cost = 0
    
    for m, sequence in enumerate(solution):
        if len(sequence) == 0:
            stats['machine_loads'].append(0)
            stats['machine_products'].append(0)
            stats['total_demand_per_machine'].append(0)
            stats['avg_setup_per_machine'].append(0)
            stats['total_setup_cost_per_machine'].append(0)
            stats['total_setup_time_per_machine'].append(0)
            continue
        
        # Calcula load, setup time e setup cost da máquina
        load = 0
        total_demand = 0
        total_setup_time = 0
        total_setup_cost = 0
        
        # Estado inicial da máquina
        prev_product = env.initial_state[m]
        
        for i, product in enumerate(sequence):
            demand = env.demands[product]
            total_demand += demand
            prod_time = demand / env.production_rates[product][m]
            load += prod_time
            
            # Setup do produto anterior para este
            setup_time = env.setup_times[prev_product][product][m]
            setup_cost = env.setup_costs[prev_product][product][m]
            total_setup_time += setup_time
            total_setup_cost += setup_cost
            load += setup_time
            
            prev_product = product
        
        stats['machine_loads'].append(load)
        stats['machine_products'].append(len(sequence))
        stats['total_demand_per_machine'].append(total_demand)
        stats['avg_setup_per_machine'].append(total_setup_time / len(sequence) if len(sequence) > 0 else 0)
        stats['total_setup_cost_per_machine'].append(total_setup_cost)
        stats['total_setup_time_per_machine'].append(total_setup_time)
        total_cost += total_setup_cost
    
    loads = [l for l in stats['machine_loads'] if l > 0]
    stats['load_balance_std'] = np.std(loads) if loads else 0
    stats['max_load'] = max(loads) if loads else 0
    stats['min_load'] = min(loads) if loads else 0
    stats['load_range'] = stats['max_load'] - stats['min_load']
    stats['total_cost'] = total_cost
    stats['total_setup_cost'] = total_cost
    
    return stats

def check_local_optimality(solution, env):
    """
    Verifica se a solução pode ser melhorada com movimentos simples (Swap e 2-Opt)
    Isso indica se o algoritmo RKO está falhando em atingir ótimos locais.
    """
    total_potential_savings = 0.0
    moves = []

    for m_idx, seq in enumerate(solution):
        if len(seq) < 3: continue # 2-Opt precisa de pelo menos 3 notas para fazer sentido (ou 2, mas swap cobre)
        
        current_seq_cost = _calc_seq_cost(seq, m_idx, env)
        best_machine_saving = 0.0
        
        # 1. Análise de SWAP (Troca dois elementos)
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                # Copia e aplica Swap
                new_seq = list(seq)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_cost = _calc_seq_cost(new_seq, m_idx, env)
                
                if new_cost < current_seq_cost - 1e-4:
                    saving = current_seq_cost - new_cost
                    if saving > best_machine_saving:
                        best_machine_saving = saving

        # 2. Análise de 2-OPT (Inverte segmento)
        # Tenta inverter segmento entre i e j (inclusive)
        # Ex: A-B-C-D-E, i=1(B), j=3(D) -> A-D-C-B-E
        for i in range(len(seq) - 1):
            for j in range(i + 1, len(seq)):
                new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                new_cost = _calc_seq_cost(new_seq, m_idx, env)
                
                if new_cost < current_seq_cost - 1e-4:
                    saving = current_seq_cost - new_cost
                    if saving > best_machine_saving: # Mantém o melhor dos dois mundos (Swap ou 2-Opt)
                        best_machine_saving = saving

        if best_machine_saving > 0:
            total_potential_savings += best_machine_saving
            moves.append(f"Máquina {m_idx}: LS (Swap/2-Opt) melhora {best_machine_saving:.2f}")

    return total_potential_savings, moves

def check_inter_machine_optimality(solution, env):
    """
    BUSCA LOCAL ITERATIVA Inter-Máquinas.
    Estratégia: 
    1. Encontra a pior aresta
    2. Tenta swap com todos os produtos de outras máquinas
    3. Se encontrar melhoria: APLICA o swap, atualiza a solução
    4. Repete até não encontrar mais melhorias
    """
    # Trabalha com cópia da solução para modificar iterativamente
    sol = [list(seq) for seq in solution]
    
    total_savings = 0.0
    moves = []
    iteration = 0
    MAX_ITERATIONS = 200  # Safety limit
    
    print(f"\n>> BUSCA LOCAL ITERATIVA Inter-Máquinas")
    
    while iteration < MAX_ITERATIONS:
        iteration += 1
        
        # === RECALCULA loads e arestas a cada iteração ===
        current_loads = []
        for m in range(env.num_machines):
            load = 0.0
            prev = env.initial_state[m]
            for p in sol[m]:
                rate = env.production_rates[p][m]
                load += (env.demands[p] / rate) + env.setup_times[prev][p][m]
                prev = p
            current_loads.append(load)
        
        # Calcula custos marginais (piores arestas)
        product_edge_info = []
        for m_orig in range(env.num_machines):
            seq = sol[m_orig]
            for i, p in enumerate(seq):
                prev_prod = env.initial_state[m_orig] if i == 0 else seq[i - 1]
                next_prod = seq[i + 1] if i < len(seq) - 1 else -1
                
                cost_in = env.setup_costs[prev_prod][p][m_orig]
                cost_out = env.setup_costs[p][next_prod][m_orig] if next_prod != -1 else 0.0
                cost_bypass = env.setup_costs[prev_prod][next_prod][m_orig] if next_prod != -1 else 0.0
                
                marginal_cost = cost_in + cost_out - cost_bypass
                product_edge_info.append((marginal_cost, p, m_orig, i))
        
        # Ordena por custo marginal DECRESCENTE
        product_edge_info.sort(key=lambda x: -x[0])
        
        # === PROCURA O MELHOR SWAP da iteração ===
        best_global_swap = None
        best_global_saving = 0.0
        
        # Analisa top 30 piores arestas
        for marginal_cost, p1, m1, i1 in product_edge_info[:30]:
            seq1 = sol[m1]
            
            for m2 in range(env.num_machines):
                if m2 == m1: continue
                seq2 = sol[m2]
                
                for i2, p2 in enumerate(seq2):
                    # Verifica rates
                    if env.production_rates[p1][m2] <= 1e-6 or env.production_rates[p2][m1] <= 1e-6:
                        continue
                    
                    # Custo marginal de P2 em M2
                    prev2 = env.initial_state[m2] if i2 == 0 else seq2[i2-1]
                    next2 = seq2[i2+1] if i2 < len(seq2)-1 else -1
                    cost_in2 = env.setup_costs[prev2][p2][m2]
                    cost_out2 = env.setup_costs[p2][next2][m2] if next2 != -1 else 0
                    cost_bp2 = env.setup_costs[prev2][next2][m2] if next2 != -1 else 0
                    saving2 = cost_in2 + cost_out2 - cost_bp2
                    
                    # Sequências sem os produtos
                    seq1_rem = seq1[:i1] + seq1[i1+1:]
                    seq2_rem = seq2[:i2] + seq2[i2+1:]
                    
                    # Best insertion P2 em M1
                    best_ins_cost1, best_ins_time1, best_pos1 = float('inf'), float('inf'), 0
                    prod_time_p2_m1 = env.demands[p2] / env.production_rates[p2][m1]
                    
                    for pos in range(len(seq1_rem) + 1):
                        prev = env.initial_state[m1] if pos == 0 else seq1_rem[pos-1]
                        next_p = seq1_rem[pos] if pos < len(seq1_rem) else -1
                        
                        add = env.setup_costs[prev][p2][m1]
                        if next_p != -1: add += env.setup_costs[p2][next_p][m1]
                        rem = env.setup_costs[prev][next_p][m1] if next_p != -1 else 0
                        delta_c = add - rem
                        
                        add_t = env.setup_times[prev][p2][m1]
                        if next_p != -1: add_t += env.setup_times[p2][next_p][m1]
                        rem_t = env.setup_times[prev][next_p][m1] if next_p != -1 else 0
                        delta_t = add_t + prod_time_p2_m1 - rem_t
                        
                        if delta_c < best_ins_cost1:
                            best_ins_cost1, best_ins_time1, best_pos1 = delta_c, delta_t, pos

                    # Best insertion P1 em M2
                    best_ins_cost2, best_ins_time2, best_pos2 = float('inf'), float('inf'), 0
                    prod_time_p1_m2 = env.demands[p1] / env.production_rates[p1][m2]

                    for pos in range(len(seq2_rem) + 1):
                        prev = env.initial_state[m2] if pos == 0 else seq2_rem[pos-1]
                        next_p = seq2_rem[pos] if pos < len(seq2_rem) else -1
                        
                        add = env.setup_costs[prev][p1][m2]
                        if next_p != -1: add += env.setup_costs[p1][next_p][m2]
                        rem = env.setup_costs[prev][next_p][m2] if next_p != -1 else 0
                        delta_c = add - rem
                        
                        add_t = env.setup_times[prev][p1][m2]
                        if next_p != -1: add_t += env.setup_times[p1][next_p][m2]
                        rem_t = env.setup_times[prev][next_p][m2] if next_p != -1 else 0
                        delta_t = add_t + prod_time_p1_m2 - rem_t
                        
                        if delta_c < best_ins_cost2:
                            best_ins_cost2, best_ins_time2, best_pos2 = delta_c, delta_t, pos
                    
                    # Saldo do swap
                    net_saving = (marginal_cost - best_ins_cost1) + (saving2 - best_ins_cost2)
                    
                    # Verifica capacidade
                    load1_new = _calc_load_approx(seq1_rem, m1, env) + best_ins_time1
                    load2_new = _calc_load_approx(seq2_rem, m2, env) + best_ins_time2
                    cap_ok = load1_new <= env.machine_capacities[m1] and load2_new <= env.machine_capacities[m2]
                    
                    if cap_ok and net_saving > best_global_saving + 1e-4:
                        best_global_saving = net_saving
                        best_global_swap = (p1, m1, i1, p2, m2, i2, best_pos1, best_pos2, marginal_cost)
        
        # === APLICA O MELHOR SWAP (ou para se não houver) ===
        if best_global_swap is None or best_global_saving <= 1e-4:
            print(f">> Iteração {iteration}: Sem mais melhorias. Parando.")
            break
        
        p1, m1, i1, p2, m2, i2, pos1, pos2, edge_cost = best_global_swap
        
        # Remove P1 de M1 e P2 de M2
        sol[m1].remove(p1)
        sol[m2].remove(p2)
        
        # Insere P2 em M1 na melhor posição
        sol[m1].insert(pos1, p2)
        # Insere P1 em M2 na melhor posição
        sol[m2].insert(pos2, p1)
        
        total_savings += best_global_saving
        move_str = f"[{iteration}] Swap P{p1}(M{m1}) <-> P{p2}(M{m2}): +{best_global_saving:.2f} (EdgeCost={edge_cost:.1f})"
        moves.append(move_str)
        
        if iteration <= 10 or iteration % 10 == 0:
            print(f"  {move_str}")
    
    print(f">> TOTAL: {len(moves)} swaps, economia de {total_savings:.2f}")
    
    return total_savings, moves

def _calc_load_approx(seq, m_idx, env):
    """Calcula load de uma sequência"""
    load = 0.0
    prev = env.initial_state[m_idx]
    for p in seq:
        rate = env.production_rates[p][m_idx]
        load += (env.demands[p] / rate) + env.setup_times[prev][p][m_idx]
        prev = p
    return load

def _calc_seq_cost(seq, m_idx, env):
    if not seq: return 0.0
    cost = 0.0
    prev = env.initial_state[m_idx]
    for p in seq:
        cost += env.setup_costs[prev][p][m_idx]
        prev = p
    return cost

def detailed_edge_analysis(solution, env):
    """Analisa a qualidade das arestas escolhidas"""
    edges_costs = []
    for m_idx, seq in enumerate(solution):
        if not seq: continue
        prev = env.initial_state[m_idx]
        for p in seq:
            cost = env.setup_costs[prev][p][m_idx]
            edges_costs.append(cost)
            prev = p
            
    return {
        'avg_edge': np.mean(edges_costs) if edges_costs else 0,
        'max_edge': np.max(edges_costs) if edges_costs else 0,
        'percentiles': np.percentile(edges_costs, [25, 50, 75, 90]) if edges_costs else []
    }


def print_stats(stats):
    """Imprime estatísticas de forma legível"""
    print(f"\n{'='*70}")
    print(f"Solução: {stats['name']}")
    print(f"{'='*70}")
    print(f"Máquinas usadas: {stats['num_machines_used']}/5")
    print(f"\n💰 CUSTO TOTAL: {stats['total_cost']:.2f}")
    print(f"   Setup total: {stats['total_setup_cost']:.2f}")
    
    print(f"\n⚖️  Balanceamento de carga (tempo):")
    print(f"  - Load máximo:  {stats['max_load']:.2f}")
    print(f"  - Load mínimo:  {stats['min_load']:.2f}")
    print(f"  - Range:        {stats['load_range']:.2f}")
    print(f"  - Std Dev:      {stats['load_balance_std']:.2f}")
    
    print(f"\n📊 Por máquina:")
    for m in range(5):
        if stats['machine_loads'][m] > 0:
            print(f"  Máquina {m}: Load={stats['machine_loads'][m]:7.2f} | "
                  f"Produtos={stats['machine_products'][m]:3d} | "
                  f"Setup Cost={stats['total_setup_cost_per_machine'][m]:7.2f} | "
                  f"Setup Time={stats['total_setup_time_per_machine'][m]:6.2f}")


def compare_solutions(rko_sol, lit_sol, env):
    """Compara duas soluções em detalhes"""
    print("\n" + "="*70)
    print("COMPARAÇÃO DE SOLUÇÕES - 400_5_v2.txt")
    print("="*70)
    
    # Analisa ambas
    rko_stats = analyze_solution(rko_sol, env, "RKO C++ (1200s)")
    lit_stats = analyze_solution(lit_sol, env, "Literatura (Gurobi)")
    
    # Imprime
    print_stats(rko_stats)
    print_stats(lit_stats)
    
    # Comparação Principal
    print(f"\n{'='*70}")
    print("🔍 ANÁLISE COMPARATIVA")
    print(f"{'='*70}")
    
    cost_diff = rko_stats['total_cost'] - lit_stats['total_cost']
    cost_gap = (cost_diff / lit_stats['total_cost']) * 100
    
    print(f"\n💰 CUSTO:")
    print(f"  - RKO:        {rko_stats['total_cost']:8.2f}")
    print(f"  - Literatura: {lit_stats['total_cost']:8.2f}")
    print(f"  - Diferença:  {cost_diff:8.2f} ({cost_gap:+.2f}%)  {'❌ PIOR' if cost_diff > 0 else '✅ MELHOR'}")
    
    print(f"\n⚖️  BALANCEAMENTO DE CARGA (Range):")
    print(f"  - RKO:        {rko_stats['load_range']:.2f}")
    print(f"  - Literatura: {lit_stats['load_range']:.2f}")
    print(f"  - Diferença:  {rko_stats['load_range'] - lit_stats['load_range']:.2f} ({'PIOR' if rko_stats['load_range'] > lit_stats['load_range'] else 'MELHOR'})")
    
    print(f"\n📊 DESVIO PADRÃO (Std Dev):")
    print(f"  - RKO:        {rko_stats['load_balance_std']:.2f}")
    print(f"  - Literatura: {lit_stats['load_balance_std']:.2f}")
    print(f"  - Diferença:  {rko_stats['load_balance_std'] - lit_stats['load_balance_std']:.2f} ({'PIOR' if rko_stats['load_balance_std'] > lit_stats['load_balance_std'] else 'MELHOR'})")
    
    # Verifica sobrecarga
    print(f"\n⚠️  VIOLAÇÕES DE CAPACIDADE:")
    rko_overload = sum(1 for m, l in enumerate(rko_stats['machine_loads']) if l > env.machine_capacities[m] + 1e-4)
    lit_overload = sum(1 for m, l in enumerate(lit_stats['machine_loads']) if l > env.machine_capacities[m] + 1e-4)
    print(f"  - RKO:        {rko_overload} máquinas")
    print(f"  - Literatura: {lit_overload} máquinas")
    
    # Análise de alocação de produtos
    print(f"\n📦 DISTRIBUIÇÃO DE PRODUTOS:")
    print(f"  {'Máquina':<10} {'RKO':>8} {'Literatura':>12} {'Diferença':>12}")
    print(f"  {'-'*44}")
    for m in range(5):
        rko_count = rko_stats['machine_products'][m]
        lit_count = lit_stats['machine_products'][m]
        diff = rko_count - lit_count
        print(f"  Máquina {m}  {rko_count:>8d} {lit_count:>12d} {diff:>+12d}")

    # Análise de custos de setup por máquina
    print(f"\n💸 CUSTOS DE SETUP POR MÁQUINA:")
    print(f"  {'Máquina':<10} {'RKO':>10} {'Literatura':>14} {'Diferença':>12}")
    print(f"  {'-'*48}")
    for m in range(5):
        rko_cost = rko_stats['total_setup_cost_per_machine'][m]
        lit_cost = lit_stats['total_setup_cost_per_machine'][m]
        diff = rko_cost - lit_cost
        print(f"  Máquina {m}  {rko_cost:>10.2f} {lit_cost:>14.2f} {diff:>+12.2f}")

    # Análise de Qualidade de Arestas
    rko_edges = detailed_edge_analysis(rko_sol, env)
    lit_edges = detailed_edge_analysis(lit_sol, env)
    
    print(f"\n� QUALIDADE DAS ARESTAS (Setup Individual):")
    print(f"\n QUALIDADE DAS ARESTAS (Setup Individual):")
    print(f"  Média:        RKO={rko_edges['avg_edge']:.2f} vs Lit={lit_edges['avg_edge']:.2f}")
    print(f"  Pior Aresta:  RKO={rko_edges['max_edge']:.2f} vs Lit={lit_edges['max_edge']:.2f}")
    print(f"  Mediana (P50): RKO={rko_edges['percentiles'][1]:.2f} vs Lit={lit_edges['percentiles'][1]:.2f}")
    print(f"  P90 (Cauda):   RKO={rko_edges['percentiles'][3]:.2f} vs Lit={lit_edges['percentiles'][3]:.2f}")

    # ANÁLISE INTER-MACHINE Local Search (Shift/Swap)
    import time
    start_time = time.time()
    inter_savings, inter_moves = check_inter_machine_optimality(rko_sol, env)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if inter_savings > 1e-6:
        inter_pct = (inter_savings / rko_stats['total_setup_cost']) * 100 if rko_stats['total_setup_cost'] > 0 else 0
        final_potential_cost = rko_stats['total_setup_cost'] - inter_savings
        
        print(f"\n=== ANÁLISE INTER-MÁQUINAS (SHIFT/SWAP) ===")
        print(f"Tempo de Análise: {elapsed_time:.4f}s")
        print(f"Melhoria Possível com Shift/Swap: {inter_savings:.2f} ({inter_pct:.2f}%)")
        print(f"Custo Potencial Final: {final_potential_cost:.2f}")
        print(f">> Movimentos encontrados: {len(inter_moves)}")
        print(f">> 3 Melhores Exemplos:")
        # Ordena moves por saving (string parse é chato, mas a lista já vem misturada. Vamos imprimir os 3 primeiros que achou)
        for i in range(min(3, len(inter_moves))):
             print(f"   {inter_moves[i]}")
        print(">> Conclusão: ALOCAÇÃO pode ser melhorada consideravelmente com Busca Local Inter-Máquinas.")
    else:
        print(f"\n=== ANÁLISE INTER-MÁQUINAS (SHIFT/SWAP) ===")
        print(f"Tempo de Análise: {elapsed_time:.4f}s")
        print(">> Nenhuma melhoria com Shift ou Swap simples encontrou.")
        print(">> O sistema está em um Ótimo Local muito forte ou 'travado'.")

    # ANÁLISE LOCAL SEARCH (2-OPT/SWAP)
    ls_savings, ls_moves = check_local_optimality(rko_sol, env)
    if ls_savings > 1e-6:
        ls_pct = (ls_savings / rko_stats['total_setup_cost']) * 100 if rko_stats['total_setup_cost'] > 0 else 0
        print(f"\n=== ANÁLISE INTRA-MÁQUINA (2-OPT/SWAP) ===")
        print(f"Melhoria Possível com LS Simples: {ls_savings:.2f} ({ls_pct:.2f}%)")
        print(f">> Movimentos encontrados: {len(ls_moves)}")
    else:
        print(f"\n=== ANÁLISE INTRA-MÁQUINA (2-OPT/SWAP) ===")
        print(">> Nenhuma melhoria encontrada com LS simples (Swap/2-Opt).")


if __name__ == "__main__":
    # Carrega a instância
    instance_file = "400_5_v2.txt"
    env = Sequenciamento(instance_file)
    
    print(f"Instância carregada: 400_5_v2.txt")
    print(f"Produtos: {env.num_products}, Máquinas: {env.num_machines}")
    print(f"Capacidades: {env.machine_capacities}")
    
    
    # Key do RKO (extraída do Solutions_RKO.txt - 150s: 908.04)


    rko_keys = np.array([
    0.5143068340526274484503233, 0.3901408655315919959960524, 0.8275029432727663003532825, 
    0.3238427692231391485933045, 0.9181105049954322305794108, 0.8024301797405841707444552, 
    0.0289839942902327535323437, 0.3598957006807240177614915, 0.4209128771391321133243935, 
    0.1983373682224438272481137, 0.6587467488804098492138905, 0.9026113568526619168608249, 
    0.5571587984206392007990871, 0.7359489303807721372052697, 0.8484116205403352539704542, 
    0.8910960313617923844731195, 0.3581992194783469862784386, 0.3360104000775907540088383, 
    0.1548388874842319962166215, 0.6250015878312030759289541, 0.0856616752015760957172930, 
    0.5749925603258452255772681, 0.0234480163314108765171273, 0.0079747341181653497166337, 
    0.3641082039645279766482133, 0.2192947465638099002749328, 0.4120210032531241850506376, 
    0.3718515054300841615742002, 0.8523585647091180828027746, 0.8820616573256673254022076, 
    0.2638594882036596755980895, 0.8132186992728224206317122, 0.7166836731704460117242661, 
    0.5507461983363787227929720, 0.8098172930054083895612393, 0.2739097884459935627887717, 
    0.8586972724897886655170964, 0.3482107893394074671356009, 0.8202432529705876440573320, 
    0.3884377766618464233872032, 0.0495090352076261314140382, 0.9742761595788232975579035, 
    0.0177252704796484923666089, 0.9910908226364883999437438, 0.3875211307242437119846556, 
    0.2676092638176346127920624, 0.6029882846389781336782221, 0.5386278605074649172834711, 
    0.4246142542217017368244569, 0.0856964067272546481390805, 0.0229703394328187281148157, 
    0.8665846512523058819965627, 0.3154030002532676735249595, 0.1049065678820636643431286, 
    0.3083019056248325617808348, 0.8606879779116350936973845, 0.7429399410301366080133789, 
    0.1983606881510505126620103, 0.9335840967350014230063948, 0.4947555463415465037968488, 
    0.5114887450129210755278564, 0.5641909998177921936601820, 0.8600820666237849465574072, 
    0.9226027383538296655629551, 0.2077962233436780048467796, 0.3227679818360197350202156, 
    0.8311901721497059192245160, 0.6965791589947243345193328, 0.9364119057191025996544909, 
    0.7662870647726433004365276, 0.5924808266721957039635527, 0.7805681592661140033584388, 
    0.0774429374330520420288693, 0.5271696925668591937608198, 0.9580221277514237021577514, 
    0.0122670451809818881888647, 0.7123759166386059993314461, 0.0611181599436496289512455, 
    0.6104583610687226435942421, 0.7882291830293475243252033, 0.9068075559085732040287553, 
    0.9682915328931319676186718, 0.3005407146924130579179746, 0.4543300007191287126495638, 
    0.6239431071444645482415581, 0.2261782679733051748538486, 0.7514697258394740231324249, 
    0.5842239396416032404246721, 0.6266677771269950625310230, 0.3491839281945645079296980, 
    0.3359587796600618792020043, 0.6307160985343570613892439, 0.5857809849651161027139779, 
    0.0646529081140381350900626, 0.5919303396970690966938378, 0.2467311898847591444816629, 
    0.3319823231461325185875921, 0.3817770252183275148105679, 0.5997344640456049313925746, 
    0.8070107847115156607387121, 0.3424599803346787263436113, 0.8248425812760247222144017, 
    0.1356567433963558888532219, 0.9775763765519951409288524, 0.8096550502814088279279758, 
    0.4750079484736287604995653, 0.3818227282552895118783454, 0.2041726064211487889910046, 
    0.3873192538820578456615351, 0.3442239837621036135928421, 0.0993481342782435716820544, 
    0.6067062176831803377297092, 0.7294374424501389730934875, 0.2643469208496989630852170, 
    0.1613190525389237850450996, 0.0628244620571663109354077, 0.8132061826316757668919877, 
    0.9138523332062925552676802, 0.9164632197575578453907497, 0.9406778328785763143571330, 
    0.4733126977157817716346244, 0.7919487697813788473766294, 0.0430350353964685308771898, 
    0.7972187691665577613520099, 0.9125752975364543884140289, 0.5560656258115709293932127, 
    0.7726500410263362761398298, 0.9932776962078291305147104, 0.2688375775881042262227538, 
    0.6728137199185670880297039, 0.3768499237901264953265468, 0.3432192062002980548207631, 
    0.0617477828418304397484206, 0.9518309864167243672383734, 0.0763299001156470258155906, 
    0.8361655466305540906901683, 0.8880354042542156767225947, 0.6730176774489594615502597, 
    0.1357041962441427884744627, 0.3577346937097568013719240, 0.5693351455839451569218568, 
    0.5702165763666141584664615, 0.4478428414512838351413393, 0.4397161104705655798596808, 
    0.7922744308339386698492035, 0.4618837732511580362881887, 0.3556628622322698385360695, 
    0.4203521821698542604295312, 0.7653500912983034476511079, 0.9064305523783650064828521, 
    0.1291098092994518720111330, 0.7912248119261510170829865, 0.8991858936292960180480804, 
    0.5930846728803974565735757, 0.5466725749506030629376596, 0.3545391115481294486144748, 
    0.8415025778057465544179649, 0.8494997723195828465136969, 0.7129952566727407736735245, 
    0.8488849759839817155082642, 0.1375096775413744421001638, 0.7177044998596661118384077, 
    0.1056159248090095814642808, 0.7167901047188954821720586, 0.3353794581767358118362665, 
    0.0826041918160409921467746, 0.2060775369570874426816687, 0.9004126711250060033364662, 
    0.9781113105101036930832947, 0.0572815978924770541347300, 0.2552781681157725302000472, 
    0.1341565570082291214415449, 0.3599166053804451759212668, 0.6616419554354983079136332, 
    0.3762143759371454621032171, 0.1804468526861607302613777, 0.3686717336169826886305145, 
    0.7442783784735782504071722, 0.7092448305645092654359019, 0.7991549543769362884404472, 
    0.4544817503120138768579750, 0.3649354712501161612792089, 0.6444720716865928178407330, 
    0.9727933549579915739613512, 0.2585310716568954281235904, 0.7871493091955577003915323, 
    0.1487677185747730002063349, 0.8359653021093624847281944, 0.4109579488094649013518733, 
    0.1399224972518349485728351, 0.3897757475868510179672910, 0.4723243080124679971198987, 
    0.2984388968901851568737982, 0.0833779905104367058399362, 0.9344052894822056920531850, 
    0.9767189909058600072455647, 0.5606781230407594796005810, 0.2604450737969135065696946, 
    0.3615331137226296953635085, 0.1947773400410814925720615, 0.7758546139331231605851258, 
    0.7499381927390487145856923, 0.6321933090752896999831023, 0.1827762195345566942350501, 
    0.7546098011717624620686706, 0.7893550475603092753118517, 0.0735634771481647325996889, 
    0.1466853803899536290433758, 0.2619816864217517915491840, 0.9421869563209542830506393, 
    0.8080531314691865585331243, 0.8208820460044466971183397, 0.9372096577403266071115695, 
    0.1133567522763110346772919, 0.6208565178780998117247236, 0.4847198730255871113037358, 
    0.7040007898097002669857147, 0.7108090987617486922189869, 0.2365250608595729220340331, 
    0.9142029359216152784028964, 0.5796911763917972360005137, 0.1591113916189240695686635, 
    0.1485965198204377557544831, 0.5805570078418809121245658, 0.0659747673019520364245949, 
    0.0208814021100311650314119, 0.1001027505805592299203965, 0.0418911105437699826659070, 
    0.1089525574990167777755801, 0.5934930780958395102331338, 0.0122585245592777072887181, 
    0.5827356258735563176998085, 0.4231224939585650490769808, 0.2119657412630843729139940, 
    0.2238552167136869941455046, 0.8289691968646967223932620, 0.9550285492069344739363146, 
    0.3682990352101229714065767, 0.7951738162670800580400510, 0.6391875378722687850441275, 
    0.1723564018717239354483439, 0.8494059069305532894134103, 0.6257101828672163579270205, 
    0.1541914429655160678755976, 0.4012503724814483474503390, 0.7255448396886193807020504, 
    0.1239244291880925025184368, 0.6113020335710944719309623, 0.7422182423639207549115326, 
    0.0331128935876445901209664, 0.9408040584118395610602192, 0.2927021222117231480375210, 
    0.8833093011800892124796292, 0.1720891793561725169325882, 0.9342717699054491342636197, 
    0.8267096754124247937056680, 0.7030597904433962819226167, 0.8977844365356050237991781, 
    0.0222121525148637651014827, 0.5829458674562080222614213, 0.5872261978403269244708440, 
    0.1447371133598785997165237, 0.5510005408134545090703682, 0.0793698726699904461012736, 
    0.4959185222669454939925515, 0.9077892613965872214976116, 0.4695362528704758453557133, 
    0.3779024707172939923083277, 0.4628484128288317123178786, 0.9599858642753098525446376, 
    0.3216617830526642962674089, 0.8029083762531564749664881, 0.5010603065048542248405283, 
    0.3602754473783748023762996, 0.1805218116569986541719572, 0.5308769722058862550184699, 
    0.4642744240397898569661095, 0.3106921005543324398878724, 0.0095852115089305623546467, 
    0.6760200619957303924678627, 0.4115140729616729164952460, 0.1031802446985495280706857, 
    0.3555893768676451349719514, 0.6883447778550504381200881, 0.0446101574975051665128412, 
    0.5152424780341587906562495, 0.5206768082185581070220337, 0.8190157391948258425173890, 
    0.2254586183313192215837972, 0.8138424404291612068718109, 0.5964556494301601619056896, 
    0.0052650707108890335700346, 0.1673170189909497507230185, 0.9140137732249865587519366, 
    0.4008520961468716725661920, 0.9740896984048780238296672, 0.3093805054473406368487076, 
    0.3230161390194193904434883, 0.0539776377545711227856629, 0.0821296442575176061096087, 
    0.1592519687578195786148427, 0.9501529789334003384126959, 0.9306800678202400423089102, 
    0.1095761015990891623639314, 0.3043617382073367516603923, 0.1841588939451943296443659, 
    0.7355281498152244079591355, 0.2751876677829535600139366, 0.6570712376992140502096618, 
    0.1364472962018744572887385, 0.4764127158876261813880149, 0.0042853453292676767921732, 
    0.5583051049450499236925793, 0.0397646806509518793704139, 0.6842354913385642056766756, 
    0.5985106172049921013922358, 0.6106961394075987037410869, 0.5659983246140848356375841, 
    0.5786802640958643717539189, 0.5839148152058365459637912, 0.2944177215871534580848845, 
    0.3293694816350098975021865, 0.5048845998081109565092106, 0.1281311450656646699908237, 
    0.0026254309652661953107533, 0.6533806030975580769748490, 0.6058602449036407655569292, 
    0.2350331775115618859750555, 0.5521185703284929857659336, 0.8557778203031751740326172, 
    0.3515030548659007769707330, 0.9284325031496371005701462, 0.7512329571454949839193205, 
    0.6518037872039080804498212, 0.9412777117994539644030283, 0.7202379310723665550852957, 
    0.8514186097152479648642043, 0.1473931388916511486808503, 0.7142400467708576172753965, 
    0.5635934177087401630856789, 0.0601541147567384928973588, 0.4686469394266397636705790, 
    0.8037646787220483757252509, 0.2675660926714300003048663, 0.6888048788997301885572710, 
    0.1425522954967644029267149, 0.7871350235564770203566809, 0.4172210101136848425795733, 
    0.1889733060799395070183948, 0.0674343662309028618606632, 0.9757938878236627200379871, 
    0.6288019373724688110272041, 0.8268001773146417265891728, 0.9179711523274098894376039, 
    0.4438001689076219724583439, 0.2131558204597791550405361, 0.6512886518047815709309134, 
    0.1128799685477957087664791, 0.3485249410556147120487935, 0.0075757485631278310284076, 
    0.7295342801055610948424146, 0.8486519653559846387835819, 0.1403202916005407696609097, 
    0.0924211102643625465047350, 0.5829271882836657425741578, 0.4511434921192962743674570, 
    0.4272968175960968340021395, 0.6478202797610228547142697, 0.2686826219574623419994452, 
    0.6856035240301845412602688, 0.5262677880962206566906048, 0.0272798337095200606194911, 
    0.3685119387894512099634881, 0.1020189328409117496265779, 0.4245004116747679856480602, 
    0.6657235818434210639082949, 0.5735173775362264469990237, 0.7206376226838407950125998, 
    0.2814476000215909157198269, 0.5360278242981703078484657, 0.6771953757990099020958041, 
    0.6727810015968901380389866, 0.3029218700916531670230825, 0.7283123538205590330463224, 
    0.7925527132470788771456682, 0.8068745374482018117845428, 0.0327650933801152746682739, 
    0.6131048193220102193734533, 0.6369048630753543216442836, 0.7048284734564111975174683, 
    0.8942281595375517477819471, 0.8574926767291354279265647, 0.8194743169040228369937040, 
    0.6087031064275899572280082, 0.1583712678403761953127571, 0.3215771048723479186293162, 
    0.9189688812524703376283242, 0.2213482614991800490056306, 0.9644216117839795598953856, 
    0.7192144800820780004002586
])
    # Solução da literatura (benchmark_results_2.txt) - Custo: 572.80
    lit_solution = [[121, 171, 372, 126, 301, 271, 92, 352, 3, 115, 273, 385, 30, 168, 45, 158, 266, 344, 234, 287, 322, 397, 85, 253, 257, 285, 72, 213, 198, 49, 342, 298, 398, 264, 101, 27, 164, 160, 394, 315, 232, 329, 176, 208, 214, 28, 116, 89, 272, 140, 81, 235, 41, 186, 38, 157, 110, 66, 8, 204, 55, 278, 291, 313, 250, 345, 379, 151, 292, 108, 42, 0, 77, 163, 4, 2, 381, 113, 31, 303, 239, 286, 297, 84, 228, 349, 104, 95, 276, 133, 236, 393, 251], [33, 26, 217, 6, 231, 167, 340, 331, 245, 294, 29, 392, 304, 134, 247, 46, 172, 229, 367, 203, 233, 351, 173, 201, 105, 125, 152, 290, 149, 141, 21, 154, 63, 156, 296, 184, 318, 376, 74, 327, 306, 354, 94, 312, 255, 267, 191, 17, 225, 258, 44, 103, 369, 150, 130, 363, 325, 178, 162, 51, 383, 366, 102, 132, 262, 82, 226, 114, 39, 289, 246, 128, 58, 166, 87, 333], [43, 299, 189, 71, 5, 387, 391, 79, 323, 170, 211, 382, 36, 14, 193, 124, 370, 254, 123, 169, 241, 20, 12, 67, 268, 107, 288, 280, 122, 161, 371, 365, 330, 373, 120, 293, 131, 341, 188, 206, 59, 181, 194, 88, 136, 78, 353, 56, 24, 275, 307, 389, 54, 69, 368, 202, 196, 207, 362, 321, 70, 155, 215, 129, 317, 374, 137, 144, 319, 221], [270, 16, 1, 346, 37, 205, 93, 311, 227, 106, 10, 210, 242, 187, 61, 305, 146, 378, 96, 334, 223, 76, 75, 220, 177, 98, 179, 249, 336, 183, 112, 324, 73, 19, 90, 62, 52, 222, 174, 65, 111, 302, 261, 388, 348, 328, 86, 199, 396, 310, 200, 240, 34, 212, 237, 47, 40, 320, 139, 357, 380, 190, 15, 119, 159, 64, 68, 343, 182, 274, 335, 390, 295, 243, 192, 50, 138, 118, 22, 260, 277, 356, 281, 244, 99, 386], [364, 13, 48, 316, 32, 80, 127, 117, 35, 282, 252, 283, 314, 375, 25, 355, 377, 358, 347, 100, 9, 338, 284, 309, 360, 263, 259, 308, 148, 230, 332, 279, 165, 18, 97, 23, 147, 248, 197, 326, 218, 384, 359, 209, 395, 135, 11, 256, 265, 224, 300, 142, 180, 83, 109, 361, 399, 7, 350, 60, 153, 185, 219, 339, 216, 269, 145, 57, 337, 195, 91, 175, 53, 238, 143]]
    
    if rko_keys is not None:
        # Decodifica solução RKO
        print("\n\nDecodificando solução RKO...")
        rko_solution, rko_loads = decode_rko_solution(rko_keys, env)
        
        # Compara
        compare_solutions(rko_solution, lit_solution, env)
    else:
        print("\n\nApenas analisando solução da literatura:")
        lit_stats = analyze_solution(lit_solution, env, "Literatura (Gurobi)")
        print_stats(lit_stats)
