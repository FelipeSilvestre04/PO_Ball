import numpy as np
import os
import sys
import time
import math
import copy
import gurobipy as gp
from gurobipy import GRB

current_directory = os.path.dirname(os.path.abspath(__file__))

class LiMilneSolver:
    def __init__(self, instance_path, time_limit_gurobi=30):
        self.instance_name = os.path.basename(instance_path)
        self.time_limit_gurobi = time_limit_gurobi
        self.read_instance(instance_path)

    def read_instance(self, instance_path):
        """Lê a instância no formato Li & Milne."""
        if not os.path.exists(instance_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {instance_path}")

        try:
            with open(instance_path, 'r') as f:
                tokens = f.read().split()
            
            iterator = iter(tokens)
            self.num_products = int(next(iterator))
            self.num_machines = int(next(iterator))
            self.machine_capacities = np.array([float(next(iterator)) for _ in range(self.num_machines)])
            self.initial_state = [int(next(iterator)) for _ in range(self.num_machines)]
            self.demands = np.array([float(next(iterator)) for _ in range(self.num_products)])
            self.production_rates = np.zeros((self.num_products, self.num_machines))
            for i in range(self.num_products):
                for r in range(self.num_machines):
                    self.production_rates[i][r] = float(next(iterator))
            self.setup_costs = np.zeros((self.num_products, self.num_products, self.num_machines))
            for r in range(self.num_machines):
                next(iterator) 
                for i in range(self.num_products):
                    for j in range(self.num_products):
                        self.setup_costs[i][j][r] = float(next(iterator))
            self.setup_times = np.zeros((self.num_products, self.num_products, self.num_machines))
            for r in range(self.num_machines):
                next(iterator)
                for i in range(self.num_products):
                    for j in range(self.num_products):
                        self.setup_times[i][j][r] = float(next(iterator))
            print(f"Instância carregada: {self.num_products} prods, {self.num_machines} máqs.")

        except StopIteration:
            raise ValueError("Arquivo incompleto.")
        except Exception as e:
            raise ValueError(f"Erro na leitura: {str(e)}")

    # =========================================================================
    # PASSO 1: ATRIBUIÇÃO E SEQUENCIAMENTO APROXIMADO
    # =========================================================================
    
    def get_sorting_orders(self):
        products = list(range(self.num_products))
        orders = []
        
        # 1. Random
        p1 = list(products)
        np.random.shuffle(p1)
        orders.append(("Random", p1))
        
        # 2. Demand Asc
        p2 = sorted(products, key=lambda i: self.demands[i])
        orders.append(("Demand Asc", p2))
        
        # 3. Dem/AvgRate
        def rule3(i):
            avg_rate = np.mean(self.production_rates[i, :])
            return self.demands[i] / avg_rate if avg_rate > 0 else float('inf')
        p3 = sorted(products, key=rule3)
        orders.append(("Dem/AvgRate Asc", p3))
        
        # 4. AvgSetupDesc
        avg_setup_costs = []
        for i in products:
            total_c = 0
            for r in range(self.num_machines):
                total_c += np.sum(self.setup_costs[i, :, r]) + np.sum(self.setup_costs[:, i, r])
            avg_setup_costs.append(total_c)
        p4 = sorted(products, key=lambda i: avg_setup_costs[i], reverse=True)
        orders.append(("AvgSetupDesc", p4))
        
        # 5. Combined
        def rule5(i):
            avg_rate = np.mean(self.production_rates[i, :])
            avg_cost = avg_setup_costs[i]
            denom = avg_rate * avg_cost
            return self.demands[i] / denom if denom > 0 else float('inf')
        p5 = sorted(products, key=rule5)
        orders.append(("Combined", p5))
        
        return orders

    def step1_constructive(self, sorted_products):
        machine_seqs = [[] for _ in range(self.num_machines)]
        machine_loads = np.zeros(self.num_machines)
        total_cost = 0.0
        unassigned = []

        for p in sorted_products:
            best_machine = -1
            best_pos = -1
            min_delta_cost = float('inf')
            best_delta_time = 0.0

            for m in range(self.num_machines):
                rate = self.production_rates[p][m]
                if rate <= 1e-6: continue

                proc_time = self.demands[p] / rate
                current_seq = [self.initial_state[m]] + machine_seqs[m]
                
                for k in range(len(current_seq)):
                    prev = current_seq[k]
                    next_p = current_seq[k+1] if k + 1 < len(current_seq) else None

                    cost_add = self.setup_costs[prev][p][m]
                    cost_rem = 0.0
                    if next_p is not None:
                        cost_add += self.setup_costs[p][next_p][m]
                        cost_rem = self.setup_costs[prev][next_p][m]
                    delta_cost = cost_add - cost_rem

                    time_add = self.setup_times[prev][p][m]
                    time_rem = 0.0
                    if next_p is not None:
                        time_add += self.setup_times[p][next_p][m]
                        time_rem = self.setup_times[prev][next_p][m]
                    delta_time = time_add + proc_time - time_rem

                    if machine_loads[m] + delta_time <= self.machine_capacities[m]:
                        # Tie-breaking rule do artigo: Menor custo -> Maior rate
                        if delta_cost < min_delta_cost:
                            min_delta_cost = delta_cost
                            best_machine = m
                            best_pos = k
                            best_delta_time = delta_time
                        elif abs(delta_cost - min_delta_cost) < 1e-6:
                            if best_machine != -1:
                                current_rate = self.production_rates[p][best_machine]
                                if rate > current_rate:
                                    best_machine = m
                                    best_pos = k
                                    best_delta_time = delta_time

            if best_machine != -1:
                machine_seqs[best_machine].insert(best_pos, p)
                machine_loads[best_machine] += best_delta_time
                total_cost += min_delta_cost
            else:
                unassigned.append(p)
                total_cost += 1e6 # Penalidade

        return machine_seqs, total_cost, unassigned

    # =========================================================================
    # PASSO 2: SEQUENCIAMENTO EXATO (Gurobi)
    # =========================================================================

    def step2_exact_sequencing(self, solution_seqs):
        new_seqs = []
        new_global_cost = 0.0

        for m_idx, products in enumerate(solution_seqs):
            if not products:
                new_seqs.append([])
                continue

            optimized_seq, cost = self._solve_machine_tsp(m_idx, products)
            
            if optimized_seq is not None:
                new_seqs.append(optimized_seq)
                new_global_cost += cost
            else:
                print(f"  [Aviso] Gurobi falhou na máq {m_idx}, mantendo original.")
                new_seqs.append(products)
                new_global_cost += self._calculate_machine_cost(m_idx, products)

        return new_seqs, new_global_cost

    def _solve_machine_tsp(self, m_idx, products):
        """
        Resolve o TSP aberto (Open TSP) para uma máquina específica.
        
        Usamos eliminação de subtour via Lazy Constraints em vez de MTZ.
        O truque para simular TSP aberto: custo de retorno ao nó inicial = 0.
        
        Args:
            m_idx: Índice da máquina
            products: Lista de produtos a sequenciar nesta máquina
            
        Returns:
            tuple: (sequência_ótima, custo) ou (None, 0.0) se falhar
        """
        initial = self.initial_state[m_idx]
        nodes = [initial] + products  # Nó 0 = estado inicial, nós 1..n = produtos
        n = len(nodes)
        
        # Mapeamentos entre índices locais e produtos originais
        idx_to_prod = {i: p for i, p in enumerate(nodes)}
        prod_to_idx = {p: i for i, p in enumerate(nodes)}

        # =====================================================================
        # CONFIGURAÇÃO DO MODELO GUROBI
        # =====================================================================
        model = gp.Model(f"OpenTSP_Maquina_{m_idx}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.time_limit_gurobi
        
        # IMPORTANTE: Habilitar Lazy Constraints para eliminação de subtour
        model.Params.LazyConstraints = 1

        # =====================================================================
        # VARIÁVEIS DE DECISÃO
        # x[i,j] = 1 se o arco (i -> j) está na solução, 0 caso contrário
        # =====================================================================
        x = {}
        eff_setup_times = {}  # Tempos de setup efetivos (0 para arcos de retorno)

        for i in range(n):
            for j in range(n):
                if i != j:
                    # TRUQUE DO TSP ABERTO:
                    # Arcos que retornam ao nó 0 (inicial) têm custo e tempo = 0
                    # Isso permite usar a estrutura de TSP fechado para resolver TSP aberto
                    if j == 0:
                        cost = 0.0
                        setup_time_val = 0.0
                    else:
                        cost = self.setup_costs[idx_to_prod[i]][idx_to_prod[j]][m_idx]
                        setup_time_val = self.setup_times[idx_to_prod[i]][idx_to_prod[j]][m_idx]

                    x[i,j] = model.addVar(obj=cost, vtype=GRB.BINARY, name=f"x_{i}_{j}")
                    eff_setup_times[i,j] = setup_time_val

        # =====================================================================
        # WARM START - Usar a solução construtiva como ponto inicial
        # =====================================================================
        if len(products) > 0:
            # Arco do início para o primeiro produto
            first_prod_idx = prod_to_idx[products[0]]
            if (0, first_prod_idx) in x:
                x[0, first_prod_idx].Start = 1.0
            
            # Arcos entre produtos consecutivos
            for k in range(len(products) - 1):
                u = prod_to_idx[products[k]]
                v = prod_to_idx[products[k+1]]
                if (u, v) in x:
                    x[u, v].Start = 1.0
            
            # Arco do último produto de volta ao início (fecha o ciclo)
            last_prod_idx = prod_to_idx[products[-1]]
            if (last_prod_idx, 0) in x:
                x[last_prod_idx, 0].Start = 1.0

        # =====================================================================
        # RESTRIÇÕES DE GRAU (cada nó tem exatamente 1 entrada e 1 saída)
        # =====================================================================
        for i in range(n):
            # Exatamente um arco saindo do nó i
            model.addConstr(
                gp.quicksum(x[i,j] for j in range(n) if j != i) == 1, 
                f"saida_{i}"
            )
            # Exatamente um arco entrando no nó i
            model.addConstr(
                gp.quicksum(x[j,i] for j in range(n) if j != i) == 1, 
                f"entrada_{i}"
            )

        # =====================================================================
        # RESTRIÇÃO DE CAPACIDADE DA MÁQUINA
        # =====================================================================
        total_proc = sum(self.demands[p] / self.production_rates[p][m_idx] for p in products)
        setup_expr = gp.quicksum(
            eff_setup_times[i,j] * x[i,j]
            for i in range(n) for j in range(n) if i != j
        )
        model.addConstr(
            total_proc + setup_expr <= self.machine_capacities[m_idx], 
            "capacidade_maquina"
        )

        # =====================================================================
        # ELIMINAÇÃO DE SUBTOUR VIA LAZY CONSTRAINTS (CALLBACK)
        # =====================================================================
        # 
        # Em vez de usar MTZ (Miller-Tucker-Zemlin) que adiciona O(n²) restrições,
        # usamos Lazy Constraints que adiciona restrições DINAMICAMENTE.
        #
        # COMO FUNCIONA:
        # 1. O Gurobi encontra uma solução inteira candidata
        # 2. O callback é chamado (where == GRB.Callback.MIPSOL)
        # 3. Verificamos se a solução tem subtours (ciclos desconectados)
        # 4. Se sim, adicionamos uma restrição que proíbe esse subtour específico
        # 5. O Gurobi continua buscando até encontrar solução sem subtours
        #
        # A restrição adicionada é a SEC (Subtour Elimination Constraint):
        #   Σ x[i,j] ≤ |S| - 1,  para todos i,j ∈ S
        # Onde S é o conjunto de nós que formam o subtour.
        # 
        # Isso diz: "num subconjunto S, pode haver no máximo |S|-1 arcos internos"
        # (se tivesse |S| arcos, formaria um ciclo fechado = subtour)
        #
        # =====================================================================
        
        def callback_subtour_elimination(model, where):
            """
            Callback que é chamado pelo Gurobi quando encontra uma solução inteira.
            Verifica se há subtours e adiciona restrições lazy para eliminá-los.
            """
            # Só executar quando encontrar uma solução inteira (MIP Solution)
            if where == GRB.Callback.MIPSOL:
                # Passo 1: Obter os valores das variáveis x[i,j] na solução atual
                vals = model.cbGetSolution(x)
                
                # Passo 2: Identificar quais arcos estão ativos (valor > 0.5)
                arcos_ativos = gp.tuplelist(
                    (i, j) for i, j in x.keys() if vals[i, j] > 0.5
                )
                
                # Passo 3: Encontrar o menor ciclo/subtour na solução
                menor_ciclo = self._find_subtour(arcos_ativos, n)
                
                # Passo 4: Se o ciclo não inclui todos os nós, é um subtour inválido!
                if len(menor_ciclo) < n:
                    # Adicionar SEC (Subtour Elimination Constraint) como Lazy Constraint
                    # Proíbe esse subtour específico: Σ x[i,j] ≤ |S| - 1 para i,j ∈ S
                    model.cbLazy(
                        gp.quicksum(
                            x[i, j] 
                            for i in menor_ciclo 
                            for j in menor_ciclo 
                            if i != j
                        ) <= len(menor_ciclo) - 1
                    )

        # =====================================================================
        # OTIMIZAÇÃO COM CALLBACK
        # =====================================================================
        model.optimize(callback_subtour_elimination)

        if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
            vals = model.getAttr('x', x)
            curr = 0
            route = []
            calc_cost = 0.0
            
            # Reconstrói a rota
            while True:
                found_next = False
                for j in range(n):
                    if curr != j and vals[curr, j] > 0.5:
                        if j == 0: 
                            # Chegou ao fim (retornou ao dummy 0). Retorna rota e custo.
                            return route, calc_cost
                        
                        prod = idx_to_prod[j]
                        route.append(prod)
                        prev_prod = idx_to_prod[curr]
                        
                        # Soma apenas o custo real (ida), não o de volta (que é 0 de qualquer forma)
                        calc_cost += self.setup_costs[prev_prod][prod][m_idx]
                        
                        curr = j
                        found_next = True
                        break
                if not found_next: break
            return route, calc_cost
        return None, 0.0

    def _find_subtour(self, edges, n):
        """
        Encontra o menor subtour (ciclo) em um conjunto de arcos.
        
        Esta função é usada pelo callback de eliminação de subtour para
        identificar ciclos desconectados na solução candidata.
        
        Args:
            edges: Lista de tuplas (i, j) representando arcos ativos
            n: Número total de nós
            
        Returns:
            list: Lista de nós que formam o menor ciclo encontrado
            
        Exemplo:
            Se edges = [(0,1), (1,2), (2,0), (3,4), (4,3)] e n=5
            Temos dois ciclos: [0,1,2] e [3,4]
            Retorna [3,4] pois é o menor
        """
        # Construir lista de adjacência
        adjacencia = [[] for _ in range(n)]
        for i, j in edges:
            adjacencia[i].append(j)
        
        # Encontrar todos os ciclos
        visitado = [False] * n
        ciclos = []
        
        for i in range(n):
            if not visitado[i]:
                ciclo = []
                atual = i
                
                # Seguir os arcos até voltar a um nó já visitado
                while not visitado[atual]:
                    visitado[atual] = True
                    ciclo.append(atual)
                    atual = adjacencia[atual][0]  # Seguir para o próximo nó
                
                ciclos.append(ciclo)
        
        # Retornar o menor ciclo (se < n nós, é um subtour inválido)
        return min(ciclos, key=len)

    def _calculate_machine_cost(self, m_idx, seq):
        if not seq: return 0.0
        cost = 0.0
        prev = self.initial_state[m_idx]
        for p in seq:
            cost += self.setup_costs[prev][p][m_idx]
            prev = p
        return cost

    # =========================================================================
    # PASSO 3: BUSCA LOCAL (Simplificada/Placeholder)
    # =========================================================================
    def step3_local_search(self, machine_seqs):
        # [cite_start]# [cite: 242] "melhoramos ainda mais a solução... por um método de busca na vizinhança"
        # Mantendo sua implementação ou placeholder aqui
        # Para ser fiel ao paper, implementaria as trocas (n-m) limitadas
        return machine_seqs, 0 # Placeholder

    def solve(self):
        print(f"=== Resolvendo Instância: {self.instance_name} ===")
        
        # 1. Passo 1
        print("\n[Passo 1] Gerando soluções iniciais...")
        orders = self.get_sorting_orders()
        best_solution = None
        best_cost = float('inf')
        
        for name, sorted_prods in orders:
            seqs, cost, unassigned = self.step1_constructive(sorted_prods)
            print(f"  Regra '{name}': Custo = {cost:.2f} | Falta: {len(unassigned)}")
            if cost < best_cost:
                best_cost = cost
                best_solution = seqs
        
        if not best_solution: return [], 0.0

        # 2. Passo 2 (Com Warm Start e Correção Open-TSP)
        print("\n[Passo 2] Refinando com Gurobi (Warm Start + Open TSP)...")
        refined_seqs, refined_cost = self.step2_exact_sequencing(best_solution)
        print(f"  >> Custo após Passo 2: {refined_cost:.2f}")

        # 3. Passo 3 (Placeholder)
        # final_seqs, final_cost = self.step3_local_search(refined_seqs)

        print("\n=== Solução Final (Li & Milne) ===")
        total_items = sum(len(s) for s in refined_seqs)
        print(f"Total Alocado: {total_items}/{self.num_products}")
        print(f"Custo Final: {refined_cost:.2f}")
        return refined_seqs, refined_cost

if __name__ == "__main__":
    # Ajuste o loop conforme suas necessidades de teste
    for num in range(100, 701, 100):    
        for maq in [5, 10, 15]:   
            instance_path = os.path.join(current_directory, 'instances', f'{num}_{maq}_v2.txt')
            if os.path.exists(instance_path):
                solver = LiMilneSolver(instance_path, time_limit_gurobi=100)
                # O retorno do solve() agora retorna seqs, cost. Capturamos isso.
                secs, cost = solver.solve()
                
                # Certifique-se de que o tempo de execução (secs no seu print original)
                # está sendo calculado ou apenas ignore se for só teste
                # Exemplo simples de log:
                with open(os.path.join(current_directory, 'Results', 'benchmark_results_2.txt'), 'a') as f:
                    f.write(f"{num}_{maq}, {cost:.2f}, {secs}\n")
            else:
                print(f"Caminho não encontrado: {instance_path}")