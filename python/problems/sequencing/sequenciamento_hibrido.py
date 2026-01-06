import numpy as np
import os
import time
import sys
import gurobipy as gp
from gurobipy import GRB

# Silenciar TODAS as mensagens do Gurobi (incluindo licença)
gp.setParam('LogToConsole', 0)

# Adiciona o caminho para importar RKO e Environment
current_directory = os.path.dirname(os.path.abspath(__file__))
python_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, python_directory)

from rko.rko import RKO
from rko.environment import RKOEnvAbstract

class SequenciamentoHibrido(RKOEnvAbstract):
    """
    Decoder Híbrido:
    1. RKO define a ordem de inserção dos produtos (via random keys)
    2. First Fit aloca cada produto na PRIMEIRA máquina viável
    3. Gurobi otimiza o sequenciamento (TSP) em cada máquina
    """

    def __init__(self, instance_path: str, time_limit_tsp: float = 5.0):
        self.save_q_learning_report = False
        self.max_time = 150
        self.time_limit_tsp = time_limit_tsp  # Tempo limite do Gurobi por máquina
        
        # Leitura da instância
        instances_dir = os.path.join(current_directory, 'instances')
        self.tempos_maquinas, self.custo_maquinas = self.read_instance(os.path.join(instances_dir, instance_path))
        
        self.itens = self.num_products
        self.maquinas = self.num_machines
        self.instance_name = instance_path.split('/')[-1]
        self.LS_type: str = 'Best'
        self.dict_best: dict = {}
        self.tam_solution = self.num_products 
        
        # Parâmetros RKO
        self.BRKGA_parameters = {
            'p': [500, 250], 'pe': [0.20, 0.15], 'pm': [0.05], 'rhoe': [0.70]
        }
        self.SA_parameters = {
            'SAmax': [10, 5], 'alphaSA': [0.5, 0.7], 'betaMin': [0.01, 0.03], 
            'betaMax': [0.05, 0.1], 'T0': [1000]
        }
        self.ILS_parameters = {'betaMin': [0.10, 0.5], 'betaMax': [0.20, 0.15]}
        self.VNS_parameters = {'kMax': [5, 3], 'betaMin': [0.05, 0.1]}
        self.PSO_parameters = {'PSize': [500, 250], 'c1': [2.05], 'c2': [2.05], 'w': [0.73]}
        self.GA_parameters = {'sizePop': [500, 250], 'probCros': [0.98], 'probMut': [0.005, 0.01]}
        self.LNS_parameters = {'betaMin': [0.10], 'betaMax': [0.30], 'TO': [10000], 'alphaLNS': [0.95, 0.9]}

    def read_instance(self, instance_path):
        """Lê a instância do arquivo no formato Li & Milne."""
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
                next(iterator)  # label
                for i in range(self.num_products):
                    for j in range(self.num_products):
                        self.setup_costs[i][j][r] = float(next(iterator))

            self.setup_times = np.zeros((self.num_products, self.num_products, self.num_machines))
            for r in range(self.num_machines):
                next(iterator)  # label
                for i in range(self.num_products):
                    for j in range(self.num_products):
                        self.setup_times[i][j][r] = float(next(iterator))

            print(f"Instância carregada: {self.num_products} produtos, {self.num_machines} máquinas.")
            return self.setup_times, self.setup_costs

        except StopIteration:
            raise ValueError("O arquivo de instância terminou inesperadamente.")
        except Exception as e:
            raise ValueError(f"Erro ao ler o arquivo: {str(e)}")

    def decoder(self, keys: np.ndarray) -> np.ndarray:
        """Transforma random keys em permutação de produtos."""
        return np.argsort(keys)

    def cost(self, solution: np.ndarray, final_solution: bool = False) -> float:
        """
        Decoder Híbrido:
        1. Best Fit: aloca produtos na MELHOR (máquina, posição) global
        2. TSP Ótimo: otimiza sequência em cada máquina com Gurobi
        """
        sorted_products = solution
        
        # =========================================================
        # ETAPA 1: ALOCAÇÃO COM BEST FIT (igual Li & Milne)
        # =========================================================
        machine_sequences = [[] for _ in range(self.num_machines)]
        machine_loads = np.zeros(self.num_machines)
        penalty = 0.0
        
        for product_idx in sorted_products:
            best_machine = -1
            best_position = -1
            min_delta_cost = float('inf')
            best_delta_time = 0.0
            
            # Best Fit: testa TODAS as máquinas e TODAS as posições
            for m in range(self.num_machines):
                rate = self.production_rates[product_idx][m]
                if rate <= 1e-6:
                    continue
                
                prod_time = self.demands[product_idx] / rate
                current_seq = machine_sequences[m]
                
                for pos in range(len(current_seq) + 1):
                    if pos == 0:
                        prev_prod = self.initial_state[m]
                    else:
                        prev_prod = current_seq[pos - 1]
                    
                    next_prod = current_seq[pos] if pos < len(current_seq) else None
                    
                    # Delta Custo
                    cost_add = self.setup_costs[prev_prod][product_idx][m]
                    cost_rem = 0.0
                    if next_prod is not None:
                        cost_add += self.setup_costs[product_idx][next_prod][m]
                        cost_rem = self.setup_costs[prev_prod][next_prod][m]
                    delta_cost = cost_add - cost_rem
                    
                    # Delta Tempo
                    time_add = self.setup_times[prev_prod][product_idx][m]
                    time_rem = 0.0
                    if next_prod is not None:
                        time_add += self.setup_times[product_idx][next_prod][m]
                        time_rem = self.setup_times[prev_prod][next_prod][m]
                    delta_time = time_add + prod_time - time_rem
                    
                    # Verifica capacidade e se é a MELHOR opção global
                    if machine_loads[m] + delta_time <= self.machine_capacities[m]:
                        if delta_cost < min_delta_cost:
                            min_delta_cost = delta_cost
                            best_machine = m
                            best_position = pos
                            best_delta_time = delta_time
                        # Desempate por taxa de produção
                        elif abs(delta_cost - min_delta_cost) < 1e-6:
                            if best_machine != -1:
                                curr_rate = self.production_rates[product_idx][best_machine]
                                if rate > curr_rate:
                                    best_machine = m
                                    best_position = pos
                                    best_delta_time = delta_time
            
            # Aloca na melhor opção encontrada
            if best_machine != -1:
                machine_sequences[best_machine].insert(best_position, product_idx)
                machine_loads[best_machine] += best_delta_time
            else:
                penalty += 100000 + (self.demands[product_idx] * 1000)
        
        # =========================================================
        # ETAPA 2: SEQUENCIAMENTO ÓTIMO (TSP) COM WARM START
        # =========================================================
        # A sequência do First Fit é usada como solução inicial (Warm Start)
        total_setup_cost = 0.0
        optimized_sequences = []
        
        for m in range(self.num_machines):
            initial_sequence = machine_sequences[m]  # Sequência do First Fit
            
            if len(initial_sequence) == 0:
                optimized_sequences.append([])
                continue
            elif len(initial_sequence) == 1:
                p = initial_sequence[0]
                cost = self.setup_costs[self.initial_state[m]][p][m]
                total_setup_cost += cost
                optimized_sequences.append(initial_sequence)
                continue
            
            # TSP com Warm Start (passa a sequência inicial)
            seq, cost = self._optimize_machine_tsp(m, initial_sequence)
            total_setup_cost += cost
            optimized_sequences.append(seq)
        
        total_cost = total_setup_cost + penalty
        
        if final_solution:
            self._print_solution(optimized_sequences, machine_loads, total_cost, total_setup_cost, penalty)
                
        return total_cost

    def _optimize_machine_tsp(self, m_idx: int, products: list) -> tuple:
        """
        Resolve o TSP aberto para uma máquina usando Gurobi.
        
        Args:
            m_idx: índice da máquina
            products: lista de produtos alocados nesta máquina
            
        Returns:
            tuple: (sequência_ótima, custo)
        """
        initial = self.initial_state[m_idx]
        nodes = [initial] + products  # nó 0 = estado inicial
        n = len(nodes)
        
        idx_to_prod = {i: p for i, p in enumerate(nodes)}
        prod_to_idx = {p: i for i, p in enumerate(nodes)}

        model = gp.Model(f"TSP_Maq_{m_idx}")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = self.time_limit_tsp
        model.Params.LazyConstraints = 1

        # Variáveis e custos
        x = {}
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Custo 0 para arcos de volta ao início (simula TSP aberto)
                    if j == 0:
                        cost = 0.0
                    else:
                        cost = self.setup_costs[idx_to_prod[i]][idx_to_prod[j]][m_idx]
                    x[i, j] = model.addVar(obj=cost, vtype=GRB.BINARY, name=f"x_{i}_{j}")

        # =========================================================
        # WARM START: Usar a sequência do First Fit como solução inicial
        # =========================================================
        # A sequência 'products' já vem ordenada do First Fit
        # Configuramos os arcos correspondentes como solução inicial
        for i in range(n):
            for j in range(n):
                if i != j:
                    x[i, j].Start = 0.0
        
        # Definir arcos da sequência inicial
        # Sequência: P0 -> products[0] -> products[1] -> ... -> products[n-2] -> P0
        for k, prod in enumerate(products):
            if k == 0:
                prev_idx = 0  # P0
            else:
                prev_idx = prod_to_idx[products[k - 1]]
            curr_idx = prod_to_idx[prod]
            x[prev_idx, curr_idx].Start = 1.0
        
        # Fechar o ciclo (volta para P0)
        last_idx = prod_to_idx[products[-1]]
        x[last_idx, 0].Start = 1.0

        # Restrições de grau
        for i in range(n):
            model.addConstr(gp.quicksum(x[i, j] for j in range(n) if j != i) == 1)
            model.addConstr(gp.quicksum(x[j, i] for j in range(n) if j != i) == 1)

        # Callback para eliminação de subtour
        def subtour_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                vals = model.cbGetSolution(x)
                edges = [(i, j) for i, j in x.keys() if vals[i, j] > 0.5]
                cycle = self._find_subtour(edges, n)
                if len(cycle) < n:
                    model.cbLazy(
                        gp.quicksum(x[i, j] for i in cycle for j in cycle if i != j) <= len(cycle) - 1
                    )

        model.optimize(subtour_callback)

        # Extrair solução
        if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
            vals = model.getAttr('x', x)
            curr = 0
            route = []
            calc_cost = 0.0
            
            while True:
                found_next = False
                for j in range(n):
                    if curr != j and vals[curr, j] > 0.5:
                        if j == 0:
                            return route, calc_cost
                        
                        prod = idx_to_prod[j]
                        route.append(prod)
                        prev_prod = idx_to_prod[curr]
                        calc_cost += self.setup_costs[prev_prod][prod][m_idx]
                        curr = j
                        found_next = True
                        break
                if not found_next:
                    break
            return route, calc_cost
        
        # Fallback: retorna ordem original
        fallback_cost = sum(
            self.setup_costs[products[i-1] if i > 0 else initial][products[i]][m_idx]
            for i in range(len(products))
        )
        return products, fallback_cost

    def _find_subtour(self, edges: list, n: int) -> list:
        """Encontra o menor ciclo em um conjunto de arcos."""
        successors = {i: j for (i, j) in edges}
        visited = [False] * n
        cycles = []
        
        for start in range(n):
            if not visited[start]:
                cycle = []
                curr = start
                while not visited[curr]:
                    visited[curr] = True
                    cycle.append(curr)
                    curr = successors.get(curr, start)
                cycles.append(cycle)
        
        return min(cycles, key=len) if cycles else []

    def _print_solution(self, sequences, loads, total_cost, setup_cost, penalty):
        """Imprime detalhes da solução final."""
        print("\n" + "="*50)
        print(f"SOLUÇÃO FINAL (HÍBRIDO) - {self.instance_name}")
        print("="*50)
        print(f"Custo Total: {total_cost:.2f}")
        print(f"Custo de Setup: {setup_cost:.2f}")
        print(f"Penalidade: {penalty:.2f}")
        print("-" * 30)
        
        for m, seq in enumerate(sequences):
            load = loads[m]
            cap = self.machine_capacities[m]
            p0 = self.initial_state[m]
            utilizacao = (load / cap) * 100 if cap > 0 else 0
            status = "OK" if load <= cap else "ESTOUROU"
            
            print(f"Máq {m} [P0={p0}]: {len(seq)} itens | {load:.1f}/{cap:.1f} ({utilizacao:.1f}%) [{status}]")
        print("="*50 + "\n")

if __name__ == "__main__":
    for num in range(100, 701, 100):
        for reactors in [5, 10, 15]:
    #         # if (num == 600 and reactors == 5) or (num == 600 and reactors == 10):
    #         #     continue
            env = SequenciamentoHibrido(f'{num}_{reactors}_v2.txt')
    #         start_time = time.time()
    #         # solutions = 0
    #         # while time.time() - start_time < 60:
    #         #     random_solution = np.random.rand(env.tam_solution)
    #         #     decoded = env.decoder(random_solution)
    #         #     cost = env.cost(decoded)
    #         #     solutions += 1
    #         # print(f'Instância: {num}_{reactors}_v2.txt | Soluções em 60s: {solutions}')
            results_dir = os.path.join(current_directory, 'Results')
            solver = RKO(env, True, save_directory=os.path.join(results_dir, 'RKO_testes.txt'))
            final_cost, final_solution, time_to_best = solver.solve(time_total=150, brkga=1, lns=1, vns=1, ils=1, sa=1, pso=1, ga=1, ms=1, runs=10)
    #         env.cost(env.decoder(final_solution), final_solution=True)

    # for num in range(100, 701, 100):
    #     for reactors in [5, 10, 15]:

    #         env = Sequenciamento(f'{num}_{reactors}_v2.txt')
    #         start_time = time.time()
    #         solutions = 0
    #         while time.time() - start_time < 60:
    #             random_solution = np.random.rand(env.tam_solution)
    #             decoded = env.decoder(random_solution)
    #             cost = env.cost(decoded)
    #             solutions += 1
            
    #         with open(os.path.join(current_directory, 'Results', 'RKO_testes.txt'), "a") as f:
    #             f.write(f'Instancia: {num}_{reactors}_v2.txt | Solucoes em 60s: {solutions}\n')