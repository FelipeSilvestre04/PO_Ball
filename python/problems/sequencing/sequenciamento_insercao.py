import numpy as np
import os
import time
import sys

# Adiciona o caminho para importar RKO e Environment
current_directory = os.path.dirname(os.path.abspath(__file__))
python_directory = os.path.dirname(os.path.dirname(current_directory))
sys.path.insert(0, python_directory)

from rko.rko import RKO
from rko.environment import RKOEnvAbstract

class SequenciamentoInsercao(RKOEnvAbstract):
    """
    Decoder: Best Fit + 2-opt
    1. RKO define a ordem de inserção dos produtos (via random keys)
    2. Best Fit aloca cada produto na MELHOR (máquina, posição)
    3. 2-opt melhora a sequência em cada máquina (heurística rápida)
    """

    def __init__(self, instance_path: str):
        self.save_q_learning_report = False
        self.max_time = 150
        
        # Leitura da instância
        instances_dir = os.path.join(current_directory, 'instances')
        self.tempos_maquinas, self.custo_maquinas = self.read_instance(os.path.join(instances_dir, instance_path))
        
        self.itens = self.num_products
        self.maquinas = self.num_machines
        self.instance_name = instance_path.split('/')[-1]
        self.LS_type: str = 'Best'
        self.dict_best: dict = {}
        self.tam_solution = self.num_products 
        
        # Parâmetros RKO (populações maiores porque agora é rápido!)
        self.BRKGA_parameters = {
            'p': [1000, 500], 'pe': [0.20, 0.15], 'pm': [0.05], 'rhoe': [0.70]
        }
        self.SA_parameters = {
            'SAmax': [10, 5], 'alphaSA': [0.5, 0.7], 'betaMin': [0.01, 0.03], 
            'betaMax': [0.05, 0.1], 'T0': [1000]
        }
        self.ILS_parameters = {'betaMin': [0.10, 0.5], 'betaMax': [0.20, 0.15]}
        self.VNS_parameters = {'kMax': [5, 3], 'betaMin': [0.05, 0.1]}
        self.PSO_parameters = {'PSize': [1000, 500], 'c1': [2.05], 'c2': [2.05], 'w': [0.73]}
        self.GA_parameters = {'sizePop': [1000, 500], 'probCros': [0.98], 'probMut': [0.005, 0.01]}
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
        Decoder: Best Fit + 2-opt
        1. Best Fit: aloca produtos na MELHOR (máquina, posição) global
        2. 2-opt: melhora sequência em cada máquina (heurística rápida)
        """
        sorted_products = solution
        
        # =========================================================
        # ETAPA 1: ALOCAÇÃO COM BEST FIT
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
        # ETAPA 2: MELHORIA COM 2-OPT EM CADA MÁQUINA
        # =========================================================
        total_setup_cost = 0.0
        optimized_sequences = []
        
        for m in range(self.num_machines):
            sequence = machine_sequences[m]
            
            if len(sequence) <= 1:
                # 0 ou 1 produto: não precisa otimizar
                optimized_sequences.append(sequence)
                if len(sequence) == 1:
                    total_setup_cost += self.setup_costs[self.initial_state[m]][sequence[0]][m]
                continue
            
            # Aplica 2-opt para melhorar a sequência
            improved_seq = self._two_opt(sequence, m)
            optimized_sequences.append(improved_seq)
            
            # Calcula custo da sequência otimizada
            total_setup_cost += self._calculate_sequence_cost(improved_seq, m)
        
        total_cost = total_setup_cost + penalty
        
        if final_solution:
            self._print_solution(optimized_sequences, machine_loads, total_cost, total_setup_cost, penalty)
                
        return total_cost

    def _two_opt(self, sequence: list, m_idx: int, max_iterations: int = 1) -> list:
        """
        Aplica 2-opt para melhorar a sequência de uma máquina.
        
        2-opt: tenta inverter segmentos da sequência para reduzir custo.
        Limitado a max_iterations para manter velocidade.
        """
        if len(sequence) < 2:
            return sequence
        
        seq = sequence.copy()
        initial = self.initial_state[m_idx]
        
        for iteration in range(max_iterations):
            best_delta = 0.0
            best_i = -1
            best_j = -1
            
            for i in range(len(seq) - 1):
                for j in range(i + 2, len(seq) + 1):
                    delta = self._calculate_2opt_delta(seq, i, j, m_idx, initial)
                    
                    if delta < best_delta - 1e-9:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i == -1:
                break  # Sem melhoria, para
            
            # Realizar a inversão
            seq[best_i:best_j] = reversed(seq[best_i:best_j])
        
        return seq

    def _calculate_2opt_delta(self, seq: list, i: int, j: int, m_idx: int, initial: int) -> float:
        """
        Calcula o delta de custo para uma troca 2-opt em um problema ASSIMÉTRICO.
        
        Para problemas assimétricos, inverter seq[i:j] muda:
        - O arco de entrada no segmento: prev -> seq[i] vira prev -> seq[j-1]
        - O arco de saída do segmento: seq[j-1] -> next vira seq[i] -> next  
        - TODOS os arcos internos: A->B vira B->A
        """
        n = len(seq)
        segment = seq[i:j]
        
        if len(segment) < 2:
            return 0.0  # Não faz sentido inverter 1 elemento
        
        # Nós nas bordas
        if i == 0:
            prev_node = initial
        else:
            prev_node = seq[i - 1]
        
        next_node = seq[j] if j < n else None
        
        # ===== CUSTO ORIGINAL =====
        # Arco de entrada
        cost_original = self.setup_costs[prev_node][segment[0]][m_idx]
        # Arcos internos
        for k in range(len(segment) - 1):
            cost_original += self.setup_costs[segment[k]][segment[k+1]][m_idx]
        # Arco de saída
        if next_node is not None:
            cost_original += self.setup_costs[segment[-1]][next_node][m_idx]
        
        # ===== CUSTO APÓS INVERSÃO =====
        reversed_segment = segment[::-1]
        # Arco de entrada (agora aponta para o último do segmento original)
        cost_new = self.setup_costs[prev_node][reversed_segment[0]][m_idx]
        # Arcos internos (agora invertidos)
        for k in range(len(reversed_segment) - 1):
            cost_new += self.setup_costs[reversed_segment[k]][reversed_segment[k+1]][m_idx]
        # Arco de saída
        if next_node is not None:
            cost_new += self.setup_costs[reversed_segment[-1]][next_node][m_idx]
        
        return cost_new - cost_original

    def _calculate_sequence_cost(self, sequence: list, m_idx: int) -> float:
        """Calcula o custo total de setup de uma sequência."""
        if len(sequence) == 0:
            return 0.0
        
        initial = self.initial_state[m_idx]
        cost = self.setup_costs[initial][sequence[0]][m_idx]
        
        for i in range(1, len(sequence)):
            cost += self.setup_costs[sequence[i-1]][sequence[i]][m_idx]
        
        return cost

    def _print_solution(self, sequences, loads, total_cost, setup_cost, penalty):
        """Imprime detalhes da solução final."""
        print("\n" + "="*50)
        print(f"SOLUÇÃO FINAL (Best Fit + 2-opt) - {self.instance_name}")
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
            env = SequenciamentoInsercao(f'{num}_{reactors}_v2.txt')
            results_dir = os.path.join(current_directory, 'Results')
            solver = RKO(env, True, save_directory=os.path.join(results_dir, 'RKO_testes.txt'))
            final_cost, final_solution, time_to_best = solver.solve(time_total=150, brkga=1, lns=1, vns=1, ils=1, sa=1, pso=1, ga=1, ms=1, runs=10)
    