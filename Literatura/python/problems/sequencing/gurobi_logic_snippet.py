
# =============================================================================
# HELPER: ENCONTRAR SUBTOUR (Do benchmark.py)
# =============================================================================
def _find_subtour(edges, n):
    """
    Encontra o menor subtour (ciclo) em um conjunto de arcos.
    Esta função é usada pelo callback de eliminação de subtour.
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
                if not adjacencia[atual]: break # Previne erro se não tiver saída
                atual = adjacencia[atual][0]  # Seguir para o próximo nó
            
            if ciclo:
                ciclos.append(ciclo)
    
    # Retornar o menor ciclo (se < n nós, é um subtour inválido)
    if not ciclos: return []
    return min(ciclos, key=len)

# =============================================================================
# FUNÇÃO GUROBI: Otimiza sequência de UMA máquina (Open TSP)
# =============================================================================
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False
    print("Warning: Gurobi not found. Optimization analysis will be skipped.")

def optimize_machine_sequence_gurobi(m_idx, products, env):
    """
    Calcula a sequência ÓTIMA para um conjunto de produtos alocados a uma máquina.
    Usa Gurobi para resolver o Open TSP.
    Retorna: (sequência_ótima, custo_ótimo)
    """
    if not HAS_GUROBI or not products:
        cost = 0.0
        prev = env.initial_state[m_idx]
        for p in products:
            cost += env.setup_costs[prev][p][m_idx]
            prev = p
        return products, cost

    initial = env.initial_state[m_idx]
    nodes = [initial] + products  # Nó 0 = estado inicial, nós 1..n = produtos
    n = len(nodes)
    
    # Mapeamentos
    idx_to_prod = {i: p for i, p in enumerate(nodes)}
    prod_to_idx = {p: i for i, p in enumerate(nodes)}

    # Modelo
    model = gp.Model(f"OpenTSP_M{m_idx}")
    model.Params.OutputFlag = 0
    model.Params.LazyConstraints = 1 # Habilitar Lazy Constraints
    
    x = {}
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # TRUQUE TSP ABERTO: Retorno ao nó 0 tem custo 0
                if j == 0:
                    cost = 0.0
                else:
                    cost = env.setup_costs[idx_to_prod[i]][idx_to_prod[j]][m_idx]
                
                x[i,j] = model.addVar(obj=cost, vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # Restrições de Grau
    for i in range(n):
        model.addConstr(gp.quicksum(x[i,j] for j in range(n) if j != i) == 1, f"out_{i}")
        model.addConstr(gp.quicksum(x[j,i] for j in range(n) if j != i) == 1, f"in_{i}")

    # Capacidade (Opcional se só queremos sequenciar, mas bom pra garantir consistência)
    # Como os produtos já ESTÃO alocados, assumimos que cabem ou que queremos apenas o melhor custo de setup.
    # Vamos ignorar capacidade aqui para focar puramente em SEQUENCIAMENTO (TSP).

    # Callback Subtour
    def callback_subtour(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(x)
            arcos_ativos = gp.tuplelist((i, j) for i, j in x.keys() if vals[i, j] > 0.5)
            menor_ciclo = _find_subtour(arcos_ativos, n)
            
            if len(menor_ciclo) < n and len(menor_ciclo) > 0:
                model.cbLazy(gp.quicksum(x[i, j] for i in menor_ciclo for j in menor_ciclo if i != j) <= len(menor_ciclo) - 1)

    model.optimize(callback_subtour)

    if model.Status == GRB.OPTIMAL or (model.Status == GRB.TIME_LIMIT and model.SolCount > 0):
        # Reconstrói a rota
        vals = model.getAttr('x', x)
        curr = 0 # Começa do nó dummy (estado inicial)
        route = []
        calc_cost = 0.0
        
        while True:
            found_next = False
            for j in range(n):
                if curr != j and vals[curr, j] > 0.5:
                    if j == 0: # Voltou ao início (fim da rota aberta)
                        return route, calc_cost
                    
                    prod = idx_to_prod[j]
                    route.append(prod)
                    prev_prod = idx_to_prod[curr]
                    calc_cost += env.setup_costs[prev_prod][prod][m_idx]
                    
                    curr = j
                    found_next = True
                    break
            if not found_next: break
        return route, calc_cost

    return products, 0.0 # Fallback
