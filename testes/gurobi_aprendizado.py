import gurobipy as gp
from gurobipy import GRB
import numpy as np

# # Criar modelo
# model = gp.Model("exercicio_1")

# # Variáveis de decisão (lb=0 significa lower bound = 0)
# x = model.addVar(lb=0, name="x")
# y = model.addVar(lb=0, name="y")

# # Função objetivo (MAXIMIZE ou MINIMIZE)
# model.setObjective(3*x + 2*y, GRB.MAXIMIZE)

# # Restrições
# model.addConstr(x + y <= 4, "restricao_1")
# model.addConstr(2*x + y <= 6, "restricao_2")

# # Otimizar
# model.optimize()

# # Resultados
# if model.status == GRB.OPTIMAL:
#     print(f"Valor ótimo: {model.objVal}")
#     print(f"x = {x.X}, y = {y.X}")

# import gurobipy as gp
# from gurobipy import GRB
# import numpy as np

# def create_knapsack_instances(num_products):
#     products = []
#     for i in range(num_products):
#         weight = np.random.randint(1, 10)
#         value = np.random.randint(1, 10)
#         products.append((weight, value))

#     # Capacidade entre 30% e 60% do peso total (problema mais interessante)
#     knapsack_size = (0.3 + 0.3 * np.random.uniform()) * sum(w for w, _ in products)
#     return products, knapsack_size

# num_items = 10
# products, knapsack_size = create_knapsack_instances(num_items)

# print(f"Produtos (peso, valor): {products}")
# print(f"Capacidade da mochila: {knapsack_size:.2f}")
# print(f"Peso total disponível: {sum(w for w, _ in products)}")
# print("-" * 50)

# model = gp.Model("knapsack")
# model.Params.OutputFlag = 0  # Silencia o output do Gurobi

# # Variáveis binárias
# x = model.addVars(num_items, vtype=GRB.BINARY, name="x")

# # Objetivo: maximizar valor
# model.setObjective(
#     gp.quicksum(x[i] * products[i][1] for i in range(num_items)), 
#     GRB.MAXIMIZE
# )

# # Restrição: respeitar capacidade
# model.addConstr(
#     gp.quicksum(x[i] * products[i][0] for i in range(num_items)) <= knapsack_size
# )

# # RESOLVER O MODELO!
# model.optimize()

# # Exibir resultados
# if model.status == GRB.OPTIMAL:
#     print(f"\n✅ Valor ótimo: {model.objVal}")
#     print(f"Itens selecionados:")
#     peso_total = 0
#     for i in range(num_items):
#         if x[i].X > 0.5:
#             print(f"  Item {i}: peso={products[i][0]}, valor={products[i][1]}")
#             peso_total += products[i][0]
#     print(f"Peso total usado: {peso_total:.2f} / {knapsack_size:.2f}")




def create_rotine(num_cities):
    cities = np.zeros((num_cities, num_cities))
    for i in range(num_cities):
        # cities.append([])
        for j in range(num_cities):
            if i != j:
                cities[i][j] = np.random.randint(1, 10)

    return cities

def optimize_machiine(route_matriz, num_cities):

    def find_subtour(edges):
        """
        Verifica se há subtour e retorna o menor ciclo encontrado.
        
        Args:
            edges: Lista de tuplas (i, j) representando arcos ativos
            
        Returns:
            tuple: (tem_subtour, ciclo) onde ciclo é a lista de nós do menor ciclo
        """
        # Criar dicionário: nó_origem -> nó_destino
        successors = {i: j for (i, j) in edges}
        
        n = len(edges)  # número total de arcos = número de nós
        visited = set()
        cycles = []
        
        for start in successors.keys():
            if start in visited:
                continue
                
            # Seguir o caminho a partir de start
            cycle = []
            current = start
            
            while current not in visited:
                visited.add(current)
                cycle.append(current)
                current = successors[current]
            
            cycles.append(cycle)
        
        # Encontrar o menor ciclo
        smallest_cycle = min(cycles, key=len)
        
        # Se o menor ciclo tem menos nós que o total, é subtour
        has_subtour = len(smallest_cycle) < n
        # print(smallest_cycle)
        return has_subtour, smallest_cycle
        
        


    model = gp.Model("TSP")

    x = model.addVars(num_cities, num_cities, vtype=GRB.BINARY, name="x")

    model.setObjective(gp.quicksum(route_matriz[i][j]*x[i, j] for i in range(num_cities) for j in range(num_cities)), GRB.MINIMIZE)

    model.addConstrs(gp.quicksum(x[i, j] for j in range(num_cities)) == 1 for i in range(num_cities))
    model.addConstrs(gp.quicksum(x[i, j] for i in range(num_cities)) == 1 for j in range(num_cities))

    model.addConstrs(x[i, i] == 0 for i in range(num_cities))

    model.optimize()

    print(model.objVal)

    route = []
    for i in range(num_cities):
        for j in range(num_cities):
            if x[i, j].X > 0.5:
                print(f"x[{i}, {j}] = {x[i, j].X}")
                route.append((i, j))

    iteration = 0
    has_subtour, cycle = find_subtour(route)

    print(has_subtour)
    print(cycle)

    while has_subtour:
        print(f"\n--- Iteração {iteration}: Subtour encontrado: {cycle} ---")
        
        # SEC: Σ x[i,j] ≤ |S| - 1 para i,j ∈ S
        # CORREÇÃO: cycle é lista de NÓS, não tuplas!
        model.addConstr(
            gp.quicksum(x[i, j] for i in cycle for j in cycle if i != j) <= len(cycle) - 1,
            f"SEC_{iteration}"
        )

        model.optimize()

        # Recoletar arcos ativos
        route = []
        for i in range(num_cities):
            for j in range(num_cities):
                if x[i, j].X > 0.5:
                    route.append((i, j))
        
        has_subtour, cycle = find_subtour(route)
        iteration += 1

    print(f"\n✅ Tour válido encontrado após {iteration} iterações!")
    print(f"Custo final: {model.objVal}")
    print("\nRota final:")
    for i, j in route:
        print(f"  {i} → {j}")
