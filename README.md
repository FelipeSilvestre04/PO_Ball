# RKO - Random-Key Optimizer

Framework de otimização metaheurística para problemas combinatórios, com implementações em **Python** e **C++**.

> **Foco atual:** Problemas de Sequenciamento (Scheduling)

---

## 📁 Estrutura do Repositório

```
PO_Ball/
├── python/
│   ├── rko/                           # Framework RKO
│   │   ├── __init__.py
│   │   ├── rko.py                     # Classe principal RKO
│   │   └── environment.py             # Classe abstrata RKOEnvAbstract
│   │
│   └── problems/
│       └── sequencing/                # Problema de Sequenciamento
│           ├── sequenciamento.py      # Ambiente do problema
│           ├── benchmark.py           # Benchmark com Gurobi
│           ├── instancia.py           # Gerador de instâncias
│           ├── verificar_sol.py       # Verificador de soluções C++
│           ├── instances/             # Arquivos de instância
│           └── Results/               # Resultados de experimentos
│
├── cpp/                               # Implementação C++ do RKO
│   └── Program/
│       ├── src/                       # Código fonte
│       └── ...
│
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Python

```python
import sys
sys.path.insert(0, 'python')

from rko import RKO
from problems.sequencing import Sequenciamento

# Criar ambiente
env = Sequenciamento('100_5_v2.txt')

# Executar otimização
solver = RKO(env, print_best=True)
cost, solution, time = solver.solve(
    time_total=60,
    brkga=1,
    sa=1,
    ils=1
)
```

### C++

```bash
cd cpp/Program
make
./runTest ../Instances/100_5.txt
```

---

## 🔧 Framework RKO

O RKO (Random-Key Optimizer) codifica soluções como vetores de chaves aleatórias no intervalo [0, 1). 
Um **decoder** específico do problema transforma essas chaves em soluções viáveis.

### Metaheurísticas Disponíveis

| Sigla | Algoritmo |
|-------|-----------|
| BRKGA | Biased Random-Key Genetic Algorithm |
| SA | Simulated Annealing |
| ILS | Iterated Local Search |
| VNS | Variable Neighborhood Search |
| PSO | Particle Swarm Optimization |
| GA | Genetic Algorithm |
| LNS | Large Neighborhood Search |

### Criando um Novo Problema

Herde de `RKOEnvAbstract` e implemente:

```python
from rko.environment import RKOEnvAbstract

class MeuProblema(RKOEnvAbstract):
    def __init__(self):
        self.tam_solution = N  # Tamanho do vetor de chaves
        # ... configurar parâmetros
    
    def decoder(self, keys):
        # Transforma chaves em solução
        return solucao
    
    def cost(self, solution, final_solution=False):
        # Retorna custo (minimização)
        return custo
```

---

## 📊 Problema de Sequenciamento

Baseado no modelo **Li & Milne (2014)** para scheduling em máquinas paralelas com:
- Tempos de setup dependentes da sequência
- Custos de setup
- Restrições de capacidade

### Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `sequenciamento.py` | Ambiente RKO com decoder Best-Fit |
| `benchmark.py` | Solver exato com Gurobi (TSP por máquina) |
| `instancia.py` | Gerador de instâncias sintéticas |
| `verificar_sol.py` | Valida soluções do C++ no Python |

---

## 👤 Maintainer

**Felipe Silvestre Cardoso Roberto**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/felipesilvestrecr/)

---

## 📄 License

MIT License
