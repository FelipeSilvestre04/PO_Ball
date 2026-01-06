# DOCUMENTAÇÃO TÉCNICA: MODELO DE SEQUENCIAMENTO DE LINHA (V1.0)

---

## 1. VISÃO GERAL DO PROBLEMA

O modelo implementado (**LineSchedulingV1**) resolve um problema de **Dimensionamento de Lotes e Sequenciamento** (Lot-Sizing and Scheduling) monomáquina, com as seguintes características principais:

| Característica | Descrição |
|----------------|-----------|
| **Setup Dependente da Sequência** | O custo e o tempo de troca dependem do produto anterior (j) e do próximo (i) |
| **Horizonte de Planejamento Discreto** | O tempo é dividido em períodos (t), que podem ser buckets pequenos ou grandes |
| **Conservação de Estado (Setup Carry-Over)** | Se o produto i termina o período t-1, a máquina pode continuar produzindo i em t sem novo setup |
| **Backlog e Inventário** | A demanda pode ser atendida por produção no período, estoque (inventário) ou postergada (backlog com penalidade) |

> [!NOTE]
> Este modelo é uma variação do **GLSP** (General Lot-Sizing and Scheduling Problem), formulado para garantir que, em cada microperíodo, ocorra no máximo uma troca de configuração (setup).

---

## 2. FORMULAÇÃO MATEMÁTICA

A implementação no PuLP segue rigorosamente a formulação descrita no PDF *BPSA Production Scheduling - Model 0.3.pdf*.

### 2.1 Conjuntos e Índices

| Símbolo | Descrição |
|---------|-----------|
| `i, j ∈ P` | Conjunto de produtos (SKUs) |
| `t ∈ T` | Conjunto de períodos de planejamento (T₁, ..., Tₘ). O período T₀ é um período "fantasma" usado para condições iniciais |

### 2.2 Parâmetros (Dados de Entrada)

| Parâmetro | Descrição |
|-----------|-----------|
| `Dᵢₜ` | Demanda do produto i no período t |
| `Cₜ` | Capacidade de tempo disponível no período t (em minutos) |
| `Tₜ` | Tempo para produzir 1000 unidades (inverso da taxa de produção) |
| `SCᵢⱼ` | Custo de setup para trocar de j para i |
| `STᵢⱼ` | Tempo de setup (downtime) para trocar de j para i |

### 2.3 Variáveis de Decisão

#### Contínuas:

| Variável | Descrição |
|----------|-----------|
| `Xᵢₜ ≥ 0` | Quantidade produzida do produto i no período t |
| `Iᵢₜ ≥ 0` | Nível de estoque do produto i ao final de t |
| `Bᵢₜ ≥ 0` | Nível de backlog (demanda não atendida) do produto i em t |

#### Binárias:

| Variável | Descrição |
|----------|-----------|
| `Zᵢⱼₜ ∈ {0,1}` | **Variável de Setup.** Indica se houve uma troca PARA o produto i, vindo DO produto j no período t |
| `ZAᵢₜ ∈ {0,1}` | **Variável de Setup Carry-Over.** Indica se a máquina "trouxe" a configuração do produto i do período anterior |

### 2.4 Função Objetivo

O objetivo é **minimizar** uma soma ponderada de custos:

```
min Σᵢₜ (W_backlog · Bᵢₜ + W_inv · Iᵢₜ) + Σᵢⱼₜ (SCᵢⱼ · Zᵢⱼₜ)
```

> [!TIP]
> Na implementação, a função objetivo também penaliza o número de setups (multiplicando o custo por 2) para evitar trocas excessivas que não tragam benefício real.

---

## 3. RESTRIÇÕES (O Núcleo do Modelo)

### 3.1 Balanceamento de Massa (Restrição 1)

Garante que: **Demanda = Produção + Estoque Anterior - Estoque Atual + Variação de Backlog**

```
Xᵢₜ + Iᵢ₍ₜ₋₁₎ - Iᵢₜ + Bᵢₜ - Bᵢ₍ₜ₋₁₎ = Dᵢₜ
```

---

### 3.2 Lógica de Setup e Produção (Restrições 2 e 3)

Só é permitido produzir i se houver um setup para i (Z) ou se a configuração já estava em i (ZA):

```
Xᵢₜ ≤ M · (Σⱼ Zᵢⱼₜ + ZAᵢₜ)
```

Forçamos que um setup só ocorra se houver produção em t ou t+1:

```
Σⱼ Zᵢⱼₜ ≤ Xᵢₜ + Xᵢ₍ₜ₊₁₎
```

---

### 3.3 Capacidade (Restrição 4)

O tempo total gasto (Produção + Tempos de Setup) não pode exceder a capacidade do período:

```
Σᵢ (Xᵢₜ · Tₜ) + Σᵢⱼ (Zᵢⱼₜ · STⱼᵢ) ≤ Cₜ
```

---

### 3.4 Controle de Sequenciamento (GLSP - Restrições 5 a 9)

Esta é a parte crítica que gerencia a sequência `j → i`:

| Restrição | Fórmula | Descrição |
|-----------|---------|-----------|
| **Único Setup por Período** | `Σᵢⱼ Zᵢⱼₜ ≤ 1` | No máximo uma troca ocorre por período |
| **Único Estado (Carry-Over)** | `Σᵢ ZAᵢₜ ≤ 1` | A máquina só pode estar configurada para um produto no início do período |
| **Conexão de Setup** | `Zᵢⱼₜ ≤ ZAⱼₜ` | Se trocamos de j para i no período t, o estado da máquina devia ser j |

#### Continuidade do Estado (Linking Constraints):

Para manter o estado i em t (`ZAᵢₜ=1`), a máquina deve ter terminado t-1 no estado i. Isso ocorre se:
- Houve um setup para i em t-1, **OU**
- Já estava em i (carry-over) em t-1

---

### 3.5 Condições Iniciais (Restrições 11 e 12)

O modelo fixa as variáveis no período T₀ para zero, exceto o setup inicial:

1. Lê-se o produto inicial da linha (ex: `CAABSKP...` no arquivo CSV)
2. Define-se `ZAₖ₀ = 1` para esse produto k e `0` para os outros

> [!IMPORTANT]
> Isso força que o primeiro setup do horizonte de planejamento deva ser compatível com o produto que já está na máquina.

---

## 4. ESTRUTURA DAS INSTÂNCIAS (INPUTS)

O sistema ingere dados de planilhas CSV para popular os parâmetros do modelo:

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARQUIVOS DE ENTRADA                          │
├─────────────────────────────────────────────────────────────────┤
│  Production Need.csv  ──────►  Demanda (Dᵢₜ)                    │
│  Production Rate.csv  ──────►  Capacidade (Cₜ) e Tempo (Tₜ)     │
│  Initial Setup.csv    ──────►  Estado Inicial (ZAₖ₀)            │
│  Setup Matrix         ──────►  Custos (SCᵢⱼ) e Tempos (STᵢⱼ)    │
└─────────────────────────────────────────────────────────────────┘
```

### A. Production Need (`...Production Need.csv`)

| Campo | Descrição |
|-------|-----------|
| **Conteúdo** | Tabela com colunas de datas (ex: 2025-10-31, 2025-11-01) e linhas por SKU |
| **Mapeamento** | O código lê essas datas e as agrupa em "Períodos" (T₁, T₂...) com base no parâmetro `planning_hours_slot` |
| **No Modelo** | Preenche o parâmetro `Dᵢₜ` (Demanda) |

---

### B. Production Rate (`...Production Rate.csv`)

| Campo | Descrição |
|-------|-----------|
| **Conteúdo** | Capacidade da linha (ex: 2000 latas/minuto) por dia |
| **Mapeamento** | O código converte essa taxa para "Tempo para produzir 1000 unidades" |
| **No Modelo** | Define o parâmetro `Tₜ` e `Cₜ` (Capacidade total em minutos do período) |
| **Cálculo** | `Tₜ = 1 / Rate` |

---

### C. Initial Setup (`...Initial Setup.csv`)

| Campo | Descrição |
|-------|-----------|
| **Conteúdo** | Indica qual SKU está na linha no momento zero |
| **No Modelo** | Usado para fixar a variável binária `ZAᵢ₀` (Restrição 12) |
| **Exemplo** | Se a linha começa com Produto A, então `ZA_A,0 = 1`. Produzir B no primeiro período exigirá setup `Z_B,A,1` |

---

### D. Setup Matrix (Implícito/Calculado)

O código possui funções `_gen_setup_cost_matrix` e `_gen_setup_time_matrix`:
- Lê custos de troca entre SKUs
- Se `sku_from != sku_to`, atribui custo e tempo
- Calcula o tempo baseado em multiplicadores definidos na configuração de execução

---

## 5. DETALHES DE IMPLEMENTAÇÃO (Python/PuLP)

### Geração de Períodos Dinâmicos

O método `_gen_periods` cria a discretização do tempo:
- Se você definir slots de 4 horas, ele quebra o horizonte de dias do CSV em blocos de 4h (T₁, T₂, ...)

> [!WARNING]
> Buckets menores aumentam a precisão do sequenciamento, mas aumentam drasticamente o número de variáveis binárias (Zᵢⱼₜ), tornando a resolução via Gurobi/CBC mais lenta.

### Solver

| Configuração | Valor |
|--------------|-------|
| **Solver primário** | Gurobi (com licença) |
| **Fallback** | CBC (PuLP default) |
| **MIPFocus** | 2 (foco em otimalidade) |
| **Heurísticas** | 0.5 (agressivas - prioriza solução factível rápida) |

### Variáveis Dummy/Segurança

Como o PuLP pode ter problemas com caracteres especiais em nomes de produtos, o código usa um mapa interno (`_secure_label`) para renomear produtos para `PROD1`, `PROD2`, etc. durante a otimização.

### Saída de Dados

O método `build_results` decodifica as variáveis do solver de volta para objetos de negócio (`ProductionItem`, `SetupItem`), facilitando a geração de relatórios ou gráficos de Gantt.

---

## RESUMO PARA O USUÁRIO

> [!IMPORTANT]
> Este modelo é uma ferramenta robusta para **planejamento tático de curto/médio prazo**.
> 
> **Pergunta que ele responde:** "Qual a sequência de produção que minimiza meus custos de troca e atrasos, respeitando a capacidade da linha e o estado atual da máquina?"

### Complexidade Computacional

```
O(n² · t) variáveis binárias
```

O ajuste do tamanho do período (`Cₜ`) é a **principal alavanca de performance**.

---

## FLUXO DO MODELO

```
                    ┌──────────────────┐
                    │  DADOS DE ENTRADA │
                    │  (Arquivos CSV)   │
                    └────────┬─────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │           PRÉ-PROCESSAMENTO            │
        │  • Geração de períodos (T₁, T₂, ...)   │
        │  • Agregação de demandas               │
        │  • Cálculo de matriz de setup          │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │        MODELO DE OTIMIZAÇÃO            │
        │  • Variáveis: X, I, B (contínuas)      │
        │  • Variáveis: Z, ZA (binárias)         │
        │  • Restrições: Balanço, Capacidade,    │
        │                Sequenciamento          │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │             SOLVER (MIP)               │
        │  • Gurobi (primário)                   │
        │  • CBC (fallback)                      │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │         SAÍDA / RESULTADOS             │
        │  • Sequência ótima de produção         │
        │  • Níveis de estoque/backlog           │
        │  • Gráficos de Gantt                   │
        └────────────────────────────────────────┘
```
