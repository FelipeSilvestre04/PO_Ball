#ifndef _PROBLEM_H
#define _PROBLEM_H

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

// 1. ESTRUTURA DE DADOS (Igual ao seu __init__ no Python)
struct TProblemData
{
    int n; // Tamanho do vetor de chaves (Obrigatório pelo RKO)

    int num_products;
    int num_machines;
    
    std::vector<double> machine_capacities; // T_r
    std::vector<int> initial_state;         // P0
    std::vector<double> demands;            // d_i
    
    // Matrizes linearizadas (vetor 1D) para performance
    // Acesso: [prod * num_machines + machine]
    std::vector<double> production_rates; 
    
    // Acesso: [prev_prod * num_products * num_machines + curr_prod * num_machines + machine]
    // Ou melhor: vetor de vetores para facilitar leitura, mas linear é mais rápido.
    // Vamos usar linearização: index = (prev * num_prod * num_mach) + (curr * num_mach) + mach
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
};

// 2. LEITURA DE DADOS
void ReadData(char name[], TProblemData &data)
{
    std::ifstream file(name);
    if (!file.is_open()) {
        printf("\nERROR: File (%s) not found!\n", name);
        exit(1);
    }

    // Lê dimensões
    file >> data.num_products >> data.num_machines;
    data.n = data.num_products; // Tamanho do cromossomo

    // Lê capacidades
    data.machine_capacities.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.machine_capacities[i];

    // Lê estado inicial
    data.initial_state.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.initial_state[i];

    // Lê demandas
    data.demands.resize(data.num_products);
    for(int i=0; i<data.num_products; ++i) file >> data.demands[i];

    // Lê taxas de produção
    data.production_rates.resize(data.num_products * data.num_machines);
    for(int i=0; i<data.num_products; ++i) {
        for(int m=0; m<data.num_machines; ++m) {
            file >> data.production_rates[i * data.num_machines + m];
        }
    }

    // Lê custos de setup
    // Formato Li & Milne: Bloco por máquina
    data.setup_costs.resize(data.num_products * data.num_products * data.num_machines);
    std::string label;
    for(int m=0; m<data.num_machines; ++m) {
        file >> label; // "Reactor_Cost_X"
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                // Indexação: [i][j][m]
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_costs[idx];
            }
        }
    }

    // Lê tempos de setup
    data.setup_times.resize(data.num_products * data.num_products * data.num_machines);
    for(int m=0; m<data.num_machines; ++m) {
        file >> label; // "Reactor_Time_X"
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_times[idx];
            }
        }
    }
    
    file.close();
}

// 3. DECODER (Cheapest Insertion / Best Fit)
double Decoder(TSol &s, const TProblemData &data)
{
    // 1. Ordena produtos pelas chaves (argsort)
    std::vector<int> sorted_products(data.n);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    // Estruturas da solução
    // Usamos vector de vectors para as sequências
    std::vector<std::vector<int>> machine_seqs(data.num_machines);
    std::vector<double> machine_loads(data.num_machines, 0.0);
    double total_setup_cost = 0.0;
    double penalty = 0.0;

    // 2. Alocação Gulosa
    for (int product_idx : sorted_products) {
        int best_machine = -1;
        int best_pos = -1;
        double min_delta_cost = std::numeric_limits<double>::infinity();
        double best_time_increase = 0.0;

        // Testa todas as máquinas
        for (int m = 0; m < data.num_machines; ++m) {
            double rate = data.production_rates[product_idx * data.num_machines + m];
            if (rate <= 1e-6) continue;

            double prod_time = data.demands[product_idx] / rate;
            const auto& current_seq = machine_seqs[m];
            int seq_len = current_seq.size();

            // Testa todas as posições (0 a seq_len)
            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev_prod = (pos == 0) ? data.initial_state[m] : current_seq[pos - 1];
                int next_prod = (pos < seq_len) ? current_seq[pos] : -1;

                // Acesso linearizado aos custos/tempos: [i][j][m]
                // Index = (i * num_prods * num_mach) + (j * num_mach) + m
                
                // Delta Custo
                int idx_prev_curr = (prev_prod * data.num_products * data.num_machines) + (product_idx * data.num_machines) + m;
                double cost_add = data.setup_costs[idx_prev_curr];
                double cost_rem = 0.0;

                if (next_prod != -1) {
                    int idx_curr_next = (product_idx * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    int idx_prev_next = (prev_prod * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    cost_add += data.setup_costs[idx_curr_next];
                    cost_rem = data.setup_costs[idx_prev_next];
                }
                double delta_cost = cost_add - cost_rem;

                // Delta Tempo
                int idx_time_prev_curr = (prev_prod * data.num_products * data.num_machines) + (product_idx * data.num_machines) + m;
                double time_add = data.setup_times[idx_time_prev_curr];
                double time_rem = 0.0;

                if (next_prod != -1) {
                    int idx_time_curr_next = (product_idx * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    int idx_time_prev_next = (prev_prod * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    time_add += data.setup_times[idx_time_curr_next];
                    time_rem = data.setup_times[idx_time_prev_next];
                }
                double delta_time = time_add + prod_time - time_rem;

                // Verifica Capacidade
                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < min_delta_cost) {
                        min_delta_cost = delta_cost;
                        best_machine = m;
                        best_pos = pos;
                        best_time_increase = delta_time;
                    }
                    // Desempate por taxa (opcional, se quiser idêntico ao Python)
                    else if (std::abs(delta_cost - min_delta_cost) < 1e-6) {
                        if (best_machine != -1) {
                            double curr_rate = data.production_rates[product_idx * data.num_machines + best_machine];
                            if (rate > curr_rate) {
                                best_machine = m;
                                best_pos = pos;
                                best_time_increase = delta_time;
                            }
                        }
                    }
                }
            }
        }

        // Realiza Alocação
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(machine_seqs[best_machine].begin() + best_pos, product_idx);
            machine_loads[best_machine] += best_time_increase;
            total_setup_cost += min_delta_cost;
        } else {
            // Penalidade
            penalty += 100000.0 + (data.demands[product_idx] * 1000.0);
        }
    }
    
    return total_setup_cost + penalty;
}

void FreeMemoryProblem(TProblemData &data)
{
    data.machine_capacities.clear();
    data.initial_state.clear();
    data.demands.clear();
    data.production_rates.clear();
    data.setup_costs.clear();
    data.setup_times.clear();
}

// Função auxiliar para imprimir a solução na tela (C++ -> Python check)
void PrintSolution(TSol &s, const TProblemData &data)
{
    // 1. Reconstrói a solução (mesma lógica do Decoder)
    std::vector<int> sorted_products(data.n);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    std::vector<std::vector<int>> machine_seqs(data.num_machines);
    std::vector<double> machine_loads(data.num_machines, 0.0);

    for (int product_idx : sorted_products) {
        int best_machine = -1;
        int best_pos = -1;
        double min_delta_cost = std::numeric_limits<double>::infinity();
        double best_time_increase = 0.0;

        for (int m = 0; m < data.num_machines; ++m) {
            double rate = data.production_rates[product_idx * data.num_machines + m];
            if (rate <= 1e-6) continue;
            double prod_time = data.demands[product_idx] / rate;
            const auto& current_seq = machine_seqs[m];
            int seq_len = current_seq.size();

            for (int pos = 0; pos <= seq_len; ++pos) {
                int prev_prod = (pos == 0) ? data.initial_state[m] : current_seq[pos - 1];
                int next_prod = (pos < seq_len) ? current_seq[pos] : -1;

                int idx_prev_curr = (prev_prod * data.num_products * data.num_machines) + (product_idx * data.num_machines) + m;
                double cost_add = data.setup_costs[idx_prev_curr];
                double cost_rem = 0.0;

                if (next_prod != -1) {
                    int idx_curr_next = (product_idx * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    int idx_prev_next = (prev_prod * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    cost_add += data.setup_costs[idx_curr_next];
                    cost_rem = data.setup_costs[idx_prev_next];
                }
                double delta_cost = cost_add - cost_rem;

                int idx_time_prev_curr = (prev_prod * data.num_products * data.num_machines) + (product_idx * data.num_machines) + m;
                double time_add = data.setup_times[idx_time_prev_curr];
                double time_rem = 0.0;

                if (next_prod != -1) {
                    int idx_time_curr_next = (product_idx * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    int idx_time_prev_next = (prev_prod * data.num_products * data.num_machines) + (next_prod * data.num_machines) + m;
                    time_add += data.setup_times[idx_time_curr_next];
                    time_rem = data.setup_times[idx_time_prev_next];
                }
                double delta_time = time_add + prod_time - time_rem;

                if (machine_loads[m] + delta_time <= data.machine_capacities[m]) {
                    if (delta_cost < min_delta_cost) {
                        min_delta_cost = delta_cost;
                        best_machine = m;
                        best_pos = pos;
                        best_time_increase = delta_time;
                    }
                    else if (std::abs(delta_cost - min_delta_cost) < 1e-6) {
                        if (best_machine != -1) {
                            double curr_rate = data.production_rates[product_idx * data.num_machines + best_machine];
                            if (rate > curr_rate) {
                                best_machine = m;
                                best_pos = pos;
                                best_time_increase = delta_time;
                            }
                        }
                    }
                }
            }
        }
        if (best_machine != -1) {
            machine_seqs[best_machine].insert(machine_seqs[best_machine].begin() + best_pos, product_idx);
            machine_loads[best_machine] += best_time_increase;
        }
    }

    // 2. IMPRIME A SEQUENCIA FORMATADA (Python-friendly)
    printf("\n=== SEQUENCE_START ===\n");
    printf("[");
    for (int m = 0; m < data.num_machines; ++m) {
        printf("[");
        for (size_t i = 0; i < machine_seqs[m].size(); ++i) {
            printf("%d", machine_seqs[m][i]);
            if (i < machine_seqs[m].size() - 1) printf(", ");
        }
        printf("]");
        if (m < data.num_machines - 1) printf(", ");
    }
    printf("]\n");
    printf("=== SEQUENCE_END ===\n");
}

#endif