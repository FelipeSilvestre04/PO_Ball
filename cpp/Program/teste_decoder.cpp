#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>

// ==========================================
// 1. ESTRUTURAS (Mocks do RKO)
// ==========================================
struct TSol {
    std::vector<double> rk;
    double ofv;
};

struct TProblemData {
    int n;
    int num_products;
    int num_machines;
    std::vector<double> machine_capacities;
    std::vector<int> initial_state;
    std::vector<double> demands;
    std::vector<double> production_rates;
    std::vector<double> setup_costs;
    std::vector<double> setup_times;
};

// ==========================================
// 2. LEITURA DE DADOS (Cópia do Problem.h)
// ==========================================
void ReadData(std::string name, TProblemData &data) {
    std::ifstream file(name);
    if (!file.is_open()) {
        std::cerr << "ERRO: Arquivo (" << name << ") nao encontrado!" << std::endl;
        exit(1);
    }

    file >> data.num_products >> data.num_machines;
    data.n = data.num_products;

    data.machine_capacities.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.machine_capacities[i];

    data.initial_state.resize(data.num_machines);
    for(int i=0; i<data.num_machines; ++i) file >> data.initial_state[i];

    data.demands.resize(data.num_products);
    for(int i=0; i<data.num_products; ++i) file >> data.demands[i];

    data.production_rates.resize(data.num_products * data.num_machines);
    for(int i=0; i<data.num_products; ++i) {
        for(int m=0; m<data.num_machines; ++m) {
            file >> data.production_rates[i * data.num_machines + m];
        }
    }

    data.setup_costs.resize(data.num_products * data.num_products * data.num_machines);
    std::string label;
    for(int m=0; m<data.num_machines; ++m) {
        file >> label; 
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_costs[idx];
            }
        }
    }

    data.setup_times.resize(data.num_products * data.num_products * data.num_machines);
    for(int m=0; m<data.num_machines; ++m) {
        file >> label;
        for(int i=0; i<data.num_products; ++i) {
            for(int j=0; j<data.num_products; ++j) {
                int idx = (i * data.num_products * data.num_machines) + (j * data.num_machines) + m;
                file >> data.setup_times[idx];
            }
        }
    }
    file.close();
    std::cout << ">> Instancia carregada: " << data.num_products << " produtos, " << data.num_machines << " maquinas." << std::endl;
}

// ==========================================
// 3. DECODER (BEST FIT / CHEAPEST INSERTION)
// ==========================================
// (Certifique-se de que esta é a mesma lógica do seu Problem.h)
double Decoder(TSol &s, const TProblemData &data)
{
    std::vector<int> sorted_products(data.n);
    std::iota(sorted_products.begin(), sorted_products.end(), 0);
    std::sort(sorted_products.begin(), sorted_products.end(), [&](int i, int j) {
        return s.rk[i] < s.rk[j];
    });

    std::vector<std::vector<int>> machine_seqs(data.num_machines);
    std::vector<double> machine_loads(data.num_machines, 0.0);
    double total_setup_cost = 0.0;
    double penalty = 0.0;

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
            total_setup_cost += min_delta_cost;
        } else {
            penalty += 100000.0 + (data.demands[product_idx] * 1000.0);
        }
    }
    
    return total_setup_cost + penalty;
}

// ==========================================
// 4. MAIN - TESTE DE PERFORMANCE POR TEMPO
// ==========================================
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: ./test_decoder [caminho_instancia] [tempo_segundos]" << std::endl;
        return 1;
    }

    std::string instance_path = argv[1];
    double duration_limit = (argc >= 3) ? std::atof(argv[2]) : 60.0; // Default 60s

    // 1. Carregar Dados
    TProblemData data;
    ReadData(instance_path, data);

    // 2. Preparar Gerador
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    TSol sol;
    sol.rk.resize(data.n);

    // 3. Loop de Benchmark
    unsigned long long evaluations = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    double current_cost = 0;

    std::cout << "\n>> Iniciando Teste de Throughput (" << duration_limit << " segundos)..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    while (true) {
        // Gera chaves novas e avalia
        for(int i=0; i<data.n; ++i) sol.rk[i] = dist(rng);
        current_cost = Decoder(sol, data);
        
        if (current_cost < best_cost) {
            best_cost = current_cost;
        }
        
        evaluations++;

        // Checa o tempo a cada X iterações para não pesar
        if (evaluations % 100 == 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = current_time - start_time;
            if (elapsed.count() >= duration_limit) break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double total_time = elapsed_seconds.count();

    // 4. Relatório
    double evals_per_sec = evaluations / total_time;

    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "TEMPO TOTAL:      " << std::fixed << std::setprecision(2) << total_time << " s" << std::endl;
    std::cout << "TOTAL AVALIACOES: " << evaluations << std::endl;
    std::cout << "VELOCIDADE:       " << (int)evals_per_sec << " sol/s" << std::endl;
    std::cout << "MELHOR CUSTO:     " << std::fixed << std::setprecision(2) << best_cost << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}