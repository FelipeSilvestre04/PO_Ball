"""
Environment para RKO - Lot-Sizing and Scheduling Problem
"""
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from rko import RKO


def carregar_instancia(instance_path: str) -> dict:
    """
    Carrega uma instância de um arquivo .txt gerado pelo instance_generator.py
    
    Args:
        instance_path: Caminho para o arquivo .txt da instância
        
    Returns:
        dict: Dicionário com todos os dados necessários para rodar o modelo
    """
    with open(instance_path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f.readlines()]
    
    # Parse sections
    section = None
    params = {}
    skus = []
    prod_need_data = []
    prod_rate_data = []
    setup_data = []
    
    for line in lines:
        if not line or line.startswith('#'):
            if '# PARAMETERS' in line:
                section = 'params'
            elif '# SKUS' in line:
                section = 'skus'
            elif '# PRODUCTION_NEED' in line:
                section = 'prod_need'
            elif '# PRODUCTION_RATE' in line:
                section = 'prod_rate'
            elif '# SETUP_MATRIX' in line:
                section = 'setup'
            continue
        
        if section == 'params':
            key, value = line.split('=', 1)
            params[key.strip()] = value.strip()
        elif section == 'skus':
            skus.append(line)
        elif section == 'prod_need':
            parts = line.split(',')
            prod_need_data.append({
                'sku': parts[0],
                'deadline': datetime.fromisoformat(parts[1]),
                'prod_need': float(parts[2])
            })
        elif section == 'prod_rate':
            parts = line.split(',')
            prod_rate_data.append({
                'ref_date': datetime.fromisoformat(parts[0]),
                'prod_rate': float(parts[1])
            })
        elif section == 'setup':
            parts = line.split(',')
            setup_data.append({
                'sku_from': parts[0],
                'sku_to': parts[1],
                'setup_cost': float(parts[2]),
                'setup_time': float(parts[3]) if len(parts) > 3 else 60
            })
    
    # Build DataFrames
    prod_need = pd.DataFrame(prod_need_data)
    production_rate = pd.DataFrame(prod_rate_data)
    setup_matrix = pd.DataFrame(setup_data)
    
    # Parse horizon
    horizon_start = datetime.fromisoformat(params['horizon_start'])
    horizon_days = int(params['horizon_days'])
    last_demand_date = horizon_start + timedelta(days=horizon_days)
    
    # Get instance name
    instance_name = Path(instance_path).stem
    n_skus = len(skus)
    n_periods = horizon_days * (24 // int(params['planning_slot_hours']))
    
    print(f"\n📂 Instância: {instance_name}")
    print(f"   📦 SKUs: {n_skus}, 📅 Períodos: {n_periods}")
    
    return {
        'instance_name': instance_name,
        'produtos': skus,
        'n_skus': n_skus,
        'n_periods': n_periods,
        'initial_setup': params['initial_setup'],
        'horizon_start': horizon_start,
        'horizon_days': horizon_days,
        'last_demand_date': last_demand_date,
        'planning_hours_slot': int(params['planning_slot_hours']),
        'prod_need': prod_need,
        'production_rate': production_rate,
        'setup_matrix': setup_matrix,
        'min_setup_time': 30,
        'max_setup_time': 120,
    }


class BallEnv:
    """
    Environment para o problema de Lot-Sizing and Scheduling.
    
    Cada chave do RKO representa uma decisão de produção/setup.
    O decoder interpreta as chaves e constrói uma solução.
    """

    def __init__(self, instance_path: str):
        """
        Inicializa o environment carregando a instância.
        
        Args:
            instance_path: Caminho para o arquivo .txt da instância
        """
        # Carregar dados da instância
        self.instance_data = carregar_instancia(instance_path)
        self.instance_name = self.instance_data['instance_name']
        
        # Extrair dados principais
        self.n_skus = self.instance_data['n_skus']
        self.n_periods = self.instance_data['n_periods']
        self.produtos = self.instance_data['produtos']
        self.initial_setup = self.instance_data['initial_setup']
        
        # Construir matriz de setup como dict para acesso rápido
        self.setup_cost = {}
        self.setup_time = {}
        for _, row in self.instance_data['setup_matrix'].iterrows():
            key = (row['sku_from'], row['sku_to'])
            self.setup_cost[key] = row['setup_cost']
            self.setup_time[key] = row['setup_time']
        
        # Demanda por período
        self.demand = self._build_demand_dict()
        print(self.demand)
        
        # Tamanho da solução: produção de cada SKU em cada período
        # Formato: [X_s0_t0, X_s0_t1, ..., X_s0_tn, X_s1_t0, ..., X_sn_tn]
        self.tam_solution = self.n_skus * self.n_periods
        
        # Parâmetros de otimização
        self.max_time = 300
        self.LS_type = 'Best'
        self.dict_best = {}
        
        # Parâmetros dos algoritmos
        self._init_algorithm_parameters()
    
    def _gen_periods(self) -> list:
        """
        Gera lista de períodos com datas de início e fim.
        Mesma lógica de line_scheduling_v1._gen_periods
        """
        horizon_start = self.instance_data['horizon_start']
        planning_hours_slot = self.instance_data['planning_hours_slot']
        last_demand_date = self.instance_data['last_demand_date']
        
        # T0 é período fora do horizonte
        periods = [{
            "label": "T0",
            "start_date": None,
            "end_date": None,
            "period_idx": 0
        }]
        
        # Ajustar datas
        horizon_start = horizon_start.replace(minute=0, second=0, microsecond=0)
        last_demand_date = (last_demand_date + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        
        period_start = horizon_start
        period_idx = 1
        
        while period_start < last_demand_date:
            period_end = min(
                period_start + timedelta(hours=planning_hours_slot),
                last_demand_date
            )
            
            periods.append({
                "label": f"T{period_idx}",
                "start_date": period_start,
                "end_date": period_end,
                "period_idx": period_idx
            })
            
            period_start = period_end
            period_idx += 1
        
        return periods
    
    def _get_related_period_label(self, date: datetime) -> str:
        """
        Dado uma data, retorna o label do período que contém essa data.
        Mesma lógica de line_scheduling_v1._get_related_period_label
        """
        for period in self.periods:
            if period["start_date"] is not None:
                if period["start_date"] <= date <= period["end_date"]:
                    return period["label"]
        
        # Se não encontrar, retorna último período
        return self.periods[-1]["label"]
    
    def _build_demand_dict(self) -> dict:
        """
        Constrói dicionário de demanda por (sku, período).
        Mesma lógica do modelo matemático:
        1. Mapeia cada deadline -> período correspondente
        2. Agrupa demandas por (sku, período)
        """
        # Gerar períodos com datas
        self.periods = self._gen_periods()
        
        # Criar lookup de período por label
        self.periods_by_label = {p["label"]: p for p in self.periods}
        
        # Inicializar demanda zerada para todos (sku, período)
        demand = {}
        for sku in self.produtos:
            for period in self.periods[1:]:  # Ignora T0
                demand[(sku, period["period_idx"])] = 0
        
        # Mapear cada demanda para o período correto
        prod_need = self.instance_data['prod_need']
        
        for _, row in prod_need.iterrows():
            sku = row['sku']
            deadline = row['deadline']
            qty = row['prod_need']
            
            # Encontrar período para este deadline
            period_label = self._get_related_period_label(deadline)
            period_idx = self.periods_by_label[period_label]["period_idx"]
            
            # Agregar demanda (somar se já existir)
            if sku in self.produtos:
                key = (sku, period_idx)
                if key in demand:
                    demand[key] += qty
                else:
                    demand[key] = qty
        
        # Log do mapeamento
        total_demand = sum(demand.values())
        non_zero = sum(1 for v in demand.values() if v > 0)
        print(f"   📊 Demanda total: {total_demand:.0f}, Entradas não-zero: {non_zero}")
        
        return demand
    
    def _init_algorithm_parameters(self):
        """Inicializa parâmetros dos algoritmos metaheurísticos."""
        self.BRKGA_parameters = {
            'p': [100],      
            'pe': [0.20],    
            'pm': [0.10],    
            'rhoe': [0.70]   
        }
        self.SA_parameters = {
            'SAmax': [50],     
            'alphaSA': [0.99], 
            'betaMin': [0.05], 
            'betaMax': [0.25], 
            'T0': [10000]      
        }
        self.ILS_parameters = {
            'betaMin': [0.10],
            'betaMax': [0.20]
        }
        self.VNS_parameters = {
            'kMax': [5],
            'betaMin': [0.05]
        }
        self.PSO_parameters = {
            'PSize': [100],    
            'c1': [2.05],      
            'c2': [2.05],      
            'w': [0.73]        
        }
        self.GA_parameters = {
            'sizePop': [100],
            'probCros': [0.98],
            'probMut': [0.005]
        }
        self.LNS_parameters = {
            'betaMin': [0.10],
            'betaMax': [0.30],
            'TO': [1000],
            'alphaLNS': [0.95]
        }
    
    def decoder(self, keys: np.ndarray) -> dict:
        """
        Decodifica um vetor de chaves aleatórias em uma solução.
        
        Args:
            keys: Array de valores em [0, 1] de tamanho tam_solution
            
        Returns:
            dict: Solução com produção, inventário, backlog e setups
        """
       
        production = np.zeros((self.n_skus, self.n_periods))
        inventory = np.zeros((self.n_skus, self.n_periods))
        backlog = np.zeros((self.n_skus, self.n_periods))
        setups = []
        

        slots_per_day = 24 // self.instance_data['planning_hours_slot']
        
        prod_rate_df = self.instance_data['production_rate']
        if not prod_rate_df.empty:
            avg_daily_rate = prod_rate_df['prod_rate'].mean()
            capacity_per_period = avg_daily_rate / slots_per_day
        else:
            capacity_per_period = 2000 / slots_per_day  
        

        
        return {
            'production': production,
            'inventory': inventory,
            'backlog': backlog,
            'setups': setups
        }
    
    def cost(self, solution: dict, final_solution: bool = False) -> float:
        """
        Calcula o custo de uma solução.
        
        Args:
            solution: Dicionário com produção, inventário, backlog e setups
            final_solution: Se True, calcula custos adicionais
            
        Returns:
            float: Custo total (menor é melhor)
        """
        # Pesos da função objetivo
        BACKLOG_WEIGHT = 0.01
        INVENTORY_WEIGHT = 0.0001
        SETUP_WEIGHT = 1.0
        
        # Custo de backlog
        backlog_cost = BACKLOG_WEIGHT * solution['backlog'].sum()
        
        # Custo de inventário
        inventory_cost = INVENTORY_WEIGHT * solution['inventory'].sum()
        
        # Custo de setup
        setup_cost = 0
        for setup in solution['setups']:
            key = (setup['from'], setup['to'])
            setup_cost += self.setup_cost.get(key, 0.5) * SETUP_WEIGHT
        
        total = backlog_cost + inventory_cost + setup_cost
        
        return total


# =============================================================================
# EXECUÇÃO COM MESMAS INSTÂNCIAS DO MODELO MATEMÁTICO
# =============================================================================
if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    # Importar RKO
    from rko import RKO
    
    # Diretórios
    base_dir = Path(__file__).parent.parent
    instances_dir = base_dir / "instances"
    images_dir = base_dir / "images_rko"
    images_dir.mkdir(exist_ok=True)
    
    # Mesmas configurações do modelo matemático
    skus = [4, 6, 8]
    days = [7, 14, 21]
    density = [100, 150]
    setup_types = ['random', 'proportional', 'uniform']
    
    # Lista para armazenar resultados
    resultados_csv = []
    
    print("="*60)
    print("🚀 RKO - EXECUTANDO MESMAS INSTÂNCIAS DO MODELO MATEMÁTICO")
    print("="*60)
    
    for sku in skus:
        for day in days:
            for den in density:
                for setup_type in setup_types:
                    # Montar nome do arquivo
                    instance_name = f"inst_{sku}skus_{day}days_{setup_type}_d{den}.txt"
                    instance_file = instances_dir / instance_name
                    
                    if instance_file.exists():
                        print(f"\n{'─'*50}")
                        print(f"📂 Processando: {instance_name}")
                        
                        # Criar environment
                        env = BallEnv(str(instance_file))

                        while True:
                            pass
                        
                        # Executar RKO
                        # solver = RKO(env, True)
                        # solver.solve(
                        #     time_total=300, 
                        #     brkga=1, ms=1, sa=1, vns=1, ils=1, lns=1, pso=1, ga=1, 
                        #     restart=0.5, runs=1
                        # )
                        
                        # Obter melhor solução
                        best_keys = solver.best_solution
                        best_cost = solver.best_cost
                        tempo = solver.elapsed_time if hasattr(solver, 'elapsed_time') else 300
                        
                        # Adicionar ao CSV
                        resultados_csv.append({
                            'instancia': instance_name,
                            'fo': best_cost,
                            'tempo_s': tempo,
                            'metodo': 'RKO'
                        })
                        
                        # Gerar visualização
                        if best_keys is not None:
                            solution = env.decoder(best_keys)
                            output_name = images_dir / f"{instance_name.replace('.txt', '_rko.png')}"
                            
                            # Criar figura simples
                            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
                            
                            # Produção
                            ax1 = axes[0]
                            for s, sku_name in enumerate(env.produtos):
                                ax1.plot(range(env.n_periods), solution['production'][s, :], 
                                        label=sku_name[:10], marker='o', markersize=3)
                            ax1.set_title(f'Produção por Período - {instance_name} (FO={best_cost:.4f})')
                            ax1.set_xlabel('Período')
                            ax1.set_ylabel('Quantidade')
                            ax1.legend(loc='upper right', fontsize=8)
                            ax1.grid(True, alpha=0.3)
                            
                            # Inventário/Backlog
                            ax2 = axes[1]
                            for s, sku_name in enumerate(env.produtos):
                                ax2.bar(np.arange(env.n_periods) - 0.2 + s*0.1, 
                                       solution['inventory'][s, :], width=0.1, alpha=0.7, label=f'{sku_name[:8]} Inv')
                                ax2.bar(np.arange(env.n_periods) - 0.2 + s*0.1, 
                                       -solution['backlog'][s, :], width=0.1, alpha=0.4, hatch='//')
                            ax2.axhline(y=0, color='black', linewidth=1)
                            ax2.set_title('Inventário (+) e Backlog (-)')
                            ax2.set_xlabel('Período')
                            ax2.set_ylabel('Quantidade')
                            ax2.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            plt.savefig(output_name, dpi=150, bbox_inches='tight', facecolor='white')
                            plt.close()
                            
                            print(f"   📊 Imagem salva: {output_name.name}")
                        
                        print(f"   ✅ FO: {best_cost:.4f}")
                    else:
                        print(f"\n⚠️ Não encontrado: {instance_name}")
    
    # Salvar CSV
    csv_path = base_dir / "resultados_rko.csv"
    df_resultados = pd.DataFrame(resultados_csv)
    df_resultados.to_csv(csv_path, index=False)
    
    print("\n" + "="*60)
    print(f"📊 Resultados salvos em: {csv_path}")
    print("="*60)
    print(df_resultados.to_string())
