"""
Gerador Sistemático de Instâncias para Line Scheduling.

Este script lê o arquivo Excel real da Ball e gera variantes sistemáticas
com diferentes:
- Número de SKUs (2, 4, 6, 8)
- Horizonte de planejamento (7, 14, 21, 28 dias)
- Proporção de demanda (0.5, 1.0, 1.5)
- Tipos de matriz de setup (uniforme, proporcional, aleatória)

Gera arquivos .txt e .xlsx para cada instância.

Autor: Felipe Silvestre
Data: 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os


# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

EXCEL_FILE = "2025_10_28_BRFR Test (1).xlsx"
OUTPUT_DIR = Path("instances")

# Variações sistemáticas
N_SKUS_LIST = [2, 4, 6, 8, 10, 12, 14]           # Número de SKUs
HORIZONS_DAYS = [7, 14, 21, 28, 35, 42, 48]  # Dias de planejamento (até ~7 semanas)
DEMAND_SCALES = [0.5, 1.0, 1.5]      # Escala de demanda
SETUP_TYPES = ["uniform", "proportional", "random"]  # Tipo de matriz de setup

# Parâmetros fixos
PLANNING_SLOT_HOURS = 12  # Bucket de 12 horas
PRODUCTION_RATE_PER_DAY = 2000  # Unidades por dia (igual ao Excel)


# ============================================================================
# FUNÇÕES DE LEITURA DO EXCEL
# ============================================================================

def load_excel_data(excel_path: str) -> dict:
    """
    Carrega os dados do Excel da Ball.
    
    Returns:
        Dict com DataFrames: prod_need, prod_rate, initial_setup
    """
    xl = pd.ExcelFile(excel_path)
    
    # Production Need - formato wide (datas como colunas)
    prod_need_wide = pd.read_excel(xl, sheet_name="Production Need")
    
    # Identificar colunas de data (são datetime)
    date_cols = [c for c in prod_need_wide.columns if isinstance(c, datetime)]
    id_cols = ["Plant", "Line", "Customer Code", "Order Number", "SKU"]
    
    # Converter para formato long
    prod_need = prod_need_wide.melt(
        id_vars=id_cols,
        value_vars=date_cols,
        var_name="deadline",
        value_name="prod_need"
    )
    prod_need = prod_need[prod_need["prod_need"] > 0]  # Remover zeros
    prod_need = prod_need.rename(columns={"SKU": "sku"})
    
    # Production Rate - formato wide
    prod_rate_wide = pd.read_excel(xl, sheet_name="Production Rate")
    date_cols_rate = [c for c in prod_rate_wide.columns if isinstance(c, datetime)]
    
    prod_rate = prod_rate_wide.melt(
        id_vars=["Plant", "Line", "Size"],
        value_vars=date_cols_rate,
        var_name="ref_date",
        value_name="prod_rate"
    )
    
    # Initial Setup
    initial_setup = pd.read_excel(xl, sheet_name="Initial Setup")
    
    return {
        "prod_need": prod_need,
        "prod_rate": prod_rate,
        "initial_setup": initial_setup,
        "all_skus": prod_need["sku"].unique().tolist(),
        "all_dates": sorted(date_cols),
    }


# ============================================================================
# FUNÇÕES DE GERAÇÃO DE VARIANTES
# ============================================================================

def generate_setup_matrix(skus: list, setup_type: str, seed: int = 42) -> pd.DataFrame:
    """
    Gera matriz de custo/tempo de setup entre SKUs.
    
    Args:
        skus: Lista de SKUs
        setup_type: "uniform", "proportional", "random"
        seed: Semente para reprodutibilidade
    
    Returns:
        DataFrame com colunas [sku_from, sku_to, setup_cost, setup_time]
    """
    np.random.seed(seed)
    n = len(skus)
    
    setup_data = []
    for i, sku_from in enumerate(skus):
        for j, sku_to in enumerate(skus):
            if sku_from == sku_to:
                cost = 0.0
                time = 0
            else:
                if setup_type == "uniform":
                    # Custo uniforme para todos os setups
                    cost = 0.5
                    time = 60  # 60 minutos
                elif setup_type == "proportional":
                    # Custo proporcional à "distância" no índice (simula similaridade)
                    distance = abs(i - j)
                    cost = 0.2 + (distance / n) * 0.6  # 0.2 a 0.8
                    time = 30 + int((distance / n) * 90)  # 30 a 120 minutos
                else:  # random
                    cost = np.random.uniform(0.2, 0.8)
                    time = np.random.randint(30, 120)
            
            setup_data.append({
                "sku_from": sku_from,
                "sku_to": sku_to,
                "setup_cost": round(cost, 3),
                "setup_time": time
            })
    
    return pd.DataFrame(setup_data)


def generate_instance(
    base_data: dict,
    n_skus: int,
    horizon_days: int,
    demand_scale: float,
    setup_type: str,
    instance_id: int
) -> dict:
    """
    Gera uma instância específica baseada nos parâmetros.
    
    Returns:
        Dict com todos os dados da instância
    """
    # Selecionar SKUs (primeiros n_skus da lista original)
    selected_skus = base_data["all_skus"][:n_skus]
    
    # Selecionar datas do horizonte
    all_dates = base_data["all_dates"]
    horizon_start = all_dates[0]
    horizon_end = horizon_start + timedelta(days=horizon_days)
    selected_dates = [d for d in all_dates if horizon_start <= d < horizon_end]
    
    # Se não tiver datas suficientes, gerar novas
    if len(selected_dates) < horizon_days:
        selected_dates = [horizon_start + timedelta(days=i) for i in range(horizon_days)]
    
    # Filtrar demanda
    prod_need = base_data["prod_need"].copy()
    prod_need = prod_need[prod_need["sku"].isin(selected_skus)]
    prod_need = prod_need[prod_need["deadline"].isin(selected_dates) | 
                          (prod_need["deadline"] <= horizon_end)]
    
    # Se não tiver demanda suficiente, gerar demanda sintética
    if prod_need.empty or prod_need["prod_need"].sum() == 0:
        prod_need_data = []
        base_demand = 500  # Demanda base por SKU por dia
        for sku in selected_skus:
            for date in selected_dates:
                demand = int(base_demand * demand_scale * np.random.uniform(0.5, 1.5))
                if demand > 0:
                    prod_need_data.append({
                        "sku": sku,
                        "deadline": date,
                        "prod_need": demand
                    })
        prod_need = pd.DataFrame(prod_need_data)
    else:
        # Aplicar escala de demanda
        prod_need["prod_need"] = (prod_need["prod_need"] * demand_scale).astype(int)
    
    # Gerar taxa de produção
    prod_rate_data = []
    for date in selected_dates:
        prod_rate_data.append({
            "ref_date": date,
            "prod_rate": PRODUCTION_RATE_PER_DAY
        })
    prod_rate = pd.DataFrame(prod_rate_data)
    
    # Setup inicial (primeiro SKU)
    initial_setup = selected_skus[0]
    
    # Gerar matriz de setup
    setup_matrix = generate_setup_matrix(selected_skus, setup_type, seed=instance_id)
    
    # Nome da instância
    instance_name = f"inst_{n_skus}skus_{horizon_days}days_{setup_type}_d{int(demand_scale*100)}"
    
    return {
        "name": instance_name,
        "instance_id": instance_id,
        "n_skus": n_skus,
        "horizon_days": horizon_days,
        "demand_scale": demand_scale,
        "setup_type": setup_type,
        "skus": selected_skus,
        "horizon_start": horizon_start,
        "horizon_end": horizon_end,
        "planning_slot_hours": PLANNING_SLOT_HOURS,
        "prod_need": prod_need,
        "prod_rate": prod_rate,
        "initial_setup": initial_setup,
        "setup_matrix": setup_matrix,
        "stats": {
            "total_demand": prod_need["prod_need"].sum(),
            "n_periods": horizon_days * 2,  # 2 slots de 12h por dia
            "n_demand_entries": len(prod_need),
        }
    }


# ============================================================================
# FUNÇÕES DE EXPORTAÇÃO
# ============================================================================

def export_to_txt(instance: dict, output_dir: Path) -> Path:
    """
    Exporta instância para arquivo .txt no formato do modelo.
    
    Formato:
    # Linha 1: Comentário com nome da instância
    # n_skus, horizon_days, planning_slot_hours, initial_setup
    # SKUs (um por linha)
    # Demanda: sku, deadline, quantidade (uma entrada por linha)
    # Taxa de produção: data, taxa (uma entrada por linha)
    # Matriz de setup: sku_from, sku_to, custo, tempo
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{instance['name']}.txt"
    
    with open(filepath, "w", encoding="utf-8") as f:
        # Cabeçalho
        f.write(f"# Instance: {instance['name']}\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Stats: {instance['stats']}\n")
        f.write("\n")
        
        # Parâmetros gerais
        f.write("# PARAMETERS\n")
        f.write(f"n_skus={instance['n_skus']}\n")
        f.write(f"horizon_days={instance['horizon_days']}\n")
        f.write(f"planning_slot_hours={instance['planning_slot_hours']}\n")
        f.write(f"initial_setup={instance['initial_setup']}\n")
        f.write(f"horizon_start={instance['horizon_start'].isoformat()}\n")
        f.write("\n")
        
        # Lista de SKUs
        f.write("# SKUS\n")
        for sku in instance["skus"]:
            f.write(f"{sku}\n")
        f.write("\n")
        
        # Demanda
        f.write("# PRODUCTION_NEED: sku,deadline,quantity\n")
        for _, row in instance["prod_need"].iterrows():
            deadline_str = row["deadline"].isoformat() if isinstance(row["deadline"], datetime) else str(row["deadline"])
            f.write(f"{row['sku']},{deadline_str},{int(row['prod_need'])}\n")
        f.write("\n")
        
        # Taxa de produção
        f.write("# PRODUCTION_RATE: date,rate\n")
        for _, row in instance["prod_rate"].iterrows():
            date_str = row["ref_date"].isoformat() if isinstance(row["ref_date"], datetime) else str(row["ref_date"])
            f.write(f"{date_str},{int(row['prod_rate'])}\n")
        f.write("\n")
        
        # Matriz de setup
        f.write("# SETUP_MATRIX: sku_from,sku_to,cost,time_minutes\n")
        for _, row in instance["setup_matrix"].iterrows():
            f.write(f"{row['sku_from']},{row['sku_to']},{row['setup_cost']},{row['setup_time']}\n")
    
    return filepath


def export_to_xlsx(instance: dict, output_dir: Path) -> Path:
    """
    Exporta instância para arquivo .xlsx para visualização.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"{instance['name']}.xlsx"
    
    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # Parâmetros
        params_df = pd.DataFrame([{
            "Parâmetro": "Nome",
            "Valor": instance["name"]
        }, {
            "Parâmetro": "Número de SKUs",
            "Valor": instance["n_skus"]
        }, {
            "Parâmetro": "Horizonte (dias)",
            "Valor": instance["horizon_days"]
        }, {
            "Parâmetro": "Escala de Demanda",
            "Valor": instance["demand_scale"]
        }, {
            "Parâmetro": "Tipo de Setup",
            "Valor": instance["setup_type"]
        }, {
            "Parâmetro": "Setup Inicial",
            "Valor": instance["initial_setup"]
        }, {
            "Parâmetro": "Períodos (12h)",
            "Valor": instance["stats"]["n_periods"]
        }, {
            "Parâmetro": "Demanda Total",
            "Valor": instance["stats"]["total_demand"]
        }])
        params_df.to_excel(writer, sheet_name="Parâmetros", index=False)
        
        # SKUs
        skus_df = pd.DataFrame({"SKU": instance["skus"]})
        skus_df.to_excel(writer, sheet_name="SKUs", index=False)
        
        # Demanda
        prod_need = instance["prod_need"].copy()
        prod_need.to_excel(writer, sheet_name="Demanda", index=False)
        
        # Demanda pivoteada (para visualização)
        if not prod_need.empty:
            try:
                pivot = prod_need.pivot_table(
                    values="prod_need",
                    index="sku",
                    columns="deadline",
                    aggfunc="sum",
                    fill_value=0
                )
                pivot.to_excel(writer, sheet_name="Demanda (Pivot)")
            except:
                pass  # Se não conseguir pivotar, ignora
        
        # Taxa de produção
        instance["prod_rate"].to_excel(writer, sheet_name="Taxa Produção", index=False)
        
        # Matriz de setup (formato tabular)
        instance["setup_matrix"].to_excel(writer, sheet_name="Setup Matrix", index=False)
        
        # Matriz de setup (formato matricial para visualização)
        setup_pivot = instance["setup_matrix"].pivot(
            index="sku_from",
            columns="sku_to",
            values="setup_cost"
        )
        setup_pivot.to_excel(writer, sheet_name="Setup Matrix (Pivot)")
    
    return filepath


# ============================================================================
# GERAÇÃO SISTEMÁTICA
# ============================================================================

def generate_all_instances(excel_path: str, output_dir: Path) -> list:
    """
    Gera todas as combinações de instâncias sistematicamente.
    
    Ordem: Do menor para o maior (aumentando complexidade)
    """
    print("=" * 60)
    print("GERADOR DE INSTÂNCIAS - LINE SCHEDULING")
    print("=" * 60)
    
    # Carregar dados base
    print(f"\n📂 Carregando dados de: {excel_path}")
    base_data = load_excel_data(excel_path)
    print(f"   SKUs disponíveis: {len(base_data['all_skus'])}")
    print(f"   Datas disponíveis: {len(base_data['all_dates'])}")
    
    # Gerar instâncias
    instances = []
    instance_id = 0
    
    # Ordem sistemática: SKUs → Dias → Setup → Demanda
    # (do menor para o maior)
    print(f"\n🔧 Gerando instâncias...")
    
    for n_skus in N_SKUS_LIST:
        for horizon_days in HORIZONS_DAYS:
            for setup_type in SETUP_TYPES:
                for demand_scale in DEMAND_SCALES:
                    instance_id += 1
                    
                    # Limitar SKUs ao disponível
                    actual_n_skus = min(n_skus, len(base_data["all_skus"]))
                    
                    instance = generate_instance(
                        base_data=base_data,
                        n_skus=actual_n_skus,
                        horizon_days=horizon_days,
                        demand_scale=demand_scale,
                        setup_type=setup_type,
                        instance_id=instance_id
                    )
                    
                    # Exportar
                    txt_path = export_to_txt(instance, output_dir)
                    xlsx_path = export_to_xlsx(instance, output_dir)
                    
                    instances.append({
                        "id": instance_id,
                        "name": instance["name"],
                        "n_skus": actual_n_skus,
                        "horizon_days": horizon_days,
                        "setup_type": setup_type,
                        "demand_scale": demand_scale,
                        "total_demand": instance["stats"]["total_demand"],
                        "n_periods": instance["stats"]["n_periods"],
                        "txt_file": str(txt_path),
                        "xlsx_file": str(xlsx_path),
                    })
                    
                    print(f"   [{instance_id:3d}] {instance['name']}")
    
    # Gerar índice
    index_df = pd.DataFrame(instances)
    index_path = output_dir / "instances_index.xlsx"
    index_df.to_excel(index_path, index=False)
    
    print(f"\n✅ Geradas {len(instances)} instâncias")
    print(f"📁 Diretório: {output_dir.absolute()}")
    print(f"📋 Índice: {index_path}")
    
    return instances


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Verificar se o arquivo Excel existe
    if not Path(EXCEL_FILE).exists():
        print(f"❌ Arquivo não encontrado: {EXCEL_FILE}")
        print("   Por favor, coloque o arquivo Excel na pasta Ball/")
        exit(1)
    
    # Gerar instâncias
    instances = generate_all_instances(EXCEL_FILE, OUTPUT_DIR)
    
    # Estatísticas finais
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS FINAIS")
    print("=" * 60)
    
    df = pd.DataFrame(instances)
    print(f"\nInstâncias por número de SKUs:")
    print(df.groupby("n_skus").size())
    
    print(f"\nInstâncias por horizonte (dias):")
    print(df.groupby("horizon_days").size())
    
    print(f"\nInstâncias por tipo de setup:")
    print(df.groupby("setup_type").size())
    
    print("\n✅ Geração concluída!")
