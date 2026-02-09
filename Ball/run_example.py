"""
Script de exemplo para rodar o LineSchedulingV1.

Este script cria dados de teste sintÃ©ticos e executa o modelo MILP
de Lot-Sizing and Scheduling usando PuLP (CBC solver).

Autor: Gerado para Felipe Silvestre
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Import do modelo
from line_scheduling_v1 import (
    LineSchedulingV1,
    _gen_products_list,
    _gen_setup_cost_matrix,
    _gen_setup_time_matrix,
    _gen_periods,
    _gen_time_to_produce,
)


def carregar_instancia(instance_path: str) -> dict:
    """
    Carrega uma instÃ¢ncia de um arquivo .txt gerado pelo instance_generator.py
    
    Args:
        instance_path: Caminho para o arquivo .txt da instÃ¢ncia
        
    Returns:
        dict: DicionÃ¡rio com todos os dados necessÃ¡rios para rodar o modelo
    """
    import os
    from pathlib import Path
    
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
    
    # Get instance name for output
    instance_name = Path(instance_path).stem
    
    print(f"\nðŸ“‚ InstÃ¢ncia carregada: {instance_name}")
    print(f"   ðŸ“¦ SKUs: {len(skus)}")
    print(f"   ðŸ“… Horizonte: {horizon_days} dias")
    print(f"   ðŸ“Š Demandas: {len(prod_need_data)}")
    
    return {
        'instance_name': instance_name,
        'produtos': skus,
        'initial_setup': params['initial_setup'],
        'horizon_start': horizon_start,
        'last_demand_date': last_demand_date,
        'planning_hours_slot': int(params['planning_slot_hours']),
        'prod_need': prod_need,
        'production_rate': production_rate,
        'setup_matrix': setup_matrix,
        'min_setup_time': 30,
        'max_setup_time': 120,
    }



def criar_dados_de_teste():
    """
    Cria dados sintÃ©ticos para teste do modelo.
    
    Simula 3 produtos, 7 dias de planejamento, com slots de 12 horas.
    """
    
    # ParÃ¢metros do problema
    produtos = ["SKU_A", "SKU_B", "SKU_C"]
    initial_setup = "SKU_A"  # Produto jÃ¡ na linha
    
    # Horizonte de planejamento
    horizon_start = datetime(2025, 1, 1, 0, 0, 0)
    last_demand_date = datetime(2025, 1, 7, 0, 0, 0)
    planning_hours_slot = 12  # Buckets de 12 horas
    
    # --- 1. Demanda (Production Need) ---
    # Deadline Ã s 00:00 de cada dia (modelo vai antecipar produÃ§Ã£o)
    prod_need_data = []
    for day in range(1, 8):
        deadline = datetime(2025, 1, day, 0, 0, 0)
        for sku in produtos:
            # Demanda aleatÃ³ria entre 1000 e 5000 unidades
            demand = np.random.randint(1000, 5000)
            prod_need_data.append({
                "sku": sku,
                "deadline": deadline,
                "prod_need": demand
            })
    
    prod_need = pd.DataFrame(prod_need_data)
    print(f"\nðŸ“¦ Demanda Total por Produto:")
    print(prod_need.groupby("sku")["prod_need"].sum())
    
    # --- 2. Taxa de ProduÃ§Ã£o (Production Rate) ---
    # Capacidade de produzir X unidades por dia
    prod_rate_data = []
    for day in range(1, 8):
        prod_rate_data.append({
            "ref_date": datetime(2025, 1, day),
            "prod_rate": 10000  # 10.000 unidades por dia
        })
    
    production_rate = pd.DataFrame(prod_rate_data)
    
    # --- 3. Matriz de Setup (Setup Matrix) ---
    # Setup cost/complexity entre produtos (0 a 1)
    # Remover duplicatas (initial_setup jÃ¡ estÃ¡ em produtos se for igual)
    all_skus = list(set(produtos + [initial_setup]))
    
    setup_data = []
    for sku_from in all_skus:
        for sku_to in all_skus:
            if sku_from != sku_to:
                # Custo de setup aleatÃ³rio entre 0.2 e 0.8
                cost = np.random.uniform(0.2, 0.8)
            else:
                cost = 0.0
            setup_data.append({
                "sku_from": sku_from,
                "sku_to": sku_to,
                "setup_cost": cost
            })
    
    setup_matrix = pd.DataFrame(setup_data)
    # Garantir que nÃ£o hÃ¡ duplicatas
    setup_matrix = setup_matrix.drop_duplicates(subset=["sku_from", "sku_to"])
    
    return {
        "produtos": produtos,
        "initial_setup": initial_setup,
        "horizon_start": horizon_start,
        "last_demand_date": last_demand_date, 
        "planning_hours_slot": planning_hours_slot,
        "prod_need": prod_need,
        "production_rate": production_rate,
        "setup_matrix": setup_matrix,
        "min_setup_time": 30,   # Minutos de setup mÃ­nimo
        "max_setup_time": 120,  # Minutos de setup mÃ¡ximo
    }


def rodar_modelo(dados):
    """
    Configura e executa o modelo LineSchedulingV1.
    """
    
    print("\n" + "="*60)
    print("ðŸ­ INICIANDO MODELO DE LOT-SIZING AND SCHEDULING")
    print("="*60)
    
    # Import OptzRun
    from entities.optimization_run import OptzRun
    
    # Instanciar modelo vazio
    modelo = LineSchedulingV1()
    
    # Configurar OptzRun (necessÃ¡rio para a funÃ§Ã£o objetivo)
    modelo.optz_run = OptzRun(
        setup_cost=1.0,
        min_setup_time=dados["min_setup_time"],
        max_setup_time=dados["max_setup_time"],
        planning_slot_size=dados["planning_hours_slot"]
    )
    
    # Configurar dados manualmente (bypass do sistema Ball)
    modelo.prod_need = dados["prod_need"]
    modelo.initial_setup = dados["initial_setup"]
    modelo.line = "LINE_TEST"  # Nome da linha de produÃ§Ã£o
    modelo.plant = "PLANT_TEST"  # Nome da planta
    
    # Gerar lista de produtos
    modelo.products = _gen_products_list(
        prod_need=dados["prod_need"],
        initial_setup=dados["initial_setup"]
    )
    print(f"\nðŸ“‹ Produtos: {modelo.products}")
    
    # Gerar matrizes de setup
    modelo.setup_cost = _gen_setup_cost_matrix(
        products=modelo.products,
        setup_matrix=dados["setup_matrix"]
    )
    modelo.setup_time = _gen_setup_time_matrix(
        products=modelo.products,
        setup_matrix=dados["setup_matrix"],
        min_setup_time_multiplier=dados["min_setup_time"],
        max_setup_time_multiplier=dados["max_setup_time"]
    )
    
    # Gerar perÃ­odos
    modelo.periods = _gen_periods(
        horizon_start=dados["horizon_start"],
        planning_hours_slot=dados["planning_hours_slot"],
        last_demand_date=dados["last_demand_date"]
    )
    print(f"â° PerÃ­odos gerados: {len(modelo.periods)} (incluindo T0)")
    
    # Adicionar taxa de produÃ§Ã£o
    modelo.periods = _gen_time_to_produce(
        periods=modelo.periods,
        production_rate=dados["production_rate"]
    )
    
    # Criar problema LP
    print("\nðŸ”§ Criando modelo de otimizaÃ§Ã£o...")
    modelo.lp_model = modelo._LineSchedulingV1__instantiate_problem()
    
    # Mostrar estatÃ­sticas do modelo
    print(f"   ðŸ“Š VariÃ¡veis: {len(modelo.lp_model.variables()):,}")
    print(f"   ðŸ“Š RestriÃ§Ãµes: {len(modelo.lp_model.constraints):,}")
    
    # Resolver
    print("\nðŸš€ Resolvendo modelo...")
    modelo.optimize(
        time_limit_s=300,  # 5 minutos de limite
        verbose=True
    )
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("ðŸ“Š RESULTADOS")
    print("="*60)
    
    from pulp import LpStatus
    print(f"Status: {LpStatus[modelo.lp_model.status]}")
    
    if modelo.lp_model.status == 1:  # Optimal
        fo = modelo.get_fo()
        print(f"Valor da FunÃ§Ã£o Objetivo: {fo:.2f}")
        
        # Mostrar algumas variÃ¡veis de exemplo
        print("\nðŸ“‹ ProduÃ§Ã£o por perÃ­odo (valores > 0):")
        for var in modelo.lp_model.variables():
            if var.name.startswith("Production") and var.varValue and var.varValue > 0:
                print(f"   {var.name}: {var.varValue:.0f}")
        
        print("\nðŸ”„ Setups realizados:")
        for var in modelo.lp_model.variables():
            if var.name.startswith("Setup_") and var.varValue and var.varValue > 0:
                print(f"   {var.name}")
    else:
        fo = None
    
    # Retornar modelo e mÃ©tricas para CSV
    return {
        'modelo': modelo,
        'fo': fo,
        'tempo': modelo.lp_model.solutionTime if hasattr(modelo.lp_model, 'solutionTime') else 0,
        'status': modelo.lp_model.status
    }


def visualizar_solucao(modelo, output_path="solucao_scheduling.png"):
    """
    Gera uma visualizaÃ§Ã£o completa da soluÃ§Ã£o do modelo de scheduling.
    
    Mostra:
    - ProduÃ§Ã£o por perÃ­odo e produto (Gantt)
    - Demanda, InventÃ¡rio e Backlog por produto
    - EstatÃ­sticas gerais
    """
    
    # Coletar dados de todas as variÃ¡veis
    production_data = []
    inventory_data = []
    backlog_data = []
    setup_data = []
    
    for var in modelo.lp_model.variables():
        if var.varValue is None:
            continue
            
        name = var.name
        value = var.varValue
        
        # Production: Production_PROD_T1
        if name.startswith("Production_") and value > 0:
            parts = name.split("_")
            production_data.append({
                "product": parts[1],
                "period_num": int(parts[2][1:]),
                "amount": value
            })
        
        # Inventory: Inventory_PROD_T1
        elif name.startswith("Inventory_"):
            parts = name.split("_")
            inventory_data.append({
                "product": parts[1],
                "period_num": int(parts[2][1:]),
                "amount": value
            })
        
        # Backlog: Backlog_PROD_T1
        elif name.startswith("Backlog_"):
            parts = name.split("_")
            backlog_data.append({
                "product": parts[1],
                "period_num": int(parts[2][1:]),
                "amount": value
            })
        
        # Setup: Setup_PROD_PROD_T1
        elif name.startswith("Setup_") and value > 0:
            parts = name.split("_")
            setup_data.append({
                "product_from": parts[2],
                "product_to": parts[1],
                "period_num": int(parts[3][1:])
            })
    
    if not production_data:
        print("âŒ Sem dados de produÃ§Ã£o para visualizar")
        return None
    
    # DataFrames
    df_prod = pd.DataFrame(production_data)
    df_inv = pd.DataFrame(inventory_data) if inventory_data else pd.DataFrame()
    df_back = pd.DataFrame(backlog_data) if backlog_data else pd.DataFrame()
    df_setup = pd.DataFrame(setup_data) if setup_data else pd.DataFrame()
    
    # Produtos e perÃ­odos
    products = sorted(df_prod["product"].unique())
    all_periods = sorted(df_prod["period_num"].unique())
    max_period = max(all_periods)
    
    # Cores
    colors = plt.cm.Set2(np.linspace(0, 1, len(products)))
    product_colors = {prod: colors[i] for i, prod in enumerate(products)}
    
    # ========== CRIAR FIGURA COM 3 SUBPLOTS ==========
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[2, 1.5, 1], width_ratios=[3, 1])
    
    ax_gantt = fig.add_subplot(gs[0, :])
    ax_balance = fig.add_subplot(gs[1, 0])
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])
    
    # ========== 1. GRÃFICO DE GANTT (PRODUÃ‡ÃƒO) ==========
    bar_height = 0.6
    y_positions = {prod: i for i, prod in enumerate(products)}
    
    for _, row in df_prod.iterrows():
        product = row["product"]
        period = row["period_num"]
        amount = row["amount"]
        
        ax_gantt.barh(y=y_positions[product], width=0.8, left=period - 0.4,
                      height=bar_height, color=product_colors[product],
                      edgecolor='black', linewidth=0.5, alpha=0.85)
        ax_gantt.text(period, y_positions[product], f'{amount:.0f}',
                      ha='center', va='center', fontsize=7, fontweight='bold', color='black')
    
    # Marcadores de setup
    if not df_setup.empty:
        for _, row in df_setup.iterrows():
            if row["product_to"] in y_positions:
                ax_gantt.plot(row["period_num"] - 0.4, 
                             y_positions[row["product_to"]] + bar_height/2 + 0.1,
                             marker='v', markersize=8, color='red',
                             markeredgecolor='darkred', markeredgewidth=1)
    
    ax_gantt.set_yticks(list(y_positions.values()))
    ax_gantt.set_yticklabels(list(y_positions.keys()), fontsize=10, fontweight='bold')
    ax_gantt.set_xlim(0.5, max_period + 0.5)
    ax_gantt.set_xticks(range(1, max_period + 1))
    ax_gantt.set_xticklabels([f'T{i}' for i in range(1, max_period + 1)], fontsize=8)
    ax_gantt.set_xlabel('PerÃ­odo', fontsize=11, fontweight='bold')
    ax_gantt.set_ylabel('Produto', fontsize=11, fontweight='bold')
    ax_gantt.grid(axis='x', linestyle='--', alpha=0.4)
    
    fo_value = modelo.get_fo() if hasattr(modelo, 'get_fo') else 0
    ax_gantt.set_title(f'PRODUÃ‡ÃƒO POR PERÃODO (FO = {fo_value:.2f})', fontsize=13, fontweight='bold')
    
    legend_patches = [mpatches.Patch(color=product_colors[p], label=p) for p in products]
    legend_patches.append(plt.Line2D([0], [0], marker='v', color='w', 
                          markerfacecolor='red', markersize=8, label='Setup'))
    ax_gantt.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=min(len(products)+1, 4))
    
    # ========== 2. GRÃFICO DE INVENTÃRIO E BACKLOG ==========
    x = np.arange(1, max_period + 1)
    width = 0.8 / len(products)
    
    for idx, product in enumerate(products):
        offset = (idx - len(products)/2 + 0.5) * width
        
        if not df_inv.empty:
            inv_vals = df_inv[df_inv["product"] == product].set_index("period_num").reindex(x, fill_value=0)["amount"].values
            ax_balance.bar(x + offset, inv_vals, width*0.9, 
                          label=f'{product}', 
                          color=product_colors[product], alpha=0.7, edgecolor='green', linewidth=1)
        
        if not df_back.empty:
            back_vals = df_back[df_back["product"] == product].set_index("period_num").reindex(x, fill_value=0)["amount"].values
            ax_balance.bar(x + offset, -back_vals, width*0.9,
                          color=product_colors[product], alpha=0.4, edgecolor='red', linewidth=1, hatch='//')
    
    ax_balance.axhline(y=0, color='black', linewidth=1)
    ax_balance.set_xlabel('PerÃ­odo', fontsize=10)
    ax_balance.set_ylabel('InventÃ¡rio (+) / Backlog (-)', fontsize=10)
    ax_balance.set_title('INVENTÃRIO E BACKLOG POR PERÃODO', fontsize=11, fontweight='bold')
    ax_balance.set_xticks(x)
    ax_balance.set_xticklabels([f'T{i}' for i in x], fontsize=8)
    ax_balance.grid(axis='y', linestyle='--', alpha=0.4)
    ax_balance.legend(fontsize=7, loc='upper left')
    
    # ========== 3. ESTATÃSTICAS ==========
    ax_stats.axis('off')
    
    total_prod = df_prod["amount"].sum()
    total_inv = df_inv["amount"].sum() if not df_inv.empty else 0
    total_back = df_back["amount"].sum() if not df_back.empty else 0
    num_setups = len(df_setup)
    
    stats_text = (
        f"RESUMO\n"
        f"{'â”€'*25}\n\n"
        f"ProduÃ§Ã£o: {total_prod:,.0f}\n\n"
        f"InventÃ¡rio: {total_inv:,.0f}\n\n"
        f"Backlog: {total_back:,.0f}\n\n"
        f"Setups: {num_setups}\n\n"
        f"FO: {fo_value:,.2f}\n"
    )
    
    ax_stats.text(0.1, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ========== 4. TABELA RESUMO POR PRODUTO ==========
    ax_table.axis('off')
    
    table_data = []
    for product in products:
        prod_sum = df_prod[df_prod["product"] == product]["amount"].sum()
        inv_sum = df_inv[df_inv["product"] == product]["amount"].sum() if not df_inv.empty else 0
        back_sum = df_back[df_back["product"] == product]["amount"].sum() if not df_back.empty else 0
        table_data.append([product, f'{prod_sum:,.0f}', f'{inv_sum:,.0f}', f'{back_sum:,.0f}'])
    
    table = ax_table.table(cellText=table_data,
                           colLabels=['Produto', 'ProduÃ§Ã£o', 'InventÃ¡rio', 'Backlog'],
                           loc='center', cellLoc='center', colColours=['lightblue']*4)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax_table.set_title('RESUMO POR PRODUTO', fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nðŸ“Š VisualizaÃ§Ã£o salva em: {output_path}")
    return output_path



if __name__ == "__main__":

    skus = [4,6,8]
    days = [7,14,21]
    density = [100, 150]
    setup_types = ['random', 'proportional', 'uniform']

    # Criar pasta de imagens se nÃ£o existir
    images_dir = Path(__file__).parent / "images"
    images_dir.mkdir(exist_ok=True)
    
    instances_dir = Path(__file__).parent / "instances"
    
    # Lista para armazenar resultados do CSV
    resultados_csv = []

    for sku in skus:
        for day in days:
            for den in density:
                for setup_type in setup_types:
                    # Montar nome do arquivo
                    instance_file = instances_dir / f"inst_{sku}skus_{day}days_{setup_type}_d{den}.txt"
                    
                    if instance_file.exists():
                        # Carregar instÃ¢ncia de arquivo
                        dados = carregar_instancia(str(instance_file))
                        output_name = images_dir / f"inst_{sku}skus_{day}days_{setup_type}_d{den}_solucao.png"

                        # Rodar modelo (agora retorna dict)
                        resultado = rodar_modelo(dados)
                        modelo = resultado['modelo']
                        
                        # Adicionar ao CSV
                        resultados_csv.append({
                            'instancia': instance_file.name,
                            'fo': resultado['fo'],
                            'tempo_s': resultado['tempo'],
                            'status': resultado['status']
                        })
                        
                        # Gerar visualizaÃ§Ã£o
                        visualizar_solucao(modelo, output_path=str(output_name))
                        
                        print(f"\nâœ… {instance_file.name} executado com sucesso!")
                    else:
                        print(f"\nâš ï¸ Arquivo nÃ£o encontrado: {instance_file.name}")
    
    # Salvar CSV com resultados
    csv_path = Path(__file__).parent / "resultados_benchmark.csv"
    df_resultados = pd.DataFrame(resultados_csv)
    df_resultados.to_csv(csv_path, index=False)
    print(f"\nðŸ“Š Resultados salvos em: {csv_path}")
    print(df_resultados.to_string())

    


