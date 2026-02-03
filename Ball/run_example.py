"""
Script de exemplo para rodar o LineSchedulingV1.

Este script cria dados de teste sintéticos e executa o modelo MILP
de Lot-Sizing and Scheduling usando PuLP (CBC solver).

Autor: Gerado para Felipe Silvestre
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

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


def criar_dados_de_teste():
    """
    Cria dados sintéticos para teste do modelo.
    
    Simula 3 produtos, 7 dias de planejamento, com slots de 12 horas.
    """
    
    # Parâmetros do problema
    produtos = ["SKU_A", "SKU_B", "SKU_C"]
    initial_setup = "SKU_A"  # Produto já na linha
    
    # Horizonte de planejamento
    horizon_start = datetime(2025, 1, 1, 0, 0, 0)
    last_demand_date = datetime(2025, 1, 7, 0, 0, 0)
    planning_hours_slot = 12  # Buckets de 12 horas
    
    # --- 1. Demanda (Production Need) ---
    # Deadline às 00:00 de cada dia (modelo vai antecipar produção)
    prod_need_data = []
    for day in range(1, 8):
        deadline = datetime(2025, 1, day, 0, 0, 0)
        for sku in produtos:
            # Demanda aleatória entre 1000 e 5000 unidades
            demand = np.random.randint(1000, 5000)
            prod_need_data.append({
                "sku": sku,
                "deadline": deadline,
                "prod_need": demand
            })
    
    prod_need = pd.DataFrame(prod_need_data)
    print(f"\n📦 Demanda Total por Produto:")
    print(prod_need.groupby("sku")["prod_need"].sum())
    
    # --- 2. Taxa de Produção (Production Rate) ---
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
    # Remover duplicatas (initial_setup já está em produtos se for igual)
    all_skus = list(set(produtos + [initial_setup]))
    
    setup_data = []
    for sku_from in all_skus:
        for sku_to in all_skus:
            if sku_from != sku_to:
                # Custo de setup aleatório entre 0.2 e 0.8
                cost = np.random.uniform(0.2, 0.8)
            else:
                cost = 0.0
            setup_data.append({
                "sku_from": sku_from,
                "sku_to": sku_to,
                "setup_cost": cost
            })
    
    setup_matrix = pd.DataFrame(setup_data)
    # Garantir que não há duplicatas
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
        "min_setup_time": 30,   # Minutos de setup mínimo
        "max_setup_time": 120,  # Minutos de setup máximo
    }


def rodar_modelo(dados):
    """
    Configura e executa o modelo LineSchedulingV1.
    """
    
    print("\n" + "="*60)
    print("🏭 INICIANDO MODELO DE LOT-SIZING AND SCHEDULING")
    print("="*60)
    
    # Import OptzRun
    from entities.optimization_run import OptzRun
    
    # Instanciar modelo vazio
    modelo = LineSchedulingV1()
    
    # Configurar OptzRun (necessário para a função objetivo)
    modelo.optz_run = OptzRun(
        setup_cost=1.0,
        min_setup_time=dados["min_setup_time"],
        max_setup_time=dados["max_setup_time"],
        planning_slot_size=dados["planning_hours_slot"]
    )
    
    # Configurar dados manualmente (bypass do sistema Ball)
    modelo.prod_need = dados["prod_need"]
    modelo.initial_setup = dados["initial_setup"]
    modelo.line = "LINE_TEST"  # Nome da linha de produção
    modelo.plant = "PLANT_TEST"  # Nome da planta
    
    # Gerar lista de produtos
    modelo.products = _gen_products_list(
        prod_need=dados["prod_need"],
        initial_setup=dados["initial_setup"]
    )
    print(f"\n📋 Produtos: {modelo.products}")
    
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
    
    # Gerar períodos
    modelo.periods = _gen_periods(
        horizon_start=dados["horizon_start"],
        planning_hours_slot=dados["planning_hours_slot"],
        last_demand_date=dados["last_demand_date"]
    )
    print(f"⏰ Períodos gerados: {len(modelo.periods)} (incluindo T0)")
    
    # Adicionar taxa de produção
    modelo.periods = _gen_time_to_produce(
        periods=modelo.periods,
        production_rate=dados["production_rate"]
    )
    
    # Criar problema LP
    print("\n🔧 Criando modelo de otimização...")
    modelo.lp_model = modelo._LineSchedulingV1__instantiate_problem()
    
    # Mostrar estatísticas do modelo
    print(f"   📊 Variáveis: {len(modelo.lp_model.variables()):,}")
    print(f"   📊 Restrições: {len(modelo.lp_model.constraints):,}")
    
    # Resolver
    print("\n🚀 Resolvendo modelo...")
    modelo.optimize(
        time_limit_s=300,  # 5 minutos de limite
        verbose=True
    )
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("📊 RESULTADOS")
    print("="*60)
    
    from pulp import LpStatus
    print(f"Status: {LpStatus[modelo.lp_model.status]}")
    
    if modelo.lp_model.status == 1:  # Optimal
        fo = modelo.get_fo()
        print(f"Valor da Função Objetivo: {fo:.2f}")
        
        # Mostrar algumas variáveis de exemplo
        print("\n📋 Produção por período (valores > 0):")
        for var in modelo.lp_model.variables():
            if var.name.startswith("Production") and var.varValue and var.varValue > 0:
                print(f"   {var.name}: {var.varValue:.0f}")
        
        print("\n🔄 Setups realizados:")
        for var in modelo.lp_model.variables():
            if var.name.startswith("Setup_") and var.varValue and var.varValue > 0:
                print(f"   {var.name}")
    
    return modelo


if __name__ == "__main__":
    # Semente para reprodutibilidade
    np.random.seed(42)
    
    # Criar dados de teste
    dados = criar_dados_de_teste()
    
    # Rodar modelo
    modelo = rodar_modelo(dados)
    
    print("\n✅ Modelo executado com sucesso!")
