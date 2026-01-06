#!/bin/bash

# 1. Configurações
# Caminho das instâncias no Windows traduzido para WSL
INST_DIR="/mnt/c/Users/felip/OneDrive/Documentos/GitHub/PO_Ball/Problems/Sequenciamento/instances"
TIME_LIMIT=150

# 2. Garante que está compilado (por segurança)
echo "Compilando..."
g++ -std=c++14 -o runTest src/Main/main.cpp -O3 -fopenmp

# 3. Loop das Instâncias
echo "Iniciando Bateria de Testes..."
echo "Configuração: 10 Runs (interno) | ${TIME_LIMIT}s por Run"

# Loop de Produtos (100, 200, ... 700)
for n in {100..700..100}; do
    # Loop de Máquinas (5, 10, 15)
    for m in 5 10 15; do
        
        # Define o nome do arquivo (ajuste se você usou _v2 ou não)
        # Padrão do gerador: 100_5.txt
        INSTANCE_FILE="${INST_DIR}/${n}_${m}_v2.txt"
        
        # Verifica se o arquivo existe antes de rodar
        if [ -f "$INSTANCE_FILE" ]; then
            echo "------------------------------------------------"
            echo "Rodando: ${n} Produtos | ${m} Máquinas"
            echo "Arquivo: $INSTANCE_FILE"
            
            # Executa o RKO
            ./runTest "$INSTANCE_FILE" $TIME_LIMIT
            
        else
            echo "ALERTA: Instância $INSTANCE_FILE não encontrada. Pulando."
        fi
    done
done

echo "------------------------------------------------"
echo "FIM DOS EXPERIMENTOS!"
echo "Resultados salvos em: ../Results/Results_RKO.csv"

