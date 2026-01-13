#!/bin/bash
# =============================================================================
# TESTE RÁPIDO - Apenas 2 instâncias por 150s cada
# Resultado salvo em: Results/benchmark_teste.txt
# =============================================================================

TEMPO=150  # 2.5 minutos por instância

# Diretórios
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="$SCRIPT_DIR/../../../cpp/Program"
INSTANCES_DIR="$SCRIPT_DIR/instances"
RESULTS_DIR="$SCRIPT_DIR/Results"
RESULTS_FILE="$RESULTS_DIR/benchmark_teste.txt"

echo "=============================================================="
echo "  TESTE RÁPIDO - 150s por instância"
echo "  Resultados em: $RESULTS_FILE"
echo "=============================================================="

# Compila
echo ">> Compilando..."
cd "$CPP_DIR"
g++ -O3 -o teste_decoder teste_decoder.cpp
if [ $? -ne 0 ]; then
    echo "ERRO na compilação!"
    exit 1
fi
echo ">> Compilado com sucesso!"

# Cria diretório de resultados
mkdir -p "$RESULTS_DIR"

# Inicia arquivo de resultados (overwrite)
echo "" > "$RESULTS_FILE"
echo "==============================================================" >> "$RESULTS_FILE"
echo "# TESTE - $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_FILE"
echo "# Tempo: ${TEMPO}s por instância" >> "$RESULTS_FILE"
echo "--------------------------------------------------------------" >> "$RESULTS_FILE"

# Instânc# Lista de instâncias para teste
INSTANCIAS=("100_5_v2.txt")
# INSTANCIAS=("100_5_v2.txt" "700_15_v2.txt")

for INST in "${INSTANCIAS[@]}"; do
    INST_PATH="$INSTANCES_DIR/$INST"
    
    echo ""
    echo ">> Executando: $INST"
    echo "   Início: $(date '+%H:%M:%S')"
    
    if [ ! -f "$INST_PATH" ]; then
        echo "   ERRO: Arquivo não encontrado!"
        echo "$INST, ARQUIVO_NAO_ENCONTRADO" >> "$RESULTS_FILE"
        continue
    fi
    
    # Executa e mostra saída em tempo real
    "$CPP_DIR/teste_decoder" "$INST_PATH" "$TEMPO" | tee -a "$RESULTS_FILE"
    
    echo "" >> "$RESULTS_FILE"
done

echo ""
echo "=============================================================="
echo "TESTE CONCLUÍDO!"
echo "Resultados em: $RESULTS_FILE"
echo "=============================================================="
