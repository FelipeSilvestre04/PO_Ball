#!/bin/bash
# =============================================================================
# Script para executar benchmark C++ no WSL
# Instâncias MAIORES primeiro, 1200 segundos cada
# =============================================================================

TEMPO=1200  # 20 minutos por instância

# Diretórios (ajuste se necessário - estes são caminhos WSL)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="$SCRIPT_DIR/../../../cpp/Program"
INSTANCES_DIR="$SCRIPT_DIR/instances"
RESULTS_DIR="$SCRIPT_DIR/Results"
RESULTS_FILE="$RESULTS_DIR/benchmark_cpp_1200s.txt"

# Cores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=============================================================="
echo "  BENCHMARK C++ - SEQUENCIAMENTO (1200s por instância)"
echo "=============================================================="
echo ""

# Verifica se o diretório de instâncias existe
if [ ! -d "$INSTANCES_DIR" ]; then
    echo -e "${RED}ERRO: Diretório de instâncias não encontrado: $INSTANCES_DIR${NC}"
    exit 1
fi

# Compila o código C++ se necessário
echo -e "${YELLOW}>> Compilando código C++...${NC}"
cd "$CPP_DIR"

# Compila com otimizações máximas
g++ -O3 -march=native -o teste_decoder teste_decoder.cpp 2>/dev/null

if [ $? -ne 0 ]; then
    echo -e "${RED}ERRO: Falha na compilação!${NC}"
    echo "Tentando compilar sem -march=native..."
    g++ -O3 -o teste_decoder teste_decoder.cpp
    if [ $? -ne 0 ]; then
        echo -e "${RED}ERRO: Compilação falhou!${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}>> Compilação OK!${NC}"
echo ""

# Cria diretório de resultados
mkdir -p "$RESULTS_DIR"

# Cabeçalho do arquivo de resultados (append)
echo "" >> "$RESULTS_FILE"
echo "==============================================================" >> "$RESULTS_FILE"
echo "# Benchmark C++ - $(date '+%Y-%m-%d %H:%M:%S')" >> "$RESULTS_FILE"
echo "# Tempo por instância: ${TEMPO}s" >> "$RESULTS_FILE"
echo "# Formato: instancia, melhor_custo, avaliacoes, velocidade" >> "$RESULTS_FILE"
echo "--------------------------------------------------------------" >> "$RESULTS_FILE"

# Lista de instâncias - MAIORES PRIMEIRO (COM sufixo _v2)
INSTANCIAS=(
    "700_15_v2.txt"
    "700_10_v2.txt"
    "700_5_v2.txt"
    "600_15_v2.txt"
    "600_10_v2.txt"
    "600_5_v2.txt"
    "500_15_v2.txt"
    "500_10_v2.txt"
    "500_5_v2.txt"
    "400_15_v2.txt"
    "400_10_v2.txt"
    "400_5_v2.txt"
    "300_15_v2.txt"
    "300_10_v2.txt"
    "300_5_v2.txt"
    "200_15_v2.txt"
    "200_10_v2.txt"
    "200_5_v2.txt"
    "100_15_v2.txt"
    "100_10_v2.txt"
    "100_5_v2.txt"
)

TOTAL=${#INSTANCIAS[@]}
TEMPO_TOTAL=$((TOTAL * TEMPO))
echo "Total de instâncias: $TOTAL"
echo "Tempo estimado: $((TEMPO_TOTAL / 3600))h $((TEMPO_TOTAL % 3600 / 60))min"
echo ""
echo "=============================================================="

COUNT=0
for INST in "${INSTANCIAS[@]}"; do
    COUNT=$((COUNT + 1))
    INST_PATH="$INSTANCES_DIR/$INST"
    
    echo ""
    echo -e "${YELLOW}[$COUNT/$TOTAL] $INST${NC}"
    echo "    Início: $(date '+%H:%M:%S')"
    
    # Verifica se o arquivo existe
    if [ ! -f "$INST_PATH" ]; then
        echo -e "    ${RED}ARQUIVO NÃO ENCONTRADO - pulando${NC}"
        echo "$INST, ARQUIVO_NAO_ENCONTRADO" >> "$RESULTS_FILE"
        continue
    fi
    
    # Executa o benchmark
    OUTPUT=$("$CPP_DIR/teste_decoder" "$INST_PATH" "$TEMPO" 2>&1)
    
    # Extrai informações da saída
    BEST_COST=$(echo "$OUTPUT" | grep "MELHOR CUSTO:" | awk -F': ' '{print $2}')
    EVALS=$(echo "$OUTPUT" | grep "TOTAL AVALIACOES:" | awk -F': ' '{print $2}')
    SPEED=$(echo "$OUTPUT" | grep "VELOCIDADE:" | awk -F': ' '{print $2}')
    
    echo -e "    ${GREEN}Melhor custo: $BEST_COST${NC}"
    echo "    Avaliações: $EVALS"
    echo "    Velocidade: $SPEED"
    
    # Salva no arquivo
    echo "$INST, $BEST_COST, $EVALS, $SPEED" >> "$RESULTS_FILE"
done

echo ""
echo "=============================================================="
echo -e "${GREEN}BENCHMARK CONCLUÍDO!${NC}"
echo "Resultados salvos em: $RESULTS_FILE"
echo "=============================================================="
