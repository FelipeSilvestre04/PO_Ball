#!/bin/bash
# =============================================================================
# Run RKO Benchmark (C++) - All 21 Instances (Reverse Order)
# Time: 1200s per instance
# =============================================================================

TIME_LIMIT=1200

# Directories
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_DIR="$SCRIPT_DIR/../../../cpp/Program"
INSTANCES_DIR="$SCRIPT_DIR/instances"

# Ensure output directory exists (relative to C++ binary)
mkdir -p "$CPP_DIR/../Results"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=============================================================="
echo "  BENCHMARK RKO (C++)"
echo "  Order: 700_15 -> 100_5"
echo "  Time: ${TIME_LIMIT}s per instance"
echo "=============================================================="

# 1. Compile
echo -e "${YELLOW}>> Compiling C++ code...${NC}"
cd "$CPP_DIR"
make
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Compilation failed!${NC}"
    exit 1
fi
echo -e "${GREEN}>> Compilation OK!${NC}"
echo ""

# 2. Instance List (Reverse Order)
INSTANCES=(
    "700_15_v2.txt" "700_10_v2.txt" "700_5_v2.txt"
    "600_15_v2.txt" "600_10_v2.txt" "600_5_v2.txt"
    "500_15_v2.txt" "500_10_v2.txt" "500_5_v2.txt"
    "400_15_v2.txt" "400_10_v2.txt" "400_5_v2.txt"
    "300_15_v2.txt" "300_10_v2.txt" "300_5_v2.txt"
    "200_15_v2.txt" "200_10_v2.txt" "200_5_v2.txt"
    "100_15_v2.txt" "100_10_v2.txt" "100_5_v2.txt"
)

TOTAL=${#INSTANCES[@]}
COUNT=0

for INST in "${INSTANCES[@]}"; do
    COUNT=$((COUNT + 1))
    INST_PATH="$INSTANCES_DIR/$INST"
    
    echo "--------------------------------------------------------------"
    echo -e "${YELLOW}[$COUNT/$TOTAL] Processing: $INST${NC}"
    echo "Start: $(date '+%H:%M:%S')"
    
    if [ ! -f "$INST_PATH" ]; then
        echo -e "${RED}FILE NOT FOUND: $INST_PATH${NC}"
        continue
    fi
    
    # Execute RKO
    ./runTest "$INST_PATH" "$TIME_LIMIT"
    
    echo ""
done

echo "=============================================================="
echo -e "${GREEN}BENCHMARK COMPLETED!${NC}"
echo "Results saved in: cpp/Results/Results_RKO.csv"
echo "=============================================================="
