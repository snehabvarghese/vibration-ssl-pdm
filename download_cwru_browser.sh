#!/usr/bin/env bash
# download_cwru_browser.sh
# Opens all 64 CWRU .mat download URLs in your browser.
# Save the downloaded files to data/raw/
# Usage: chmod +x download_cwru_browser.sh && ./download_cwru_browser.sh

DATA_RAW="/Users/snehabvarghese/vibration-ssl-pdm/data/raw"
BASE="https://engineering.case.edu/sites/default/files"

FILES=(97 98 99 100 105 106 107 108 118 119 120 121 130 131 132 133
       144 145 146 147 156 158 159 160 169 170 171 172 185 186 187 188
       197 198 199 200 209 210 211 212 222 223 224 225 234 235 236 237
       246 247 248 249 258 259 260 261 3001 3002 3003 3004 3005 3006 3007 3008)

echo "Checking which files are already present..."
MISSING=()
for fid in "${FILES[@]}"; do
  SZ=$(stat -f%z "$DATA_RAW/${fid}.mat" 2>/dev/null || echo 0)
  if [[ -f "$DATA_RAW/${fid}.mat" ]] && [[ $SZ -gt 10000 ]]; then
    echo "  v ${fid}.mat"
  else
    MISSING+=("$fid")
  fi
done

if [[ ${#MISSING[@]} -eq 0 ]]; then
  echo "All 64 files present!"; exit 0
fi

echo ""
echo "${#MISSING[@]} files missing. Opening browser downloads..."
echo "Save all files to: $DATA_RAW"
echo ""

for fid in "${MISSING[@]}"; do
  open "$BASE/${fid}.mat"
  sleep 0.3
done

echo ""
echo "=========================================="
echo "After all files are downloaded, run:"
echo "  ./run_phase1.sh --from 2"
echo "=========================================="
