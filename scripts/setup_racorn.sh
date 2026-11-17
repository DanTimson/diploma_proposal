#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
up=${RACORN_DIR:-external/RACORN}
rev=b1a587ad2f5d4048cf50c8e2710e5070864590c3
subrev=09aaff9d20dadbdbe133d3aba41b3c768c49888a
if [[ ! -d "$up" ]]; then
 git clone https://github.com/naver/racorn.git "$up"
 git -C "$up" checkout --detach "$rev"
fi
[[ $(git -C "$up" rev-parse HEAD) == "$rev" ]] || { echo 'Wrong RACORN revision'; exit 1; }
[[ -z $(git -C "$up" diff HEAD --) ]] || { echo 'Tracked RACORN changes; refusing build'; exit 1; }
git -C "$up" submodule update --init third_party/NumKong
[[ $(git -C "$up/third_party/NumKong" rev-parse HEAD) == "$subrev" ]] || exit 1
mkdir -p "$up/build/bench"
"${CXX:-g++}" -std=c++17 -O3 -march=native -DRACORN_USE_NUMKONG=1 -I "$up/include" \
 "$up/bench/bench_synthfilter.cpp" -o "$up/build/bench/bench_synthfilter" -pthread
python3 - "$up" <<'PY'
import hashlib,json,sys,subprocess,os
from pathlib import Path
p=Path(sys.argv[1]);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
files=list((p/'include').rglob('*.hpp'))+[p/'bench/bench_synthfilter.cpp']
receipt=dict(binary_sha256=sha(p/'build/bench/bench_synthfilter'),sources={str(f.relative_to(p)):sha(f) for f in files},
 compiler=subprocess.check_output([os.environ.get('CXX','g++'),'--version'],text=True).splitlines()[0],
 flags=['-std=c++17','-O3','-march=native','-DRACORN_USE_NUMKONG=1','-pthread'])
(p/'build/bench/build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
