#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
up=${ACORN_DIR:-external/ACORN}
rev=c259f11cdbec880671658eaa0d8747e2cc9de79d
if [[ ! -d "$up" ]]; then
  git clone https://github.com/guestrin-lab/ACORN.git "$up"
  git -C "$up" checkout --detach "$rev"
fi
[[ $(git -C "$up" rev-parse HEAD) == "$rev" ]] || { echo 'Wrong ACORN revision'; exit 1; }
# A CMake export fix only; no search/construction changes. Safe to rerun.
python3 - "$up" <<'PY'
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]); name='CMakeLists.txt'
original=subprocess.check_output(['git','-C',str(root),'show','HEAD:'+name],text=True)
fixed=original.replace('target_include_directories(faiss PUBLIC ${nlohmann_json_SOURCE_DIR}/include)',
 'target_include_directories(faiss PUBLIC $<BUILD_INTERFACE:${nlohmann_json_SOURCE_DIR}/include>)')
if fixed==original: raise SystemExit('Expected CMake line missing')
changed=subprocess.check_output(['git','-C',str(root),'diff','--name-only','HEAD'],text=True).splitlines()
if any(p!=name for p in changed) or (root/name).read_text() not in (original,fixed):
 raise SystemExit('Unexpected upstream modifications; refusing to overwrite')
(root/name).write_text(fixed)
PY
"${CMAKE:-cmake}" -S "$up" -B "$up/build" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release "$@"
"${CMAKE:-cmake}" --build "$up/build" --target faiss -j "${BUILD_JOBS:-2}"
mkdir -p .local/d004-native
"${CXX:-g++}" -O3 -std=c++17 -fopenmp -I "$up" native/acorn_probe.cpp \
  -L "$up/build/faiss" -Wl,-rpath,"$(realpath "$up/build/faiss")" -lfaiss \
  -o .local/d004-native/acorn_probe
python3 - "$up" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve(); binary=Path('.local/d004-native/acorn_probe')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=dict(binary_sha256=sha(binary),adapter_sha256=sha('native/acorn_probe.cpp'),
 library_sha256=sha(root/'build/faiss/libfaiss.so'),
 upstream_revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
 cmake_diff=subprocess.check_output(['git','-C',str(root),'diff','HEAD','--','CMakeLists.txt'],text=True))
binary.with_suffix('.build.json').write_text(json.dumps(receipt,indent=2)+'\n')
PY
