#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
revision=659041e1b12e3ab41c70ad12d11a29a883f17e15
repo=external/DiskANN-diversity
command -v cmake >/dev/null || { echo 'Install cmake and the dependencies listed in topics/D_hnsw/d003/README.md'; exit 1; }
if [[ ! -d "$repo" ]]; then
  git clone --branch diversity --single-branch https://github.com/microsoft/DiskANN.git "$repo"
  git -C "$repo" checkout --detach "$revision"
fi
[[ "$(git -C "$repo" rev-parse HEAD)" == "$revision" ]] || { echo 'Existing upstream checkout has a different revision; refusing to alter it.'; exit 1; }
[[ -z "$(git -C "$repo" diff HEAD --)" ]] || { echo 'Existing upstream tracked sources are modified; refusing to build.'; exit 1; }
# Explicit cstdint include fixes an upstream missing include on modern GCC.
# Additional CMake path options can be passed as arguments for non-system deps.
cmake -S "$repo" -B "$repo/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS='-include cstdint' "$@"
cmake --build "$repo/build" --target build_memory_index search_memory_index -j "${BUILD_JOBS:-2}"
