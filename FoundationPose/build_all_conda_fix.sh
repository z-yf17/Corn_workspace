#!/usr/bin/env bash

set -euo pipefail



PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )



# Ensure pip builds without isolation (so it can see torch in this conda env)

export PIP_NO_BUILD_ISOLATION=1



# Install mycpp

cd "${PROJ_ROOT}/mycpp/"

rm -rf build

mkdir -p build

cd build

cmake ..

make -j"$(nproc)"



# Install mycuda

cd "${PROJ_ROOT}/bundlesdf/mycuda"

rm -rf build *egg* *.so

python -m pip install -e . --no-build-isolation -v



cd "${PROJ_ROOT}"


