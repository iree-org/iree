# GitHub Actions Based CI

## Debugging releases cookbook

### Build in the same Linux container as the release

```
docker run --rm -it -v $(pwd):/work stellaraccident/manylinux2014_x86_64-bazel-3.7.2:latest /bin/bash

# From within docker.
export PATH=/opt/python/cp38-cp38/bin:$PATH
python -m pip install -r bindings/python/build_requirements.txt
cd /work
python ./build_tools/gitub_actions/cmake_ci.py -B../iree-build \
  -DCMAKE_INSTALL_PREFIX=../iree-install -DCMAKE_BUILD_TYPE=Release \
  -DIREE_BUILD_SAMPLES=OFF -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_BUILD_TENSORFLOW_COMPILER=OFF -DIREE_BUILD_TFLITE_COMPILER=OFF -DIREE_BUILD_XLA_COMPILER=OFF
python ./build_tools/github_actions/cmake_ci.py --build ../iree-build
```
