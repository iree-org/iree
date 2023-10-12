# Recommended for simple development using clang and lld:
export CUDAToolkit_INCLUDE_DIR=/usr/local/cuda/include
echo $CUDAToolkit_INCLUDE_DIR

cmake -GNinja -B iree-build-host/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DIREE_CUDA_AVAILABLE=ON \
    -DIREE_CUDA_AVAILABLE=ON \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DIREE_ENABLE_LLD=OFF

# cmake --build  iree-build-host/

