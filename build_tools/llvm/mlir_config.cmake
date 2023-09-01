# export MLIR_BUILD_DIR=$HOME/tmp/mlirbuild
# export MLIR_INSTALL_DIR=$HOME/tmp/mlirroot
# cmake -G Ninja \
#     -B "$MLIR_BUILD_DIR" -S third_party/llvm-project/mlir \
#     -DLLVM_DIR="${LLVM_INSTALL_DIR}/lib/cmake/llvm" \
#     -C build_tools/llvm/mlir_config.cmake \
#     -DCMAKE_BUILD_TYPE="Release" \
#     -DPython3_EXECUTABLE='$(which $python3_command)' \
#     -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
#     -DCMAKE_INSTALL_PREFIX="$MLIR_INSTALL_DIR" \
#     -DCMAKE_C_COMPILER_LAUNCHER=ccache \
#     -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
#     -DCMAKE_C_COMPILER=clang \
#     -DCMAKE_CXX_COMPILER=clang++ \
#     -DCMAKE_LINKER=lld
# ninja -C $MLIR_BUILD_DIR install-mlirdevelopment-distribution

if(NOT EXISTS ${LLVM_DIR})
  message(FATAL_ERROR "LLVM_DIR (${LLVM_DIR}) does not exist")
endif()

# When exceptions are disabled, unwind tables are large and useless
set(LLVM_ENABLE_UNWIND_TABLES OFF CACHE BOOL "")

# Do not store debug information by default.
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

# Use the distributions below for the installation
set(LLVM_INSTALL_TOOLCHAIN_ONLY OFF CACHE BOOL "")

# Build tools and utils.
set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_BUILD_UTILS ON CACHE BOOL "")

### Distributions ###

set(LLVM_DISTRIBUTIONS
    MlirDevelopment
    CACHE STRING "")

set(LLVM_MLIR_TOOLS
      mlir-opt
      mlir-reduce
      mlir-tblgen
      mlir-translate
    CACHE STRING "")

set(LLVM_MLIR_Python_COMPONENTS
      MLIRPythonModules
      mlir-python-sources
    CACHE STRING "")

set(LLVM_MlirDevelopment_DISTRIBUTION_COMPONENTS
      mlir-cmake-exports
      mlir-headers
      mlir-libraries
      MLIRPythonModules
      ${LLVM_MLIR_TOOLS}
      ${LLVM_MLIR_Python_COMPONENTS}
    CACHE STRING "")
