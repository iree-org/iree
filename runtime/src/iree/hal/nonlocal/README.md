# IREE nonlocal HAL driver plugins

### Build

```sh
IREE_BUILD=build
IREE_SRC=iree
CMAKE_BUILD_TYPE=Debug

cmake -G Ninja -B "${IREE_BUILD}" -S "${IREE_SRC}" \
	-DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
	-DCMAKE_C_COMPILER=clang \
	-DCMAKE_CXX_COMPILER=clang++ \
        -DIREE_HAL_DRIVER_NONLOCAL_SYNC=ON \
        -DIREE_HAL_DRIVER_NONLOCAL_TASK=ON \
	-DIREE_BUILD_PYTHON_BINDINGS=ON  \
	-DNL_USE_SERVER=ON \
	-DPython3_EXECUTABLE="$(which python3)" \

cmake --build "${IREE_BUILD}"
```

### Run server

```sh
"${IREE_BUILD}"/runtime/src/iree/hal/nonlocal/elf_module_server &
```

### Run test client
"${IREE_BUILD}"/runtime/src/iree/hal/nonlocal/elf_module_test_client

# Compile/Run MLIR

```sh
iree-compile \
  --iree-hal-target-device=local --iree-hal-target-backends=llvm-cpu \
  "${IREE_SRC}"/samples/models/simple_abs.mlir \
  -o /tmp/simple_abs.vmfb
```

```sh
iree-run-module --module=/tmp/simple_abs.vmfb --device=nonlocal-sync \
  --function=abs --input=f32=-2
```
