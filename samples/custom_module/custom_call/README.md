# Custom call sample

This sample expects that you've already produced a working version of the
[async sample](/samples/custom_module/async/) and [dynamic
sample](/samples/custom_module/dynamic) (including compiler installation and
CMake setup).

This sample demonstrates adding custom calls with ABI containing just void*
pointers for inputs and outputs, executing on the host CPU.

## Instructions

1. Build the `iree_samples_custom_module_async_run` CMake target :
```
cmake -B ../iree-build/ -DCMAKE_BUILD_TYPE=RelWithDebInfo .
cmake --build ../iree-build/
cmake --build ../iree-build/ --target iree-run-module --target iree_samples_custom_module_custom_call_module
```

2. Compile the [example module](./test/example.mlir) to a .vmfb file:
```
iree-compile --iree-execution-model=async-external --iree-hal-target-backends=llvm-cpu samples/custom_module/custom_call/test/example.mlir -o=/tmp/example.vmfb
```

3. Run the example program to call the main function:
```
../iree-build/tools/iree-run-module --module=../iree-build/samples/custom_module/custom_call/module.so@create_custom_module     --module=/tmp/example.vmfb --function=main --input="2x3xi32=[1,2,3,4,5,6]"
```
