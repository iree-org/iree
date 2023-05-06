# IREE Dispatch Profiler

The IREE Dispatch Profiler is a Python-based tool designed to achieve two primary objectives: functional verification and performance profiling for individual dispatches, such as matrix multiplication, batch matrix multiplication, and convolutions. This tool ensures that performance optimizations maintain functionality and provides a convenient way to quantitatively measure performance. Additionally, the tool offers dispatch generation and compilation capabilities. In summary, the IREE dispatch profiler accomplishes the following:

- Auto-generation of MLIR dispatches (e.g., matmul, batch_matmul, convolutions, fused dispatches).
- Compilation of generated MLIR dispatches into binaries (vmfb).
- Functional verification against Python-based reference implementations.
- Performance profiling and reporting.

## Definitions

- Operation: An operation structure captures and refers to the functional description of an operation. For example, a Matmul operation includes the datatype, layout, and matrix multiplication problem shape.
- Tuning Configuration: Tuning configurations are attributes applied to the IREE compilation flow that can alter the performance of the compiled dispatch without affecting its functionality.
- Dispatch: A dispatch is a combination of an operation and its corresponding tuning configuration.

## Auto-generation of MLIR Dispatches

IREE dispatch profiler provides [`generator.py`](generator.py) that can be used to generate dispatches. Please find a sample run below:

```bash
build-debug $ python3 ../iree/experimental/dispatch_profiler/generator.py 
[Generating]: ./generated/linalg/matmul/matmul_128x128x256_f16t_f16t_f16t/matmul_128x128x256_f16t_f16t_f16t.mlir
    Emitting tuning configuration : tile_config_128x128_64x4_tensorcore_mmasync
    Emitting tuning configuration : tile_config_128x128_32x5_tensorcore_mmasync
    Emitting tuning configuration : tile_config_128x64_32x5_tensorcore_mmasync
    Emitting tuning configuration : tile_config_64x64_64x5_tensorcore_mmasync
    Emitting tuning configuration : tile_config_64x64_32x10_tensorcore_mmasync
    ...
```

This creates a `generated` folder containing dispatches organized in folders as `mlir_dialect/operation_name/`. The folder includes an .mlir file with all the dispatches for an operation.

The `generator.py` script serves as a generator for implemented operation data types, using a predefined list of problem shapes. You can also provide specific matrix multiplication shapes of interest. Examples are provided below.

#### Generating user-specified matmul shape `768x512x1024`

```bash
python3 ../iree/experimental/dispatch_profiler/generator.py --problem-m=768 --problem-n=512 --problem-k=1024
...
[Generating]: ./generated/linalg/matmul/matmul_768x512x1024_f16t_f16t_f16t/matmul_768x512x1024_f16t_f16t_f16t.mlir
[Generating]: ./generated/linalg/matmul/matmul_768x512x1024_f32t_f32t_f32t/matmul_768x512x1024_f32t_f32t_f32t.mlir
...
```

#### Generate a user-specified sweep of matmul shapes

Generate matmuls where M ranges from 64 to 1024 in increments of 128, N varies from 64 to 1024 in steps of 128, and K is fixed at 4096.

```bash
$ python3 ../iree/experimental/dispatch_profiler/generator.py --problem-m=64:1024:128 --problem-n=64:1024:128 --problem-k=4096
...
```

## Compilation of generated MLIR dispatches into binaries (vmfb)

IREE dispatch profiler provies `compile.py` that trigges `iree-compile` with appropiate compilation flags. The output of `iree-compile` vmfb files are placed in `mlir_dialect/operation_path/operation_name.mlir`. The `compiler.py` uses all the possible cpus on your machine to compile all different generated mlir source files.

```bash
python3 ../iree/experimental/dispatch_profiler/compile.py
```

Compiles all the generated source mlir dispatches. One can check the generated dispatched folder to find the vmfb files.

```bash
$ ls ./generated/linalg/matmul/matmul_64x64x4096_f16t_f16t_f16t/
iree_compile_cmd_stdout.mlir  matmul_64x64x4096_f16t_f16t_f16t.mlir  matmul_64x64x4096_f16t_f16t_f16t_profile.vmfb  matmul_64x64x4096_f16t_f16t_f16t_verify.vmfb
```

## Functional verification and performance profiling

The tool provides [`profiler.py`](profiler.py) script which can be used to trigger both verification and profiler for all the compiled dispatches. Please find some example profiling commandlines below:

### Functional verification and performance profiling of a _single_ dispatch

```
$ python3 ../iree/experimental/dispatch_profiler/profiler.py --dispatches=matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync --verification-enabled=true --profiling-enabled=true
---------------------------------------------------------------- 
Dispatch      : matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_3456x1024x2048_f16t_f16t_f16t
Configuration : tile_config_128x128_32x5_tensorcore_mmasync
Arguments     : --batch_count=1 --m=3456 --n=1024 --k=2048 --lhs=f16t --rhs=f16t --result=f16t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : SUCCESS
Runtime(ms)   : 0.062
GFLOPs        : 233798.62
```

### Performance profiling _single_ dispatch

Verification, particularly for large matrix multiplications, can be time-consuming when using a CPU-based numpy reference. To prioritize profiling speed and when functional correctness is assured, disable verification using `--verification-enabled=false`.

```bash
 python3 ../iree/experimental/dispatch_profiler/profiler.py --dispatches=matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync --verification-enabled=false --profiling-enabled=true
```

### Performance profile _single_ operation and _sweep_ tunning configurations

The `--dispatch` option accepts a comma-separated list of regex patterns to profile all tuning configurations generated for a operation. The command-line argument is formatted as `--dispatch=<regex>,<regex>`. Additionally, you can export the profiled output to a CSV file for further analysis using `--output=<filepath>`.

```bash
$ python3 ../iree/experimental/dispatch_profiler/profiler.py --dispatches=matmul_3456x1024x2048_f16t_f16t_f16t_*_tensorcore_mmasync --verification-enabled=false --profiling-enabled=true --output=data.csv
---------------------------------------------------------------- 
Dispatch      : matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x256_32x3_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_3456x1024x2048_f16t_f16t_f16t
Configuration : tile_config_128x256_32x3_tensorcore_mmasync
Arguments     : --batch_count=1 --m=3456 --n=1024 --k=2048 --lhs=f16t --rhs=f16t --result=f16t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : Not verified
Runtime(ms)   : 0.062
GFLOPs        : 233798.62
---------------------------------------------------------------- 
Dispatch      : matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_64x4_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_3456x1024x2048_f16t_f16t_f16t
Configuration : tile_config_128x128_64x4_tensorcore_mmasync
Arguments     : --batch_count=1 --m=3456 --n=1024 --k=2048 --lhs=f16t --rhs=f16t --result=f16t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : Not verified
Runtime(ms)   : 0.064
GFLOPs        : 226492.42
---------------------------------------------------------------- 
...
----------------------------------------------------------------
Dispatch      : matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_64x64_32x10_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_3456x1024x2048_f16t_f16t_f16t
Configuration : tile_config_64x64_32x10_tensorcore_mmasync
Arguments     : --batch_count=1 --m=3456 --n=1024 --k=2048 --lhs=f16t --rhs=f16t --result=f16t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : Not verified
Runtime(ms)   : 0.103
GFLOPs        : 140733.15

Writing performance report to data.csv

```

### Performance profiling a large matmul targetting _F16_ and _F32_ datatype

Another example showcasing the use of `--dispatch` to profile a matmul_3456x1024x2048 targetting F16 and F32 NVIDIA A100 Tensor Cores.

```bash
$ python3 ../iree/experimental/dispatch_profiler/profiler.py --dispatches=matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync,matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_128x128_16x5_tensorcore_mmasync 
---------------------------------------------------------------- 
Dispatch      : matmul_3456x1024x2048_f16t_f16t_f16t_tile_config_128x128_32x5_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_3456x1024x2048_f16t_f16t_f16t
Configuration : tile_config_128x128_32x5_tensorcore_mmasync
Arguments     : --batch_count=1 --m=3456 --n=1024 --k=2048 --lhs=f16t --rhs=f16t --result=f16t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : SUCCESS
Runtime(ms)   : 0.062
GFLOPs        : 233798.62
---------------------------------------------------------------- 
Dispatch      : matmul_3456x1024x2048_f32t_f32t_f32t_tile_config_128x128_16x5_tensorcore_mmasync
Provider      : IREE Codegen
OpKind        : OperationKind.Matmul
Operation     : matmul_3456x1024x2048_f32t_f32t_f32t
Configuration : tile_config_128x128_16x5_tensorcore_mmasync
Arguments     : --batch_count=1 --m=3456 --n=1024 --k=2048 --lhs=f32t --rhs=f32t --result=f32t
                --split_k_mode=N/A --split_k_slices=N/A
Verification  : SUCCESS
Runtime(ms)   : 0.122
GFLOPs        : 118815.69
----------------------------------------------------------------
```
