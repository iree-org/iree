executable_library_benchmark
---

Use `iree/hal/local/executable_library_benchmark --help` for more information.
This tool is intended for CPU codegen developers only and cuts into the system
at the lowest level possible: if you wish this was automated or easier to use
then you should be looking elsewhere in the stack.

The best inputs for this are those that result in a single dispatch function
so that you don't have to look hard to figure out what all the flags are. As
the fusion is compiler-driven this can be tricky to ensure.

Keep in mind that in IREE the generated HAL executables and the functions they
contain are an internal implementation detail of the compiler. Using this tool
is effectively the same as taking some random assembly dump of a C program and
trying to call one of the private functions inside of it: it's opaque,
ever-changing, and unfriendly for a reason!

---

### Full example using the files checked in to the repo

Start here to ensure you have a working build and see the expected output:

```
iree/hal/local/executable_library_benchmark \
    --executable_format=embedded-elf \
    --executable_file=iree/hal/local/elf/testdata/elementwise_mul_x86_64.so \
    --entry_point=0 \
    --workgroup_count_x=1 \
    --workgroup_count_y=1 \
    --workgroup_count_z=1 \
    --workgroup_size_x=1 \
    --workgroup_size_y=1 \
    --workgroup_size_z=1 \
    --binding=4xf32=1,2,3,4 \
    --binding=4xf32=100,200,300,400 \
    --binding=4xf32=0,0,0,0
```

```
---------------------------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations UserCounters...
---------------------------------------------------------------------------------------------
BM_dispatch/process_time/real_time       90.7 ns         90.9 ns      7739262 items_per_second=11.0312M/s
```

---

It can be helpful to put the flags in flagfiles (newline separated):

```
iree/hal/local/executable_library_benchmark --flagfile=my_flags.txt
```

For an example, the flags for an x86-64 run of a simple element-wise multiply:

```
iree/hal/local/executable_library_benchmark --flagfile=iree/hal/local/testdata/elementwise_mul_benchmark.txt
```

---

### Running standalone HAL executables

This approach uses an explicitly specified HAL executable without any associated
host code. When doing this the pipeline layout specifying the bindings and
push constants is chosen by the user instead of being automatically derived by
the compiler. The design of the layout can have performance implications and
it's important to try to match the kind of layout the compiler would produce or
ensure that what's being tested is relatively immune to the potential effects
(having enough work per workgroup, etc).

1. Hand-author a `hal.executable.source` op or extract a `hal.executable`

See [iree/hal/local/testdata/elementwise_mul.mlir](iree/hal/local/testdata/elementwise_mul.mlir)
for an example of the former that allows for the same source to be retargeted
to many different formats/architectures.

2. Translate the executable into the binary form consumed by the IREE loaders:

```
iree-compile \
    --compile-mode=hal-executable \
    iree/hal/local/testdata/elementwise_mul.mlir \
    -o=elementwise_mul.so \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-debug-symbols=false \
    --iree-llvmcpu-target-triple=x86_64-pc-linux-elf
```

Note that the architecture and other related LLVM flags must be specified by the
user. Some examples can be seen in [iree/hal/local/testdata/generate.sh](iree/hal/local/testdata/generate.sh).

3. Setup flags

Use the above example flagfile as a template or read below for details on how
to map the parameters. You'll need to specify the executable file and entry
point, the workgroup parameters, and any bindings and push constants used for
I/O.

---

### Running executables from full user modules

This approach extracts the embedded executable files contained within a full
IREE module and allows for benchmarking of any of them by using the
`--entry_point=` flag to select the executable. It's important to remember that
the exact set of bindings and parameters are implementation details of the
compiler and subject to change at any time - when using this approach one must
inspect the IR to find the proper way to call their kernels.

1. Build your module with the flags you want for your target architecture:

```
iree-compile \
    --iree-input-type=stablehlo \
    iree/samples/simple_embedding/simple_embedding_test.mlir \
    -o=module.vmfb \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-debug-symbols=false \
    --iree-llvmcpu-target-triple=x86_64-pc-linux-elf \
    --mlir-print-ir-after-all \
    >module_dump.mlir 2>&1
```

This produces `module_dump.mlir` containing the IR at various stages.
You'll need this to determine the flags used to invoke the dispatch.

2. Extract the executable shared object from the module:

```
unzip module.vmfb
```

This (today) results in a single extracted file you pass to the tool:

```
--executable_format=embedded-elf
--executable_file=_simple_mul_dispatch_0_llvm_binary_ex_elf.so
```

3. Find `ResolveExportOrdinalsPass` and look for the dispatch:

```mlir
  hal.command_buffer.dispatch<%cmd : !hal.command_buffer>
      target(%3 : !hal.executable)[1]
      workgroups([%c5, %c6, %c7])
```

This maps to the following flags defining the executable entry point and counts:

```
--entry_point=1
--workgroup_count_x=5
--workgroup_count_y=6
--workgroup_count_z=7
```

4. Look up in the IR from that for where bindings are specified:

```mlir
  hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer>
      layout(%0 : !hal.pipeline_layout)[%c0]
      bindings([
        %c0 = (%buffer : !hal.buffer)[%c0, %c16],
        %c1 = (%buffer_0 : !hal.buffer)[%c0, %c16],
        %c2 = (%buffer_1 : !hal.buffer)[%c0, %c16]
      ])
```

This is 3 buffers of 16 bytes each, which is enough to call most things:

```
--binding=16xi8
--binding=16xi8
--binding=16xi8
```

If you want to provide real data then you can look for the `flow.executable`
with the `!flow.dispatch.tensor` operands:

```mlir
  func.func @simple_mul_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:4xf32>,
                              %arg1: !flow.dispatch.tensor<readonly:4xf32>,
                              %arg2: !flow.dispatch.tensor<writeonly:4xf32>) {
```

Now we know each binding is 4 floats and can get more realistic test data:

```
--binding=4xf32=1,2,3,4
--binding=4xf32=100,200,300,400
--binding=4xf32=0,0,0,0
```

**Note that multiple tensors may alias to a single binding** - including
tensors of differing data types. It's best to use the generic
`[byte length]xi8` form above instead of trying to match the types in all but
the most simple scenarios. You don't want to be using this tool to verify
results and the only time it should matter what the value of the inputs are is
if there is branching behavior inside the generated code itself. These are not
good candidates for this tool.

5. Look up in the IR to see the values of push constants, if required:

```mlir
  hal.command_buffer.push_constants<%cmd : !hal.command_buffer>
      layout(%0 : !hal.pipeline_layout)
      offset(0)
      values(%c1, %c2, %c3, %c4) : i32, i32, i32, i32
```

These are often shape dimensions but by this point they are hard to guess if
non-constant. This microbenchmarking approach is not generally suited for
things like this but in cases where you know the meaning you can provide values:

```
--push_constant=1
--push_constant=2
--push_constant=3
--push_constant=4
```
