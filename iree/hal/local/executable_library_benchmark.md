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
    --executable_format=EX_ELF \
    --executable_file=iree/hal/local/elf/testdata/simple_mul_dispatch_x86_64.so \
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

### Constructing flags for other modules

1. Build your module with the flags you want for your target architecture:

```
iree-translate \
    -iree-mlir-to-vm-bytecode-module \
    -iree-input-type=mhlo \
    iree/samples/simple_embedding/simple_embedding_test.mlir \
    -o=module.vmfb \
    -iree-hal-target-backends=dylib-llvm-aot \
    -iree-llvm-link-embedded=true \
    -iree-llvm-debug-symbols=false \
    -iree-llvm-target-triple=x86_64-pc-linux-elf \
    -print-ir-after-all \
    >module_dump.mlir 2>&1
```

This produces `module_dump.mlir` containing the IR at various stages.
You'll need this to determine the flags used to invoke the dispatch.

2. Extract the executable shared object from the module:

```
7z e -aoa -bb0 -y module.vmfb
```

This (today) results in a single extracted file you pass to the tool:

```
--executable_format=EX_ELF
--executable_file=_simple_mul_dispatch_0_llvm_binary_ex_elf.so
```

3. Find `ResolveEntryPointOrdinalsPass` and look for the dispatch:

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
      layout(%0 : !hal.executable_layout)[%c0]
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
  func @simple_mul_dispatch_0(%arg0: !flow.dispatch.tensor<readonly:4xf32>,
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
      layout(%0 : !hal.executable_layout)
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

---

It can be helpful to put the flags in flagfiles (newline separated):

```
iree/hal/local/executable_library_benchmark --flagfile=my_flags.txt
```
