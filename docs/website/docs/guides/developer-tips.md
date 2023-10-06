---
icon: material/lightbulb-on
---

# IREE developer tips and tricks

The IREE compiler is built using [MLIR](https://mlir.llvm.org/), so it naturally
supports the common
[MLIR debugging workflows](https://mlir.llvm.org/getting_started/Debugging/).
For areas where IREE differentiates itself, this page lists other helpful tips
and tricks.

## Setting compiler options

Tools such as `iree-compile` take options via command-line flags. Pass `--help`
to see the full list:

```console
$ iree-compile --help

OVERVIEW: IREE compilation driver

USAGE: iree-compile [options] <input file or '-' for stdin>

OPTIONS:
  ...
```

!!! tip "Tip - Options and the Python bindings"

    If you are using the Python bindings, options can be passed via the
    `extra_args=["--flag"]` argument:

    ``` python hl_lines="12"
    import iree.compiler as ireec

    input_mlir = """
    func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
      %result = math.absf %input : tensor<f32>
      return %result : tensor<f32>
    }"""

    compiled_module = ireec.tools.compile_str(
        input_mlir,
        target_backends=["llvm-cpu"],
        extra_args=["--mlir-timing"])
    ```

## Inspecting `.vmfb` files

The IREE compiler generates [FlatBuffer](https://flatbuffers.dev/) files using
the `.vmfb` file extension, short for "Virtual Machine FlatBuffer", which can
then be loaded and executed using IREE's runtime.

??? info "Info - other output formats"

    The IREE compiler can output different formats with the ``--output-format=`
    flag:

    Flag value | Output
    ---------- | ------
    `--output-format=vm-bytecode` (default) | VM Bytecode (`.vmfb`) files
    `--output-format=vm-c` | C source modules

    VM Bytecode files are usable across a range of deployment scenarios, while
    C source modules provide low level connection points for constrained
    environments like bare metal platforms.

By default, `.vmfb` files can be opened as zip files: (1)
{ .annotate }

1. Setting `--iree-vm-emit-polyglot-zip=false` will disable this feature and
   decrease file size slightly

```console
$ unzip -d simple_abs_cpu ./simple_abs_cpu.vmfb

Archive:  ./simple_abs_cpu.vmfb
  extracting: simple_abs_cpu/module.fb
  extracting: simple_abs_cpu/abs_dispatch_0_system_elf_x86_64.so
```

The embedded binary (here an ELF shared object with CPU code) can be parsed by
standard tools:

```console
$ readelf -Ws ./simple_abs_cpu/abs_dispatch_0_system_elf_x86_64.so

Symbol table '.dynsym' contains 2 entries:
  Num:    Value          Size Type    Bind   Vis      Ndx Name
    0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
    1: 0000000000001760    17 FUNC    GLOBAL DEFAULT    7 iree_hal_executable_library_query

Symbol table '.symtab' contains 42 entries:
  Num:    Value          Size Type    Bind   Vis      Ndx Name
    0: 0000000000000000     0 NOTYPE  LOCAL  DEFAULT  UND
    1: 0000000000000000     0 FILE    LOCAL  DEFAULT  ABS abs_dispatch_0
    2: 0000000000001730    34 FUNC    LOCAL  DEFAULT    7 abs_dispatch_0_generic
    3: 00000000000034c0    80 OBJECT  LOCAL  DEFAULT    8 iree_hal_executable_library_query_v0
    4: 0000000000001780   111 FUNC    LOCAL  DEFAULT    7 iree_h2f_ieee
    5: 00000000000017f0   207 FUNC    LOCAL  DEFAULT    7 iree_f2h_ieee
    ...
```

The `iree-dump-module` tool can also be used to see information about a given
`.vmfb` file:

```console
$ iree-dump-module simple_abs.vmfb

//===---------------------------------------------------------------------===//
// @module : version 0
//===---------------------------------------------------------------------===//

Required Types:
  [  0] i32
  [  1] i64
  [  2] !hal.allocator
  [  3] !hal.buffer
  ...

Module Dependencies:
  hal, version >= 0, required

Imported Functions:
  [  0] hal.ex.shared_device() -> (!vm.ref<?>)
  [  1] hal.allocator.allocate(!vm.ref<?>, i32, i32, i64) -> (!vm.ref<?>)
  ...

Exported Functions:
  [  0] abs(!vm.ref<?>) -> (!vm.ref<?>)
  [  1] __init() -> ()

...
```

## Dumping executable files

The `--iree-hal-dump-executable-*` flags instruct the compiler to save files
related to "executable translation" (code generation for a specific hardware
target) into a directory of your choosing. If you are interested in seeing which
operations in your input program were fused into a compute kernel or what device
code was generated for a given program structure, these flags are a great
starting point.

Flag | Files dumped
---- | ------------
`iree-hal-dump-executable-files-to` | All files (meta-flag)
`iree-hal-dump-executable-sources-to` | Source `.mlir` files prior to HAL compilation
`iree-hal-dump-executable-intermediates-to` | Intermediate files (e.g. `.o` files, `.mlir` stages)
`iree-hal-dump-executable-binaries-to` | Binary files (e.g. `.so`, `.spv`, `.ptx`), as used in the `.vmfb`
`iree-hal-dump-executable-benchmarks-to` | Standalone benchmark files for `iree-benchmark-module`

=== "CPU"

    ```console hl_lines="5 6"
    $ mkdir -p /tmp/iree/simple_abs/

    $ iree-compile simple_abs.mlir \
      --iree-hal-target-backends=llvm-cpu \
      --iree-llvmcpu-link-embedded=false \
      --iree-hal-dump-executable-files-to=/tmp/iree/simple_abs \
      -o /tmp/iree/simple_abs/simple_abs_cpu.vmfb

    $ ls /tmp/iree/simple_abs

    module_abs_dispatch_0.mlir
    module_abs_dispatch_0_system_elf_x86_64_benchmark.mlir
    module_abs_dispatch_0_system_elf_x86_64.codegen.bc
    module_abs_dispatch_0_system_elf_x86_64.linked.bc
    module_abs_dispatch_0_system_elf_x86_64.optimized.bc
    module_abs_dispatch_0_system_elf_x86_64.s
    module_abs_dispatch_0_system_elf_x86_64.so
    simple_abs_cpu.vmfb
    ```

    !!! tip "Tip - Embedded and system linking"

        The default value of `--iree-llvmcpu-link-embedded=true` generates
        embedded ELF files. By disabling that flag, the compiler will produce
        platform-standard `.so` files for Linux, `.dll` files for Windows, etc.
        While embedded ELF files can be smaller and more portable, inspection of
        artifacts is easier with platform-standard shared object files.

    ??? tip "Tip - Disassembling `.bc` files with `llvm-dis`"

        The `.bc` intermediate files use the
        [LLVM BitCode](https://llvm.org/docs/BitCodeFormat.html) format, which
        can be disassembled using
        [`llvm-dis`](https://llvm.org/docs/CommandGuide/llvm-dis.html):

        ```console
        // Build `llvm-dis` from source as needed:
        $ cmake --build iree-build/ --target llvm-dis
        $ iree-build/llvm-project/bin/llvm-dis --help

        $ cd /tmp/iree/simple_abs/
        $ llvm-dis module_abs_dispatch_0_system_elf_x86_64.codegen.bc
        $ cat module_abs_dispatch_0_system_elf_x86_64.codegen.ll

        ; ModuleID = 'module_abs_dispatch_0_system_elf_x86_64.codegen.bc'
        source_filename = "abs_dispatch_0"
        target triple = "x86_64-linux-gnu"

        %iree_hal_executable_library_header_t = type { i32, ptr, i32, i32 }
        %iree_hal_executable_dispatch_attrs_v0_t = type { i16, i16 }

        ...

        define internal i32 @abs_dispatch_0_generic(
            ptr noalias nonnull align 16 %0,
            ptr noalias nonnull align 16 %1,
            ptr noalias nonnull align 16 %2) #0 {
          %4 = load %iree_hal_executable_dispatch_state_v0_t, ptr %1, align 8,
          %5 = extractvalue %iree_hal_executable_dispatch_state_v0_t %4, 10,
          %6 = load ptr, ptr %5, align 8,
          %7 = ptrtoint ptr %6 to i64,
          %8 = and i64 %7, 63,
          %9 = icmp eq i64 %8, 0,
          call void @llvm.assume(i1 %9),
          %10 = load %iree_hal_executable_dispatch_state_v0_t, ptr %1, align 8,
          %11 = extractvalue %iree_hal_executable_dispatch_state_v0_t %10, 10,
          %12 = getelementptr ptr, ptr %11, i32 1,
          %13 = load ptr, ptr %12, align 8,
          %14 = ptrtoint ptr %13 to i64,
          %15 = and i64 %14, 63,
          %16 = icmp eq i64 %15, 0,
          call void @llvm.assume(i1 %16),
          %17 = load float, ptr %6, align 4,
          %18 = call float @llvm.fabs.f32(float %17),
          store float %18, ptr %13, align 4,
          ret i32 0,
        }

        ...
        ```

=== "GPU - Vulkan"

    ```console hl_lines="5"
    $ mkdir -p /tmp/iree/simple_abs/

    $ iree-compile simple_abs.mlir \
      --iree-hal-target-backends=vulkan-spirv \
      --iree-hal-dump-executable-files-to=/tmp/iree/simple_abs \
      -o /tmp/iree/simple_abs/simple_abs_vulkan.vmfb

    $ ls /tmp/iree/simple_abs

    module_abs_dispatch_0.mlir
    module_abs_dispatch_0_vulkan_spirv_fb_benchmark.mlir
    module_abs_dispatch_0_vulkan_spirv_fb.mlir
    module_abs_dispatch_0_vulkan_spirv_fb.spv
    simple_abs_vulkan.vmfb
    ```

    ??? tip "Tip - Disassembling `.spv` files with `spirv-dis`"

        The `.spv` files use the
        [SPIR-V](https://registry.khronos.org/SPIR-V/) binary format, which can
        be disassembled using `spirv-dis` from
        [SPIR-V Tools](https://github.com/KhronosGroup/SPIRV-Tools):

        ```console
        $ cd /tmp/iree/simple_abs/
        $ spirv-dis module_abs_dispatch_0_vulkan_spirv_fb.spv

        ; SPIR-V
        ; Version: 1.0
        ; Generator: Khronos; 22
        ; Bound: 20
        ; Schema: 0
                      OpCapability Shader
                      OpExtension "SPV_KHR_storage_buffer_storage_class"
                %18 = OpExtInstImport "GLSL.std.450"
                      OpMemoryModel Logical GLSL450
                      OpEntryPoint GLCompute %abs_dispatch_0_generic "abs_dispatch_0_generic"
                      OpExecutionMode %abs_dispatch_0_generic LocalSize 1 1 1
                      OpName %__resource_var_0_0_ "__resource_var_0_0_"
                      OpName %__resource_var_0_1_ "__resource_var_0_1_"
                      OpName %abs_dispatch_0_generic "abs_dispatch_0_generic"
                      OpDecorate %_arr_float_uint_1 ArrayStride 4
                      OpMemberDecorate %_struct_2 0 Offset 0
                      OpDecorate %_struct_2 Block
                      OpDecorate %__resource_var_0_0_ Binding 0
                      OpDecorate %__resource_var_0_0_ DescriptorSet 0
                      OpDecorate %__resource_var_0_1_ Binding 1
                      OpDecorate %__resource_var_0_1_ DescriptorSet 0
              %float = OpTypeFloat 32
              %uint = OpTypeInt 32 0
            %uint_1 = OpConstant %uint 1
        %_arr_float_uint_1 = OpTypeArray %float %uint_1
          %_struct_2 = OpTypeStruct %_arr_float_uint_1
        %_ptr_StorageBuffer__struct_2 = OpTypePointer StorageBuffer %_struct_2
        %__resource_var_0_0_ = OpVariable %_ptr_StorageBuffer__struct_2 StorageBuffer
        %__resource_var_0_1_ = OpVariable %_ptr_StorageBuffer__struct_2 StorageBuffer
              %void = OpTypeVoid
                  %9 = OpTypeFunction %void
            %uint_0 = OpConstant %uint 0
        %_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
        %abs_dispatch_0_generic = OpFunction %void None %9
                %12 = OpLabel
                %15 = OpAccessChain %_ptr_StorageBuffer_float %__resource_var_0_0_ %uint_0 %uint_0
                %16 = OpLoad %float %15
                %17 = OpExtInst %float %18 FAbs %16
                %19 = OpAccessChain %_ptr_StorageBuffer_float %__resource_var_0_1_ %uint_0 %uint_0
                      OpStore %19 %17
                      OpReturn
                      OpFunctionEnd
        ```

=== "GPU - CUDA"

    ```console hl_lines="5"
    $ mkdir -p /tmp/iree/simple_abs/

    $ iree-compile simple_abs.mlir \
      --iree-hal-target-backends=cuda \
      --iree-hal-dump-executable-files-to=/tmp/iree/simple_abs \
      -o /tmp/iree/simple_abs/simple_abs_cuda.vmfb

    $ ls /tmp/iree/simple_abs

    module_abs_dispatch_0_cuda_nvptx_fb_benchmark.mlir
    module_abs_dispatch_0_cuda_nvptx_fb.codegen.bc
    module_abs_dispatch_0_cuda_nvptx_fb.linked.bc
    module_abs_dispatch_0_cuda_nvptx_fb.optimized.bc
    module_abs_dispatch_0_cuda_nvptx_fb.ptx
    module_abs_dispatch_0.mlir
    simple_abs_cuda.vmfb
    ```

    ??? tip "Tip - Disassembling `.bc` files with `llvm-dis`"

        The `.bc` intermediate files use the
        [LLVM BitCode](https://llvm.org/docs/BitCodeFormat.html) format, which
        can be disassembled using
        [`llvm-dis`](https://llvm.org/docs/CommandGuide/llvm-dis.html):

        ```console
        // Build `llvm-dis` from source as needed:
        $ cmake --build iree-build/ --target llvm-dis
        $ iree-build/llvm-project/bin/llvm-dis --help

        $ cd /tmp/iree/simple_abs/
        $ llvm-dis module_abs_dispatch_0_cuda_nvptx_fb.codegen.bc
        $ cat module_abs_dispatch_0_cuda_nvptx_fb.codegen.ll

        ; ModuleID = 'module_abs_dispatch_0_cuda_nvptx_fb.codegen.bc'
        source_filename = "abs_dispatch_0"

        declare ptr @malloc(i64)

        declare void @free(ptr)

        declare float @__nv_fabsf(float)

        define void @abs_dispatch_0_generic(ptr noalias readonly align 16 %0, ptr noalias align 16 %1) {
          %3 = ptrtoint ptr %0 to i64
          %4 = and i64 %3, 63
          %5 = icmp eq i64 %4, 0
          call void @llvm.assume(i1 %5)
          %6 = ptrtoint ptr %1 to i64
          %7 = and i64 %6, 63
          %8 = icmp eq i64 %7, 0
          call void @llvm.assume(i1 %8)
          %9 = load float, ptr %0, align 4
          %10 = call float @__nv_fabsf(float %9)
          store float %10, ptr %1, align 4
          ret void
        }

        !nvvm.annotations = !{!0, !1, !2, !3}

        !0 = !{ptr @abs_dispatch_0_generic, !"kernel", i32 1}
        !1 = !{ptr @abs_dispatch_0_generic, !"maxntidx", i32 1}
        !2 = !{ptr @abs_dispatch_0_generic, !"maxntidy", i32 1}
        !3 = !{ptr @abs_dispatch_0_generic, !"maxntidz", i32 1}
        ```

<!-- TODO(scotttodd): Link to a playground Colab notebook that dumps files? -->

## Compiling phase by phase

IREE compiles programs through a series of broad phases:

``` mermaid
graph LR
  accTitle: Compilation phases overview
  accDescr: Input to ABI to Flow to Stream to HAL to VM

  A([Input])
  A --> B([ABI])
  B --> C([Flow])
  C --> D([Stream])
  D --> E([HAL])
  E --> F([VM])
```

??? tip "Tip - available phases"

    These are the phase names available for use with the `--compile-to` and
    `--compile-from` flags described below:

    | Phase name | Description |
    | ---------- | ----------- |
    `input` | Performs input processing and lowering into core IREE input dialects (linalg/etc)
    `abi` | Adjusts the program ABI for the specified execution environment
    `preprocessing` | Applies customizable `preprocessing` prior to FLow/Stream/HAL/VM
    `flow` | Models execution data flow and partitioning using the `flow` dialect
    `stream` | Models execution partitioning and scheduling using the `stream` dialect
    `executable-sources` | Prepares `hal` dialect executables for translation, prior to codegen
    `executable-targets` | Runs code generation for `hal` dialect executables
    `hal` | Finishes `hal` dialect processing
    `vm` | Lowers to IREE's abstract virtual machine using the `vm` dialect
    `end` | Completes the full compilation pipeline

    For an accurate list of phases, see the source code or check the help output
    with a command such as:

    ```shell
    iree-compile --help | sed -n '/--compile-to/,/--/p' | head -n -1
    ```

You can output a program snapshot at intermediate phases with the
`--compile-to=<phase name>` flag:

```console
$ cat simple_abs.mlir

func.func @abs(%input : tensor<f32>) -> (tensor<f32>) {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}

$ iree-compile simple_abs.mlir --compile-to=abi

module {
  func.func @abs(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub} {
    %0 = hal.tensor.import %arg0 "input 0" : !hal.buffer_view -> tensor<f32>
    %1 = math.absf %0 : tensor<f32>
    %2 = hal.tensor.export %1 "output 0" : tensor<f32> -> !hal.buffer_view
    return %2 : !hal.buffer_view
  }
}
```

This is similar to the `--mlir-print-ir-after=` flag, but at clearly defined
pipeline phases.

Compilation can be continued from any intermediate phase. This allows for
interative workflows - compile to a phase, make edits to the `.mlir` file,
then resume compilation and continue through the pipeline:

```console
$ iree-compile simple_abs.mlir --compile-to=abi -o simple_abs_abi.mlir

$ sed \
  -e 's/math.absf/math.exp/' \
  -e 's/@abs/@exp/' \
  simple_abs_abi.mlir > simple_exp_abi.mlir

$ iree-compile simple_exp_abi.mlir \
  --iree-hal-target-backends=llvm-cpu \
  -o simple_exp_cpu.vmfb
```

or explicitly resume from an intermediate phase with `--compile-from=<phase name>`:

```console
$ iree-compile simple_exp_abi.mlir \
  --iree-hal-target-backends=llvm-cpu \
  --compile-from=abi \
  -o simple_exp_cpu.vmfb
```
