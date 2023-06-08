# IREE compiler tips and tricks

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

By default, `.vmfb` files can be opened as zip files:

<!-- TODO(scotttodd): add annotation (insiders only), qualifying "default" with
                      `--iree-vm-emit-polyglot-zip=true`
-->

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

    !!! tip

        The default value of `--iree-llvmcpu-link-embedded=true` generates
        platform-agnostic ELF files. By disabling that flag, the compiler will
        produce `.so` files for Linux, `.dll` files for Windows, etc. While ELF
        files are more portable, inspection of compiled artifacts is easier with
        platform-specific shared object files.

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

    !!! tip

        Consider using tools like `spirv-dis` from the
        [SPIR-V Tools project](https://github.com/KhronosGroup/SPIRV-Tools) to
        interact with the `.spv` files.

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

Similarly, compilation can be continued from any intermediate phase. This allows
for interative workflows - compile to a phase, make edits to the `.mlir` file,
then resume compilation and continue through the pipeline.
