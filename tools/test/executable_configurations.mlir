// RUN: iree-compile %s -o ignored.mlir \
// RUN:     --iree-hal-target-backends=vmvx \
// RUN:     --iree-hal-dump-executable-configurations-to=- | \
// RUN: iree-compile - -o /dev/null \
// RUN:     --compile-mode=hal-executable \
// RUN:     --mlir-print-ir-before=iree-hal-serialize-executables 2>&1 | \
// RUN: FileCheck %s

// This test relies on piping stdout and that there is only a single
// executable (otherwise we'd need to look at files and that's harder
// cross-platform). Real automation of this requires xargs: compile and dump a
// directory of .mlir sources by specifying a path to the dump flag instead
// of `-` (indicating stdout) and then ls | xargs them to iree-compile or
// iree-opt.
//
// Example of dumping per-dispatch executable configurations and compiling each
// to their platform binary form, dumping their MLIR prior to lowering into the
// backend representation (SPIR-V/LLVM-IR/etc):
//  iree-compile some_input.mlir -o ignored.mlir \
//      --iree-hal-target-backends=vmvx \
//      --iree-hal-dump-executable-configurations-to=configs/ | \
//  ls -1 sources/ | xargs -i sh -c "iree-compile configs/{}
//      --compile-mode=hal-executable
//      --mlir-print-ir-before=iree-hal-serialize-executables"
//
// NOTE: executable configurations are not runnable: they only exist to allow
// for iteration on executable translation. If you want to run them you need
// corresponding host code to dispatch them and can use benchmarks instead.
//
// If modifying the configs and wanting to see the changes in a full program the
// --iree-hal-substitute-executable-configurations-from= flag can be used to
// substitute one or more executables dumped with this command from a path or
// for individual executables one or more `executable_name=file.mlir` pairs can
// be repeated in `--iree-hal-substitute-executable-configuration=`.

func.func @abs(%input : tensor<f32>) -> tensor<f32> {
  %result = math.absf %input : tensor<f32>
  return %result : tensor<f32>
}

// CHECK: IR Dump Before SerializeExecutablesPass
// CHECK: hal.executable public @abs_dispatch_0
// CHECK:   hal.executable.variant public @vmvx_bytecode_fb
// CHECK:     vm.func private @abs_dispatch_0_generic
