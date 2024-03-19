// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetBackends=llvm-cpu},iree-transformation-pipeline)' %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetBackends=llvm-cpu},iree-transformation-pipeline)' --iree-llvmcpu-link-embedded=false %s | FileCheck %s

// When lowering to CPU code through LLVM, certain LLVM intrinsics require
// linking against libm (the standard C library of math functions, `-lm`).
//
// We require that our linked executables be free standing with no runtime
// dependencies, so we link implementations of the required functions into
// our executables prior to invoking a linker tool like lld. These
// implementations are mostly from musl (https://musl.libc.org/) and are
// bundled at iree/builtins/musl/.
//
// This test checks that the LLVM lowerings for certain operations are
// correctly covered by our linker configurations.
//
// See https://github.com/openxla/iree/issues/4717 for more details.

// CHECK: vm.func private @tanh
func.func @tanh(%input : tensor<f32>) -> (tensor<f32>) {
  // May introduce llvm.intr.fma (fmaf) during lowering
  %result = math.tanh %input : tensor<f32>
  return %result : tensor<f32>
}

// -----

// CHECK: vm.func private @ceil
func.func @ceil(%input : tensor<f32>) -> (tensor<f32>) {
  // May lower to llvm.intr.ceil (ceilf)
  %result = math.ceil %input : tensor<f32>
  return %result : tensor<f32>
}

// -----

// CHECK: vm.func private @floor
func.func @floor(%input : tensor<f32>) -> (tensor<f32>) {
  // May lower to llvm.intr.floor (floorf)
  %result = math.floor %input : tensor<f32>
  return %result : tensor<f32>
}

// CHECK: vm.func private @exp2
func.func @exp2(%input : tensor<f32>) -> (tensor<f32>) {
  // May lower to llvm.intr.exp2 (exp2f)
  %result = math.exp2 %input : tensor<f32>
  return %result : tensor<f32>
}

