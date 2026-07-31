// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-compile --iree-hal-target-device=local \
// RUN:   --iree-hal-local-target-device-backends=llvm-cpu \
// RUN:   --iree-llvmcpu-target-triple=x86_64-unknown-unknown-eabi-elf \
// RUN:   --iree-llvmcpu-target-cpu=znver4 \
// RUN:   --iree-opt-data-tiling --iree-llvmcpu-enable-ukernels=all \
// RUN:   --iree-hal-dump-executable-intermediates-to=%t %s -o /dev/null
// RUN: cat %t/*.linked.ll | FileCheck %s --check-prefix=LINKED
// RUN: cat %t/*.optimized.ll | FileCheck %s --check-prefix=OPT

// Regression test for ukernels getting inlined into the dispatch function.
//
// Ukernels rely fundamentally on always getting inlined, for their logic to
// specialize at compile time (data types, SIMD ISA variant, ...) so that the
// unused code paths can then be DCE'd. loadUKernelBitcode() marks every
// ukernel function `alwaysinline` to that end.
//
// That marker is not on its own sufficient: LLVM refuses to inline a callee
// whose target attributes are incompatible with the caller's.

// The two dumps bracket the LLVM pipeline: `.linked.ll` is the module right
// after the ukernel bitcode is linked in, `.optimized.ll` is it after the
// optimization pipeline ran.

// First check that this test is actually exercising a ukernel at all, so that
// it cannot silently pass by way of ukernel selection never having happened.
// LINKED: call {{.*}}@iree_uk_mmt4d(

// Then check that no direct call to a ukernel entry point survived, i.e. they
// were all inlined into the dispatch function. Note that the individual tile
// functions are still reached through a function pointer chosen at runtime and
// are deliberately not inlined; those are indirect calls and so do not name an
// `@iree_uk_` symbol here.
// OPT: define {{.*}}@matmul_dispatch_0
// OPT-NOT: call {{.*}}@iree_uk_

func.func @matmul(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>)
    -> tensor<128x128xf32> {
  %c0 = arith.constant 0.0 : f32
  %init = tensor.empty() : tensor<128x128xf32>
  %fill = linalg.fill ins(%c0 : f32) outs(%init : tensor<128x128xf32>)
      -> tensor<128x128xf32>
  %result = linalg.matmul ins(%a, %b : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%fill : tensor<128x128xf32>) -> tensor<128x128xf32>
  return %result : tensor<128x128xf32>
}
