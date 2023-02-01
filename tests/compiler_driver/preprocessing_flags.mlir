// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --output-format=vm-asm --iree-preprocessing-pass-pipeline="builtin.module(func.func(iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=16}))" %s --mlir-print-ir-after=iree-preprocessing-convert-conv2d-to-img2col --mlir-print-ir-after=iree-preprocessing-pad-linalg-ops 2>&1 | FileCheck %s

func.func @test(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%arg2 : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// Just check that the pass runs, and that the compilation finishes
// CHECK: ConvertConv2DToImg2Col (iree-preprocessing-convert-conv2d-to-img2col)
// CHECK: PadLinalgOps (iree-preprocessing-pad-linalg-ops)
// CHECK: vm.module
