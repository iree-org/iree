// RUN: iree-compile --iree-hal-target-backends=llvm-cpu --compile-to=preprocessing \
// RUN:   --iree-preprocessing-pass-pipeline="builtin.module(func.func(iree-preprocessing-convert-conv2d-to-img2col,iree-preprocessing-pad-linalg-ops{pad-size=16}))" \
// RUN:   --mlir-print-ir-after=iree-preprocessing-convert-conv2d-to-img2col --mlir-print-ir-after=iree-preprocessing-pad-linalg-ops %s 2>&1 \
// RUN:   | FileCheck %s

func.func @test(%arg0 : tensor<10x20xf32>, %arg1 : tensor<20x30xf32>, %arg2 : tensor<10x30xf32>) -> tensor<10x30xf32> {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<10x20xf32>, tensor<20x30xf32>)
      outs(%arg2 : tensor<10x30xf32>) -> tensor<10x30xf32>
  return %0 : tensor<10x30xf32>
}
// Just check that the pass runs, and that the compilation finishes
//       CHECK: ConvertConv2DToImg2Col (iree-preprocessing-convert-conv2d-to-img2col)
//       CHECK: PadLinalgOps (iree-preprocessing-pad-linalg-ops)
// CHECK-LABEL: module
//  CHECK-NEXT:   func.func @test(
//   CHECK-DAG:     %[[ARG0:.+]] = hal.tensor.import %{{[a-zA-Z0-9]+}} "input 0" : !hal.buffer_view -> tensor<10x20xf32>
//   CHECK-DAG:     %[[ARG1:.+]] = hal.tensor.import %{{[a-zA-Z0-9]+}} "input 1" : !hal.buffer_view -> tensor<20x30xf32>
//   CHECK-DAG:     %[[ARG2:.+]] = hal.tensor.import %{{[a-zA-Z0-9]+}} "input 2" : !hal.buffer_view -> tensor<10x30xf32>
//   CHECK-DAG:     %[[PAD0:.+]] = tensor.pad %[[ARG0]] low[0, 0] high[6, 12]
//   CHECK-DAG:     %[[PAD1:.+]] = tensor.pad %[[ARG1]] low[0, 0] high[12, 2]
//   CHECK-DAG:     %[[PAD2:.+]] = tensor.pad %[[ARG2]] low[0, 0] high[6, 2]
//       CHECK:     %[[PADDED:.+]] = linalg.matmul
//  CHECK-SAME:         ins(%[[PAD0]], %[[PAD1]] :
//  CHECK-SAME:         outs(%[[PAD2]] :
//       CHECK:      %[[SLICE:.+]] = tensor.extract_slice %[[PADDED]][0, 0] [10, 30] [1, 1]
//       CHECK:      hal.tensor.export %[[SLICE]]
//   CHECK-NOT: vm.module
