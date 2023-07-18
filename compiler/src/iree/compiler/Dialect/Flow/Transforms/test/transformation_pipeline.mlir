// RUN: iree-opt --split-input-file --iree-flow-transformation-pipeline %s | FileCheck %s

// CHECK-LABEL: @empty
func.func @empty() {
  // CHECK-NEXT: return
  return
}

// -----

func.func @elementwiseOps(%arg0 : tensor<4xf32>) -> tensor<4xf32> {
  %0 = arith.addf %arg0, %arg0 : tensor<4xf32>
  %1 = arith.subf %0, %arg0 : tensor<4xf32>
  %2 = arith.mulf %1, %arg0 : tensor<4xf32>
  return %2 : tensor<4xf32>
}

// CHECK-LABEL: flow.executable private @elementwiseOps_dispatch_0 {
//  CHECK-NEXT:   flow.executable.export public @elementwiseOps_dispatch_0{{.*}} workgroups() -> (index, index, index) {
//       CHECK:     %x, %y, %z = flow.dispatch.workgroup_count_from_slice
//       CHECK:     flow.return %x, %y, %z
//       CHECK:   module {
//  CHECK-NEXT:     func.func @elementwiseOps_dispatch_0{{.*}}(%arg0: !flow.dispatch.tensor<readonly:tensor<4xf32>>, %arg1: !flow.dispatch.tensor<writeonly:tensor<4xf32>>) {
//       CHECK:       %{{.+}} = linalg.generic
//       CHECK:         %{{.+}} = arith.addf %{{.+}}, %{{.+}} : f32
//  CHECK-NEXT:         %{{.+}} = arith.subf %{{.+}}, %{{.+}} : f32
//  CHECK-NEXT:         %{{.+}} = arith.mulf %{{.+}}, %{{.+}} : f32
//       CHECK: func.func @elementwiseOps(%arg0: tensor<4xf32>) -> tensor<4xf32> {
//  CHECK-NEXT:   %0 = flow.dispatch @elementwiseOps_dispatch_0::@elementwiseOps_dispatch_0{{.*}}(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
//  CHECK-NEXT:   return %0 : tensor<4xf32>
//  CHECK-NEXT: }
