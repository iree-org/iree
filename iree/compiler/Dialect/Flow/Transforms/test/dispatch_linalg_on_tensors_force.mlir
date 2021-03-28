// RUN: iree-opt -split-input-file -verify-diagnostics -iree-flow-dispatch-linalg-on-tensors-pass -iree-flow-dispatch-linalg-on-tensors-tile-sizes="1,2" -iree-flow-dispatch-linalg-on-tensors-enable-fusion-always -canonicalize -cse %s | IreeFileCheck %s


// CHECK-LABEL: func @tensor2
func @tensor2(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> attributes {iree.module.export}
{
  %f12 = constant 12.0 : f32

  // linalg.generic is fused inside the dispatch region and becomes a noop.
  // CHECK-NOT: generic
  %AA = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%A : tensor<?x?xf32>) {
    ^bb0(%a: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  // CHECK: flow.dispatch.workgroups
  // CHECK:  scf.for
  // CHECK:    scf.for
  // CHECK:      %[[AA:.*]] = linalg.generic
  // CHECK:      linalg.matmul{{.*}} ins(%[[AA]], %{{.}}
  %D = linalg.matmul ins(%AA, %B: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}

// -----

// CHECK-LABEL: func @tensor3
func @tensor3(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>)
  -> tensor<?x?xf32> attributes {iree.module.export}
{
  %f12 = constant 12.0 : f32

  // linalg.generic is fused inside the dispatch region and becomes a noop.
  // CHECK-NOT: generic
  %BB = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"] }
    outs(%B : tensor<?x?xf32>) {
    ^bb0(%b: f32):
      linalg.yield %f12 : f32
    } -> tensor<?x?xf32>

  // CHECK: flow.dispatch.workgroups
  // CHECK:  scf.for
  // CHECK:    scf.for
  // CHECK:      %[[BB:.*]] = linalg.generic{{.*}}
  // CHECK:      linalg.matmul{{.*}} ins(%{{.*}}, %[[BB]]
  %D = linalg.matmul ins(%A, %BB: tensor<?x?xf32>, tensor<?x?xf32>)
                    outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}
