// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-convert-forall-to-generic-nest-workgroup))" --allow-unregistered-dialect --split-input-file | FileCheck %s

// Test that workgroup scope creates 1 id/count pair with linearized scope.

// CHECK-LABEL: func.func @test_workgroup_scope
// CHECK:       pcf.generic
// CHECK-SAME:    scope(#iree_codegen.workgroup_scope<linearize>)
// CHECK:         execute(%{{.*}})[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// Chunk size computed from total iterations / worker count.
// CHECK:         %[[CHUNK:.+]] = arith.ceildivui
// Start = id * chunk_size.
// CHECK:         %[[START:.+]] = arith.muli %[[ID]], %[[CHUNK]]
// End = min(start + chunk_size, total).
// CHECK:         %[[END_RAW:.+]] = arith.addi %[[START]], %[[CHUNK]]
// CHECK:         %[[END:.+]] = arith.minui %[[END_RAW]]
// CHECK:         scf.forall (%[[IV:.+]]) = (%[[START]]) to (%[[END]])
// CHECK:           "foo.body"(%[[IV]])
// CHECK:           pcf.write_slice
// CHECK:         pcf.return
func.func @test_workgroup_scope(%init: tensor<64xf32>) -> tensor<64xf32> {
  %result = scf.forall (%i) in (64) shared_outs(%out = %init) -> tensor<64xf32> {
    "foo.body"(%i) : (index) -> ()
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<64xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<64xf32>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  return %result : tensor<64xf32>
}
