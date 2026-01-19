// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-test-convert-forall-to-generic-nest-workgroup))" --split-input-file | FileCheck %s --check-prefix=CHECK-3IDS
// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-test-convert-forall-to-generic-nest-workgroup{linearize=true}))" --split-input-file | FileCheck %s --check-prefix=CHECK-1ID

// Test that workgroup scope (linearize=false) creates 3 id/count pairs via
// getNativeNumProcessorIds returning 3.

// CHECK-3IDS-LABEL: func.func @test_workgroup_scope
// CHECK-3IDS:       pcf.generic
// CHECK-3IDS-SAME:    scope(#iree_codegen.workgroup_scope)
// CHECK-3IDS:         execute(%{{.*}})[%[[ID0:.+]]: index, %[[ID1:.+]]: index, %[[ID2:.+]]: index, %[[C0:.+]]: index, %[[C1:.+]]: index, %[[C2:.+]]: index]
// CHECK-3IDS:         affine.linearize_index [%[[ID0]], %[[ID1]], %[[ID2]]] by (%[[C0]], %[[C1]], %[[C2]])
// CHECK-3IDS:         arith.muli %[[C0]], %[[C1]]
// CHECK-3IDS:         arith.muli
// CHECK-3IDS:         arith.ceildivui
// CHECK-3IDS:         scf.forall
// CHECK-3IDS:         pcf.return

// Test that workgroup scope with linearize=true creates 1 id/count pair via
// getNativeNumProcessorIds returning 1.

// CHECK-1ID-LABEL: func.func @test_workgroup_scope
// CHECK-1ID:       pcf.generic
// CHECK-1ID-SAME:    scope(#iree_codegen.workgroup_scope<linearize>)
// CHECK-1ID:         execute(%{{.*}})[%[[ID:.+]]: index, %[[COUNT:.+]]: index]
// CHECK-1ID-NOT:     affine.linearize_index
// CHECK-1ID:         arith.ceildivui
// CHECK-1ID:         scf.forall
// CHECK-1ID:         pcf.return
func.func @test_workgroup_scope(%init: tensor<64xf32>) -> tensor<64xf32> {
  %result = scf.forall (%i) in (64) shared_outs(%out = %init) -> tensor<64xf32> {
    %slice = tensor.extract_slice %out[%i] [1] [1] : tensor<64xf32> to tensor<1xf32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %slice into %out[%i] [1] [1] : tensor<1xf32> into tensor<64xf32>
    }
  } {mapping = [#iree_codegen.local_mapping<0>]}
  return %result : tensor<64xf32>
}
