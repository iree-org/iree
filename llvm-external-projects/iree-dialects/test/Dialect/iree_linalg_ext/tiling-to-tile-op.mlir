// RUN: iree-dialects-opt %s -linalg-transform-interp --split-input-file | FileCheck %s

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1)[s0] -> (-d1 + s0, d0)>
module {
// CHECK-LABEL: matmul(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
  func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //      CHECK: %[[C10:.*]] = arith.constant 10 : index
  //      CHECK: iree_linalg_ext.tile %[[C10]] outs(%[[C]]: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[C_ITER:.*]]: tensor<?x?xf32>):
  //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tB:.*]] = tensor.extract_slice %[[B]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[RES:.*]] = linalg.matmul
  // CHECK-SAME:      ins(%[[tA]], %[[tB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[C_ITER]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  //      CHECK:   iree_linalg_ext.tile_yield %[[RES]] : tensor<?x?xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>)
                      outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
    return %0 : tensor<?x?xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_linalg_matmul : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "linalg.matmul"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    transform.structured.canonicalized_sequence %arg0 {
    ^bb1(%arg1: !pdl.operation):
      %0 = pdl_match @match_linalg_matmul in %arg1
      %1:2 = tile_to_iree_linalg_ext_tile_op %0 {sizes = [10]}
    }
  }
}

// -----

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (-d1 + 100, d0)>
module {
// CHECK-LABEL: matmul_static(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<100x200xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<200x300xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<100x300xf32>
  func @matmul_static(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>, %C: tensor<100x300xf32>) -> tensor<100x300xf32> {
    //      CHECK: %[[C10:.*]] = arith.constant 10 : index
    //      CHECK: iree_linalg_ext.tile %[[C10]] outs(%[[C]]: tensor<100x300xf32>) -> (tensor<100x300xf32>) {
    //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[C_ITER:.*]]: tensor<?x?xf32>):
    //      CHECK:   %[[M:.*]] = affine.min #[[$MAP]](%[[SZ]], %[[OFF]])
    //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<100x200xf32> to tensor<?x200xf32>
    //      CHECK:   %[[tC:.*]] = tensor.cast %[[C_ITER]] : tensor<?x?xf32> to tensor<?x300xf32>
    //      CHECK:   %[[RES:.*]] = linalg.matmul
    // CHECK-SAME:      ins(%[[tA]], %[[B]] : tensor<?x200xf32>, tensor<200x300xf32>)
    // CHECK-SAME:     outs(%[[tC]] : tensor<?x300xf32>) -> tensor<?x300xf32>
    //      CHECK:   %[[RES_DYN:.*]] = tensor.cast %[[RES]] : tensor<?x300xf32> to tensor<?x?xf32>
    //      CHECK:   iree_linalg_ext.tile_yield %[[RES_DYN]] : tensor<?x?xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<100x200xf32>, tensor<200x300xf32>) outs(%C : tensor<100x300xf32>) -> (tensor<100x300xf32>)
    return %0 : tensor<100x300xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_linalg_matmul : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "linalg.matmul"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    transform.structured.canonicalized_sequence %arg0 {
    ^bb1(%arg1: !pdl.operation):
      %0 = pdl_match @match_linalg_matmul in %arg1
      %1:2 = tile_to_iree_linalg_ext_tile_op %0 {sizes = [10]}
    }
  }
}
