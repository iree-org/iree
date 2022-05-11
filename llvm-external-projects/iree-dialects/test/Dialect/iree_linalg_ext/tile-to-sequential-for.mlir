// RUN: iree-dialects-opt %s --linalg-transform-interp --split-input-file | FileCheck %s

// CHECK-DAG: #[[$SUB_MAP:.*]] = affine_map<(d0)[s0, s1] -> (-d0 + s1, s0)>
// CHECK-DAG: #[[$ID1_MAP:.*]] = affine_map<(d0) -> (d0)>

module {
  // CHECK-LABEL: func @static_tile
  //  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
  //  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
  func @static_tile(%chunk_size: index, %in: tensor<?xf32>, %out: tensor<?xf32>, %out2: tensor<?xf32>) -> (tensor<?xf32>) {
    %c0 = arith.constant 0: index

    // CHECK: %[[M:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?xf32>
    // CHECK: scf.for %[[IV:.*]] = {{.*}} iter_args(%[[OUT:.*]] = %{{.*}}, %[[OUT2:.*]] = %{{.*}}) -> (tensor<?xf32>, tensor<?xf32>) {
    %0:2 = iree_linalg_ext.tile %chunk_size outs(%out: tensor<?xf32>, %out2: tensor<?xf32>)
        -> (tensor<?xf32>, tensor<?xf32>) {

    // CHECK:    %[[SIZE:.*]] = affine.min #[[$SUB_MAP]](%[[IV]])[%[[CHUNK_SIZE]], %[[M]]]
    // CHECK:    %[[O:.*]] = tensor.extract_slice %[[OUT]][%[[IV]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> to tensor<?xf32>
    // CHECK:    %[[O2:.*]] = tensor.extract_slice %[[OUT2]][%[[IV]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> to tensor<?xf32>

      // TODO: one offset and one size per tensor?
      // If not necessary in the dense strided-array world, what about the rest?
      ^bb0(%offset: index, %size: index, %st1: tensor<?xf32>, %st2: tensor<?xf32>):

        // TODO: atm this is just 1-1: out-chunk-size -> in-size.
    // CHECK:    %[[I:.*]] = tensor.extract_slice %[[IN]][%[[IV]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> to tensor<?xf32>
        %1 = tensor.extract_slice %in[%offset][%size][1] : tensor<?xf32> to tensor<?xf32>

    // CHECK:    %[[R:.*]] = linalg.generic {{.*}} ins(%[[I]] : tensor<?xf32>) outs(%[[O]] : tensor<?xf32>)
        %3 = linalg.generic {
            indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
            iterator_types = ["parallel"]}
          ins(%1: tensor<?xf32>) outs(%st1: tensor<?xf32>) {
          ^bb0(%a: f32, %b:f32):  // no predecessors
            %f42 = arith.constant 42.0: f32
            %tmp = arith.mulf %a, %f42: f32
            linalg.yield %tmp: f32
        } -> tensor<?xf32>

    // CHECK:    %[[RES:.*]] = tensor.insert_slice %[[R]] into %[[OUT]][%[[IV]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> into tensor<?xf32>
    // CHECK:    %[[RES2:.*]] = tensor.insert_slice %[[O2]] into %[[OUT2]][%[[IV]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> into tensor<?xf32>
    // CHECK:    scf.yield %[[RES]], %[[RES2]] : tensor<?xf32>, tensor<?xf32>
        iree_linalg_ext.tile_yield %3, %st2: tensor<?xf32>, tensor<?xf32> // assumes dim is 0 and stacks
    }
    return %0#0: tensor<?xf32>
  }

  transform.with_pdl_patterns {
  ^bb0(%arg0: !pdl.operation):
    pdl.pattern @match_iree_linalg_ext_tile : benefit(1) {
      %0 = operands
      %1 = types
      %2 = operation "iree_linalg_ext.tile"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
      rewrite %2 with "transform.dialect"
    }
    transform.structured.canonicalized_sequence %arg0 {
    ^bb0(%arg1: !pdl.operation):
      %0 = pdl_match @match_iree_linalg_ext_tile in %arg1
      %1 = rewrite_iree_linalg_ext_tile_to_scf_for %0
    }
  }
}
