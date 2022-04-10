// RUN: iree-opt %s 

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_matmul_target : benefit(1) {
    %args = operands
    %results = types
    %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_matmul_target in %arg1
    %tiling_1_result:2 = tile_to_iree_linalg_ext_tile_op %0 {sizes = [2]}
    %tiling_2_result:2 = tile_to_iree_linalg_ext_tile_op %tiling_1_result#0 {sizes = [0, 4]}
    %inp_2 = rewrite_iree_linalg_ext_tile_to_in_parallel %tiling_2_result#1
    %inp_1 = rewrite_iree_linalg_ext_tile_to_in_parallel %tiling_1_result#1
    transform.iree.bufferize
    transform.iree.inparallel_to_hal
  }
}
