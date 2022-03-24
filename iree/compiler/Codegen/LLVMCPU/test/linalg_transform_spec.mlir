// RUN: iree-opt %s 

pdl.pattern @pdl_matmul_target : benefit(1) {
  %args = operands
  %results = types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  rewrite %0 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %0 = match @pdl_matmul_target
  // %res contains the tiled op and the linalg_ext.tile op.
  %tiling_result:2 = tile_to_iree_linalg_ext_tile_op %0 {sizes = [2]}
  %2 = rewrite_iree_linalg_ext_tile_to_in_parallel %tiling_result#1
  // Bufferize happens at the IREE level on HAL operations, we cannot just 
  // call the linalg_transform.bufferize operation here.
  // Instead it happens automatically at the end of the linalg-transform-interp
  // pass.
}
