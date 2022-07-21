transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    // Decide fusion groups.
    %roots = transform.iree.decide_fusion_roots %arg1

    // Tile each root op into an scf.foreach_thread op.
    transform.iree.foreach %roots {
    ^bb2(%root: !pdl.operation):
      %producers = transform.iree.get_fusable_producers %root
      %foreach_op, %tiled_op = transform.tile_to_foreach_thread_op %root {num_threads = [13, 33]}
      transform.iree.foreach %producers {
      ^bb3(%producer: !pdl.operation):
        transform.fuse_into_containing_op %producer into %foreach_op
        transform.yield
      }

      transform.print %foreach_op { name = "AFTER!!!" }

      // Rewrite scf.foreach_thread op to Flow dialect ops.
      %dispatch_op = transform.iree.foreach_thread_to_flow %foreach_op
      transform.yield
    }

    //transform.print %arg1 { name = "AFTER!!!" }
  }
}
