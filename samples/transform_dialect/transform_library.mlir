module attributes { transform.with_named_sequence } {
  // Example of a custom matmul strategy. The target matmul is annotated with
  // the name of this strategy down below before strategy selection, overriding
  // default IREE codegen.
  transform.named_sequence @custom_transform_strategy(
      %variant_op: !transform.any_op {transform.consumed}) {
    // Step 1. Re-match the matmul
    // ===========================================================================
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Step 2. Tile to grid
    // ===========================================================================
    %grid_reduction, %forall_grid =
    transform.structured.tile_using_forall %matmul tile_sizes [16, 16] ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

    // Step 3. Vectorize
    // ===========================================================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    } : !transform.any_op
    %func_1 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> !transform.any_op

    // Step 4. Bufferize
    // ===========================================================================
    transform.apply_patterns to %func_1 {
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_patterns to %func_1 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_cse %func_1 : !transform.any_op
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
    transform.apply_patterns to %func_1 {
      transform.apply_patterns.linalg.erase_unnecessary_inputs
    } : !transform.any_op
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7
        workgroup_dims = [4, 8, 1] : (!transform.any_op) -> ()

    // Step 7. Do layout analysis and lower to mma
    // ===========================================================================
    %func_10 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    %func_11 = transform.iree.layout_analysis_and_distribution %func_10 : (!transform.any_op) -> (!transform.any_op)
    transform.print {name = "Ran custom_transform_strategy"}
    transform.yield
  }

  // Send it down a custom transform dialect pipeline.
  transform.named_sequence @custom_matmul(%matmul: !transform.any_op {transform.readonly}) {
    %variant_op = transform.get_parent_op %matmul {op_name = "hal.executable.variant"} : (!transform.any_op) -> !transform.any_op
    %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %subgroup_reduce = transform.param.constant #iree_codegen.translation_info<TransformDialectCodegen
                                                                               codegen_spec = @custom_transform_strategy> -> !transform.any_param
    transform.annotate %exports "translation_info" = %subgroup_reduce : !transform.any_op, !transform.any_param
    transform.print {name = "Setting matmul strategy to custom_transform_strategy"}
    transform.yield
  }

  // Send it down subgroup reduce with a custom tiling configuration.
  transform.named_sequence @use_base_vectorize(%reduce: !transform.any_op {transform.readonly}) {
    %variant_op = transform.get_parent_op %reduce {op_name = "hal.executable.variant"} : (!transform.any_op) -> !transform.any_op
    %lowering_config = transform.param.constant #iree_codegen.lowering_config<tile_sizes = [[8, 0], [1, 0], [0, 0, 4]]> -> !transform.any_param
    transform.annotate %reduce "lowering_config" = %lowering_config : !transform.any_op, !transform.any_param
    %exports = transform.structured.match ops{["hal.executable.export"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %subgroup_reduce = transform.param.constant #iree_codegen.translation_info<SPIRVBaseVectorize> -> !transform.any_param
    %workgroup_size = transform.param.constant [16 : index, 1 : index, 1 : index] -> !transform.any_param
    transform.annotate %exports "translation_info" = %subgroup_reduce : !transform.any_op, !transform.any_param
    transform.annotate %exports "workgroup_size" = %workgroup_size : !transform.any_op, !transform.any_param
    transform.print {name = "Setting reduce strategy to base vectorize"}
    transform.yield
  }

  //===------------------------------------------------------===
  // Matchers
  //===------------------------------------------------------===
  transform.named_sequence @match_matmul(%matmul: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %matmul ["linalg.matmul"] : !transform.any_op
    transform.yield %matmul : !transform.any_op
  }

  transform.named_sequence @match_reduce(%reduce: !transform.any_op {transform.readonly}) -> (!transform.any_op) {
    transform.match.operation_name %reduce ["linalg.generic"] : !transform.any_op
    %matched = transform.match.structured failures(propagate) %reduce : (!transform.any_op) -> (!transform.any_op) {
    ^bb1(%arg1: !transform.any_op):
      %c2 = transform.param.constant 2 : i64 -> !transform.param<i64>
      %rank = transform.match.structured.rank %arg1 : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %rank, %c2 : !transform.param<i64>
  
      transform.match.structured.dim %arg1[-1] {reduction} : !transform.any_op
      transform.match.structured.yield %arg1 : !transform.any_op
    }
    transform.yield %matched : !transform.any_op
  }

  transform.named_sequence @transform_main(%variant_op: !transform.any_op {transform.consumed}) {
    transform.foreach_match in %variant_op
        @match_matmul -> @custom_matmul,
        @match_reduce -> @use_base_vectorize
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
