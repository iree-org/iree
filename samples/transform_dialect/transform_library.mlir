module attributes { transform.with_named_sequence } {
  // Print and send it down normal IREE codegen.
  transform.named_sequence @custom_matmul(%matmul: !transform.any_op {transform.consumed}) {  
    %1 = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op
    transform.print {name = "Setting matmul strategy to default"}
    transform.yield
  }

  // Send it down subgroup reduce.
  transform.named_sequence @use_subgroup_reduce(%reduce: !transform.any_op {transform.readonly}) {  
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
        @match_reduce -> @use_subgroup_reduce
      : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
