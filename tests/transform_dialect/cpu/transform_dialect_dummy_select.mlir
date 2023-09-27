#transform = #iree_codegen.translation_info<TransformDialectCodegen>
#blank_config = #iree_codegen.lowering_config<tile_sizes = []>
#translation4 = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec=@print_selected4>
#matvec4_config = #iree_codegen.compilation_info<lowering_config = #blank_config, translation_info = #translation4, workgroup_size = []>
#translation6 = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec=@print_selected6>
#matvec6_config = #iree_codegen.compilation_info<lowering_config = #blank_config, translation_info = #translation6, workgroup_size = []>

module attributes { transform.with_named_sequence } {


  //===------------------------------------------------------===
  // Matvec
  //===------------------------------------------------------===
  transform.named_sequence @match_matvec4(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c4 = transform.param.constant 4 : i64 -> !transform.param<i64>

      %dim = transform.match.structured.dim %arg1[0] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %c4, %dim : !transform.param<i64>
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    %config = transform.param.constant #matvec4_config -> !transform.any_param
    transform.yield %0, %config : !transform.any_op, !transform.any_param
  }

  transform.named_sequence @match_matvec6(%arg0: !transform.any_op {transform.readonly}) -> (!transform.any_op, !transform.any_param) {
    %0 = transform.match.structured failures(propagate) %arg0 : (!transform.any_op) -> !transform.any_op {
    ^bb1(%arg1: !transform.any_op):
      %c6 = transform.param.constant 6 : i64 -> !transform.param<i64>

      %dim = transform.match.structured.dim %arg1[0] : (!transform.any_op) -> !transform.param<i64>
      transform.match.param.cmpi eq %c6, %dim : !transform.param<i64>
      transform.match.structured.yield %arg1 : !transform.any_op
    }

    %config = transform.param.constant #matvec6_config -> !transform.any_param
    transform.yield %0, %config : !transform.any_op, !transform.any_param
  }

  //===------------------------------------------------------===
  // Annotation and Application
  //===------------------------------------------------------===

  transform.named_sequence @annotate_op(%target: !transform.any_op {transform.readonly}, %config: !transform.any_param {transform.readonly}) {
    transform.annotate %target "compilation_info" = %config : !transform.any_op, !transform.any_param
    transform.yield
  }


  transform.sequence failures(propagate) {
  ^bb0(%dispatch: !transform.any_op):
    %dispatch_func = transform.structured.match ops{["func.func"]} in %dispatch : (!transform.any_op) -> !transform.any_op
    transform.foreach_match in %dispatch_func
        @match_matvec4 -> @annotate_op,
        @match_matvec6 -> @annotate_op
      : (!transform.any_op) -> (!transform.any_op)
  }
}
