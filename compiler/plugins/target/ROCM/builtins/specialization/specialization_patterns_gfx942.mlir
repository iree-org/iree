// RUN: iree-opt %s

// PDL pattern spec to annotate operations with specialization ranges.

pdl.pattern @f16_pingpong : benefit(1) {
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ]
  %elemtypes = pdl.attribute = [f16, f16, f32]
  %operands = pdl.operands
  %types = pdl.types
  %matmul = pdl.operation (%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
  pdl.apply_native_constraint "matchContraction"(
        %matmul, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  // Skip if the operation already has ranges.
  %attr_name = pdl.attribute = "iree_codegen.specialization_ranges"
  pdl.apply_native_constraint "hasAttr"(
        %matmul, %attr_name
        : !pdl.operation, !pdl.attribute) {isNegated = true}

  pdl.rewrite %matmul {
    %ranges = pdl.attribute = #util<int.assumption.multi_array[
        [<umin = 2048, udiv = 256>, <umin = 2048, udiv = 256>, <udiv = 64>], // Large pingpong
        [<umin = 1024, udiv = 128>, <umin = 1024, udiv = 128>, <udiv = 64>]  // Medium pingpong
      ]>
    pdl.apply_native_rewrite "annotateOperation"(
        %matmul, %attr_name, %ranges
        : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}

pdl.pattern @f8E4M3_pingpong : benefit(1) {
  %imaps = pdl.attribute = [
    affine_map<(d0, d1, d2) -> (d0, d2)>,
    affine_map<(d0, d1, d2) -> (d1, d2)>,
    affine_map<(d0, d1, d2) -> (d0, d1)>
  ]
  %elemtypes = pdl.attribute = [f8E4M3FNUZ, f8E4M3FNUZ, f32]
  %operands = pdl.operands
  %types = pdl.types
  %matmul = pdl.operation (%operands : !pdl.range<value>) -> (%types : !pdl.range<type>)
  pdl.apply_native_constraint "matchContraction"(
        %matmul, %elemtypes, %imaps
        : !pdl.operation, !pdl.attribute, !pdl.attribute)

  // Skip if the operation already has ranges.
  %attr_name = pdl.attribute = "iree_codegen.specialization_ranges"
  pdl.apply_native_constraint "hasAttr"(
        %matmul, %attr_name
        : !pdl.operation, !pdl.attribute) {isNegated = true}

  pdl.rewrite %matmul {
    %ranges = pdl.attribute = #util<int.assumption.multi_array[
        [<umin = 2048, udiv = 256>, <umin = 2048, udiv = 256>, <udiv = 128>], // Large pingpong
        [<umin = 1024, udiv = 128>, <umin = 1024, udiv = 128>, <udiv = 128>]  // Medium pingpong
      ]>
    pdl.apply_native_rewrite "annotateOperation"(
        %matmul, %attr_name, %ranges
        : !pdl.operation, !pdl.attribute, !pdl.attribute)
  }
}
