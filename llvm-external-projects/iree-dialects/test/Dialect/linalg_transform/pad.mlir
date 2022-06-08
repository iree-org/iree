// RUN: iree-dialects-opt --transform-dialect-interpreter %s | FileCheck %s

#map = affine_map<()[s0] -> (-s0 + 12, 5)>

// CHECK-LABEL: func.func @pad_unary
func.func @pad_unary(%arg0: tensor<24x12xf32>,
                %arg1: tensor<24x12xf32>) -> tensor<24x12xf32> {
  //      CHECK:   %[[C0:.*]] = arith.constant 0 : index
  %c0 = arith.constant 0 : index
  %c12 = arith.constant 12 : index
  %c5 = arith.constant 5 : index

  //      CHECK:   scf.for
  //      CHECK:     tensor.pad
  //      CHECK:     linalg.generic
  //      CHECK:   scf.for
  %0 = scf.for %arg3 = %c0 to %c12 step %c5 iter_args(%arg2 = %arg1) -> (tensor<24x12xf32>) {
    %ts = affine.min #map()[%arg3]
    %1 = tensor.extract_slice %arg0[0, %arg3] [24, %ts] [1, 1] : tensor<24x12xf32> to tensor<24x?xf32>
    %2 = tensor.extract_slice %arg2[0, %arg3] [24, %ts] [1, 1] : tensor<24x12xf32> to tensor<24x?xf32>

    //      CHECK:     linalg.generic
    //      CHECK:     %[[WIDTH:.*]] = affine.apply
    //      CHECK:     tensor.pad
    // CHECK-SAME:     high[%[[C0]], %[[WIDTH]]]
    //      CHECK:     linalg.elemwise_unary
    %3 = linalg.elemwise_unary ins(%1 : tensor<24x?xf32>)
                              outs(%2: tensor<24x?xf32>) -> tensor<24x?xf32>
    %4 = tensor.insert_slice %3 into %arg2[0, %arg3] [24, %ts] [1, 1] : tensor<24x?xf32> into tensor<24x12xf32>
    scf.yield %4 : tensor<24x12xf32>
  }
  return %0 : tensor<24x12xf32>
}

transform.with_pdl_patterns {
^bb0(%arg0: !pdl.operation):
  pdl.pattern @pdl_target : benefit(1) {
    %args = operands
    %results = types
    %0 = pdl.operation "linalg.elemwise_unary"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
    %1 = pdl.attribute = @pad_unary
    apply_native_constraint "nestedInFunc"(%0, %1 : !pdl.operation, !pdl.attribute)
    // TODO: we don't want this, but it is the required terminator for pdl.pattern
    rewrite %0 with "transform.dialect"
  }

  transform.structured.canonicalized_sequence %arg0 {
  ^bb1(%arg1: !pdl.operation):
    %0 = pdl_match @pdl_target in %arg1
    %1 = transform.structured.pad %0 {padding_values=[0.0 : f32, 0.0 : f32], padding_dimensions=[1], pack_paddings=[1, 1], hoist_paddings=[1, 0], transpose_paddings=[[1, 0], [0, 1]]}
  }
}
