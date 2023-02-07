
// Preprocessing with generalized packing.
// ```
//   iree-opt tests/transform_dialect/cuda/matmul-packing.mlir --iree-transform-dialect-interpreter --transform-dialect-drop-schedule
// ```
//
// Dump the IR after dispatch region formation and focus our attention on the shape of the generic.
// ```
//   iree-opt tests/transform_dialect/cpu/contraction-packing.mlir --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | iree-opt  --iree-hal-target-backends=cuda --iree-abi-transformation-pipeline  --iree-flow-transformation-pipeline  --iree-stream-transformation-pipeline  --iree-hal-configuration-pipeline | grep -5 linalg\.generic
// ```
// 
// Compile e2e using the default cuda backend path: this is subject to the 2 temporary IREE caveats mentioned in #12076:
//   - the flow pipeline calls InterchangeGenericOps: this is not a deal breaker but changes the iteration order of the packed op a bit
//   - difficulties to lower either tensor.pack/unpack and linalg_ext.pack/unpack force us to lower to linalg.fill/linalg.transpose/etc
// ```
//   iree-opt tests/transform_dialect/cuda/matmul-packing.mlir --iree-transform-dialect-interpreter --transform-dialect-drop-schedule | iree-compile -  --iree-hal-target-backends=cuda
// ```

!a_tensor_t = tensor<1234x570xf32>
!b_tensor_t = tensor<570x890xf32>
!c_tensor_t = tensor<1234x890xf32>

func.func @matmul_nnn(%arg0: !a_tensor_t, %arg1: !b_tensor_t, %arg2: !c_tensor_t) -> !c_tensor_t {
  %0 = linalg.matmul
     ins(%arg0, %arg1: !a_tensor_t, !b_tensor_t)
    outs(%arg2: !c_tensor_t) -> !c_tensor_t
  return %0 : !c_tensor_t
}

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %matmul = transform.structured.match interface{LinalgOp} in %module_op
    : (!pdl.operation) -> (!pdl.operation)
  
  transform.structured.pack_greedily %matmul
      // This packs m x n to 16x16 and pads k to the next multiple of 16 (i.e. 576).
      // Generally setting one entry of gemm_packed_sizes to: 
      //   1. some value N within bounds: packs the dimension by N (i.e. increases
      //      the linalg op rank by 1)
      //   2. N > larger than the problem size: just pads the op 
      //      (e.g. the outer dim is 1 and is canonicalized; 
      //       i.e. does not change the rank of the linalg op).
      //   3. 0 skips the dimension from consideration.
      //
      // Packing/padding sizes are configurable, dimension order can be changed
      // by permuting [0, 1, 2].
      // The default order [0, 1, 2] for (m, n, k) produces (m,n,k, mm,nn,kk) 
      // unless some dimensions are degenerate (i.e. 0 or > problem size).
      //
      //                   mm  nn  kk                           mm nn kk
      gemm_packed_sizes = [16, 16, 576] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">


  // This is a rewrite of tensor.pack/tensor.unpack to linalg_ext.pack/linalg_ext.unpack
  // that IREE currently understands.
  // TODO: Remove once IREE adopts tensor.pack/unpack.
  // TODO: Unfortunately, this does not go through and hangs in iree-compile so
  // we need to fallback to other lowering to linalg.fill/linalg.transpose/etc below.
  //
  // %func = transform.structured.match ops{["func.func"]} in %module_op
  //   : (!pdl.operation) -> (!pdl.operation)
  // transform.iree.apply_patterns %func { rewrite_pack_ops }

  // IREE does not understand tensor.pack/unpack yet, so we have to lower them
  // explicitly to a form IREE understands.
  // This is only required to generate the PTX.
  //
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">) 
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)

  %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">) 
    -> (!transform.op<"tensor.empty">, 
        !transform.op<"linalg.transpose">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)
}
