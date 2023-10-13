func.func @ukernel_example() {
  %s0 = arith.constant dense<[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]> : tensor<10xf32>
  %s1 = arith.constant dense<[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]> : tensor<10xf32>
  %arg0 = util.optimization_barrier %s0 : tensor<10xf32>
  %arg1 = util.optimization_barrier %s1 : tensor<10xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %dest = tensor.empty() : tensor<10xf32>
  // Create a dispatch that uses a workgroup size of 4.
  // `flow.dispatch.region` to capture the values needed to specify the number
  // of threads to use in the `count` region.
  %0 = flow.dispatch.region[] -> (tensor<10xf32>)
      // Use `#iree_codegen.export_config` to specify control over the execution. Currently
      // the workgroup size/block size.
      // Note: The name "iree_codegen.export_config" is also important for it to be
      // propagated through the compiler.
      attributes {iree_codegen.export_config = #iree_codegen.export_config<workgroup_size = [4]>} {
    %id = flow.dispatch.workgroup.id[0] : index
    %count = flow.dispatch.workgroup.count[0] : index

    // Compute the offset and size of the slice
    %offset = affine.apply affine_map<()[s0] -> (s0 * 4)>()[%id]
    %size = affine.min affine_map<(d0)[] -> (4, 10 - d0)>(%offset)[]

    // Extract slices of the inputs and outputs.
    %1 = tensor.extract_slice %arg0[%offset] [%size] [1] : tensor<10xf32> to tensor<?xf32>
    %2 = tensor.extract_slice %arg1[%offset] [%size] [1] : tensor<10xf32> to tensor<?xf32>
    %3 = tensor.extract_slice %dest[%offset] [%size] [1] : tensor<10xf32> to tensor<?xf32>

    // Invoke the ukernel.
    %4 = iree_codegen.ukernel.generic "simple_mul_workgroup"
      ins(%1, %2 : tensor<?xf32>, tensor<?xf32>)
      outs(%3 : tensor<?xf32>)
      (%size : index)
      // Set the operation to not incorporate any strides. The implementation
      // expects no stride arguments.
      strided_outer_dims(0) -> tensor<?xf32>

    // Insert the result back into the result at the right position.
    %5 = tensor.insert_slice %4 into %dest[%offset] [%size] [1] : tensor<?xf32> into tensor<10xf32>
    flow.return %5 : tensor<10xf32>
  } count() -> (index, index, index) {
    flow.return %c3, %c1, %c1 : index, index, index
  }
  check.expect_almost_eq_const(%0, dense<[0.0, 2.0, 8.0, 18.0, 32.0, 50.0, 72.0, 98.0, 128.0, 162.0]> : tensor<10xf32>) : tensor<10xf32>
  return
}
