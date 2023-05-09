// This example demonstrates calling dynamically imported functions in the
// runtime through the use of ukernels. This is calling the same function
// as `standalone_example.mlir`, but using the `iree_codegen.ukernel.generic`.
// This is an example of how ukernels can be called from code generated
// by IREE.

// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/plugin/standalone_plugin.sos \
// RUN:     --function=ukernel_example \
// RUN:     --module=- \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 | \
// RUN: FileCheck %s --check-prefix=CHECK-STANDALONE

// CHECK-STANDALONE: EXEC @ukernel_example
// CHECK-STANDALONE: 8xf32=8 8 8 8 8 8 8 8

func.func @ukernel_example(%arg0 : tensor<?xf32>, %arg1 : tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %dest = tensor.empty(%d0) : tensor<?xf32>
  // Create a dispatch that operands on `2` threads. Set the `worload` of
  // `flow.dispatch.region` to capture the values needed to specify the number
  // of threads to use in the `count` region.
  %0 = flow.dispatch.region[%c2] -> (tensor<?xf32>{%d0}) {
    %id = flow.dispatch.workgroup.id[0] : index
    %count = flow.dispatch.workgroup.count[0] : index
    
    // Each thread has to perform a slice of the computation.
    %tilesize = affine.apply affine_map<()[s0, s1] -> (s0 ceildiv s1)>()[%d0, %count]
    
    // Compute the offset and size of the slice
    %offset = affine.apply affine_map<()[s0, s1] -> (s0 * s1)>()[%id, %tilesize]
    %size = affine.min affine_map<(d0)[s0, s1] -> (s1 - d0, s0)>(%offset)[%tilesize, %d0]
    
    // Extract slices of the inputs and outputs.
    %1 = tensor.extract_slice %arg0[%offset] [%size] [1] : tensor<?xf32> to tensor<?xf32>
    %2 = tensor.extract_slice %arg1[%offset] [%size] [1] : tensor<?xf32> to tensor<?xf32>
    %3 = tensor.extract_slice %dest[%offset] [%size] [1] : tensor<?xf32> to tensor<?xf32>

    // Invoke the ukernel.
    %4 = iree_codegen.ukernel.generic "simple_mul_workgroup"
      ins(%1, %2 : tensor<?xf32>, tensor<?xf32>)
      outs(%3 : tensor<?xf32>)
      (%size, %id : index, index) 
      // We can include some additional fields on the parameters struct as
      // needed. Here we request which processor is executing the call and
      // its data fields as defined by runtime/src/iree/schemas/cpu_data.h.
      fn_def_attrs {hal.import.fields = ["processor_id", "processor_data"]}
      // Set the operation to not incorporate any strides. The implementation
      // expects no stride arguments.
      strided_outer_dims(0) -> tensor<?xf32>

    // Insert the result back into the result at the right position.
    %5 = tensor.insert_slice %4 into %dest[%offset] [%size] [1] : tensor<?xf32> into tensor<?xf32>
    flow.return %5 : tensor<?xf32>
  } count(%b0 : index) -> (index, index, index) {
    // Specify the number of threads to use. `%b0` represents
    // the values captured as workload (within `[` `]` in the `flow.dispatch.region` above)
    // Use that to derive the number of threads to use along `x`, `y` and `z`.
    flow.return %b0, %c1, %c1 : index, index, index
  }
  return %0 : tensor<?xf32>
}
