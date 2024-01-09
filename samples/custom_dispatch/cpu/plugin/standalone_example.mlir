// This example demonstrates calling dynamically imported functions in the
// runtime. Alternatively the functions can be embedded into the compiled IREE
// programs for hermetic deployment (see custom_dispatch/cpu/embedded/).

// NOTE: this file is identical to system_example.mlir besides the lit config
// controlling the iree-run-module flag.
// TODO(benvanik): find a way to share the files (environment variables saying
// what types to run, etc).

// RUN: iree-compile --iree-hal-target-backends=llvm-cpu %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/plugin/standalone_plugin.sos \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 | \
// RUN: FileCheck %s --check-prefix=CHECK-STANDALONE

// CHECK-STANDALONE: EXEC @mixed_invocation
// CHECK-STANDALONE: 8xf32=12 12 12 12 12 12 12 12

module @example {

  // Executable containing exported shims and calls to external functions.
  // Each executable can contain multiple exported functions and variants for
  // different architectures or even devices. It's also possible to mix hand-
  // authored functions with code generated ones even for the same functions
  // such that code generation is used as a fallback when the hand-authored
  // kernels aren't supported at runtime.
  stream.executable private @executable {
    stream.executable.export public @simple_mul workgroups(%workload: index) -> (index, index, index) {
      // This host function is used to compute the XYZ workgroup count
      // dispatched at runtime. It can query the %device for capabilities
      // and limits (last-level cache sizes, etc). The other arguments are the
      // values passed in the dispatch operation (usually things like root
      // output op tensor dimensions and other abstract values).
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      stream.return %x, %c1, %c1 : index, index, index
    }

    builtin.module {
      // External function declaration using a user-chosen calling convention.
      func.func private @simple_mul_workgroup(
            %binding0: memref<f32>,
            %binding0_offset : index,
            %binding1: memref<f32>,
            %binding1_offset : index,
            %binding2: memref<f32>,
            %binding2_offset : index,
            %dim: index, %tid: index) attributes {
        // We can include some additional fields on the parameters struct as
        // needed. Here we request which processor is executing the call and
        // its data fields as defined by runtime/src/iree/schemas/cpu_data.h.
        hal.import.fields = ["processor_id", "processor_data"],
        llvm.bareptr = true
      }

      // IREE exported function using stream bindings and operands.
      // Compiler passes will be able to optimize across this interface and
      // deduplicate bindings/operands, convert/pack operands, and inline
      // constants operands.
      func.func @simple_mul(
          %binding0: !stream.binding,
          %binding1: !stream.binding,
          %binding2: !stream.binding,
          %dim: index) {
        %c0 = arith.constant 0 : index

        // This function is invoked once per workgroup so determine where this
        // particular workgroup is in the grid. In this example we use a
        // workgroup size of 64x1x1 (which is exceedingly small for CPUs but
        // useful for demonstration).
        %workgroup_id_x = stream.dispatch.workgroup.id[0] : index
        %tid = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]

        // Bindings are accessed by reference.
        %memref0 = stream.binding.subspan %binding0[%c0] : !stream.binding -> memref<?xf32>{%dim}
        %memref1 = stream.binding.subspan %binding1[%c0] : !stream.binding -> memref<?xf32>{%dim}
        %memref2 = stream.binding.subspan %binding2[%c0] : !stream.binding -> memref<?xf32>{%dim}

        // The default `memref` lowering contains additional fields that might not be
        // always required. In this example, we only need the base and offset of the
        // `memref`s. So extract the base and offset from the memrefs.
        %base0, %offset0, %size0, %stride0 = memref.extract_strided_metadata %memref0
            : memref<?xf32> -> memref<f32>, index, index, index
        %base1, %offset1, %size1, %stride1 = memref.extract_strided_metadata %memref1
            : memref<?xf32> -> memref<f32>, index, index, index
        %base2, %offset2, %size2, %stride2 = memref.extract_strided_metadata %memref2
            : memref<?xf32> -> memref<f32>, index, index, index

        // Call the externally defined C function with an (almost) plain C
        // calling convention (see above for details about the mess memrefs
        // turn into). This will be fetched at runtime from the plugin binary.
        func.call @simple_mul_workgroup(
            %base0, %offset0, %base1, %offset1, %base2, %offset2, %dim, %workgroup_id_x)
            : (memref<f32>, index, memref<f32>, index, memref<f32>, index, index, index) -> ()

        // NOTE: this is code generated as normal - other MLIR ops can be used
        // here for looping/control flow, vector operations, linalg, etc.
        // This simple sample is just calling out to the external function but
        // microkernels fused with other code are possible.

        return
      }
    }
  }

  // Function demonstrating executable plugins and mixing plugins and codegen.
  // Invoke with:
  //  --device=local-sync
  //  --executable_plugin=standalone_plugin.sos
  //  --function=mixed_invocation
  //  --input=8xf32=2
  //  --input=8xf32=4
  func.func @mixed_invocation(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    // The only externally available metadata in the dispatch are the values
    // passed in as operands. Here we pass in the dynamic dimension.
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>

    // Dispatch a basic `ret = lhs * rhs` using an external function.
    // This form (@executable::@export) allows for automatic variant selection
    // when multi-targeting.
    %0 = flow.dispatch @executable::@simple_mul[%dim](%arg0, %arg1, %dim) : (tensor<?xf32>{%dim}, tensor<?xf32>{%dim}, index) -> tensor<?xf32>{%dim}

    // Code gen some other ops - these will interleave with hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?xf32>

    return %1 : tensor<?xf32>
  }

}  // module
