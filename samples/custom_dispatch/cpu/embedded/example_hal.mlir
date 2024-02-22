// RUN: iree-compile %s \
// RUN:     --iree-hal-executable-object-search-path=$IREE_BINARY_DIR | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 | \
// RUN: FileCheck %s

// This example demonstrates authoring and dispatching retargetable executables
// from the IREE `hal` dialect layer. This allows for target-specific code to
// be written - including unique calls for each target - as the executable
// variants are manually specified. The example_stream.mlir example shows how
// where possible the executable variant generation can be left to the compiler.
//
// Enabling this at the HAL layer allows for codegen backends translating
// executable variants to make local decisions about which external calls to
// make and where the objects come from to provide those functions. Since
// objects can be embedded in the IR it's possible for the backends to even
// generate them on-demand for embedding (such as precompiling/JITing).

// The configuration used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced and multiple variants with differing formats
// and compilation options (architectures, etc) can be embedded for runtime
// selection. By fully specifying the targets here we can target multiple
// architectures and it's always possible to embed these instead of using the
// coarse command line compiler flags that only set single targets.
//
// To avoid too much boilerplate this example only shows a single target. See
// example_stream.mlir for an example with multi-targeting as there's less
// boilerplate required at that level.
#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf"
}>

// The target devices that the program will run on.
// These can come from compiler flags and multiple targets can be supported
// It's possible, for example, to support targeting multiple devices in the same
// compiled binary (CPU + Vulkan, etc).
#cpu_target = #hal.device.target<"llvm-cpu", {
  executable_targets = [
    #x86_64_target
  ]
}>

module @example attributes {hal.device.targets = [#cpu_target]} {

  // Executable containing exported shims and calls to external functions.
  // Each executable can contain multiple exported functions and variants for
  // different architectures or even devices. It's also possible to mix hand-
  // authored functions with code generated ones even for the same functions
  // such that code generation is used as a fallback when the hand-authored
  // kernels aren't supported at runtime.
  hal.executable private @executable {

    // Variant linking in an x86-64 object file containing external functions.
    hal.executable.variant public @x86_64 target(#x86_64_target) objects([
      // Object files linked into the executable.
      // These object files are linked into the dynamic library and must meet
      // the requirements for embedded ELF linkage (no TLS, no globals, no
      // syscalls, no libc, etc).
      #hal.executable.object<{
        // Referencing a file path on disk but could also have the data
        // embedded in order to make the MLIR file hermetic/portable across
        // compilation pipelines. In the future we'll likely use MLIR's
        // external resource functionality for this. By allowing for the
        // objects to be embedded we can support JIT scenarios where some
        // layer higher or lower may be emitting the objects to link in as
        // part of the overall compilation.
        path = "samples/custom_dispatch/cpu/embedded/functions_x86_64.o"
      }>
    ]) {

      // TODO(benvanik): demonstrate hal.executable.constant.block for
      // specialization via host logic and hal.executable.constant.load for
      // referencing them in the shims.

      // Exported shim function calling the C `simple_mul_workgroup` function.
      // The ordinal must be assigned by the user and unique for the executable.
      // The layout defines the required bindings and push constants and can be
      // thought of as the function signature.
      hal.executable.export public @simple_mul ordinal(0)
          layout(#hal.pipeline.layout<push_constants = 1, sets = [
            <0, bindings = [
                <0, storage_buffer, ReadOnly>,
                <1, storage_buffer, ReadOnly>,
                <2, storage_buffer>
            ]>
          ]>) attributes {
            // Bindings are automatically inferred when possible as part of the
            // ABI but can be overridden if the user wants to use features such
            // as sparse bindings or multiple descriptor sets. To do so the
            // `hal.interface.bindings` attribute can be added to a dispatch op
            // as follows mapping tensor operands/results to the pipeline layout
            // sets/bindings:
            hal.interface.bindings = [
              #hal.interface.binding<0, 0>,
              #hal.interface.binding<0, 1>,
              #hal.interface.binding<0, 2>
            ]
          } {
      ^bb0(%device: !hal.device, %workload: index):
        // This host function is used to compute the XYZ workgroup count
        // dispatched at runtime. It can query the %device for capabilities
        // and limits (last-level cache sizes, etc). The other arguments are the
        // values passed in the dispatch operation (usually things like root
        // output op tensor dimensions and other abstract values).
        %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
        %c1 = arith.constant 1 : index
        hal.return %x, %c1, %c1 : index, index, index
      }

      // Similar to the above but in-place by using a read/write binding.
      hal.executable.export public @simple_mul_inplace ordinal(1)
          layout(#hal.pipeline.layout<push_constants = 1, sets = [
            <0, bindings = [
                <0, storage_buffer, ReadOnly>,
                <1, storage_buffer>
            ]>
          ]>) {
      ^bb0(%device: !hal.device, %workload: index):
        %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
        %c1 = arith.constant 1 : index
        hal.return %x, %c1, %c1 : index, index, index
      }

      // On the CPU side we use shims here to marshal across the ABI. This
      // allows us to hide the implementation details of how the runtime calls
      // into functions and call out to C functions that don't need to link
      // against the runtime. We could probably come up with ways of automating
      // this but that's mostly left as an exercise to the frontends that may be
      // producing this IR for input to the IREE compiler as each may have its
      // own quirks.
      builtin.module {
        // External function declaration using a user-chosen calling convention.
        // NOTE: MLIR->LLVM conversion expands each memref to a tuple and
        // there's currently no way to change that behavior.
        // Each memref becomes:
        // (%base_ptr: !llvm.ptr<f32>, %aligned_ptr: !llvm.ptr<f32>,
        //  %offset: i64, %size: i64, %stride: i64)
        // That results in the following llvm.func:
        // (!llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64,  // binding0
        //  !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64,  // binding1
        //  !llvm.ptr<f32>, !llvm.ptr<f32>, i64, i64, i64,  // binding2
        //  i64,                                            // dim
        //  i64)                                            // tid
        // And required external C function:
        // (float*, float*, size_t, size_t, size_t,
        //  float*, float*, size_t, size_t, size_t,
        //  float*, float*, size_t, size_t, size_t,
        //  size_t,
        //  size_t)
        // This is not a good state to be in as we can't then map to external
        // functions that have signatures we don't want to change. Please file
        // upstream MLIR bugs about this behavior and the ability to just pass
        // bare pointers if you care!
        //
        // NOTE: index will convert to i32 when targeting an ABI with 32-bit
        // pointers and i64 otherwise. Use size_t on the C side to allow the
        // same source code to work when compiled in either mode.
        func.func private @simple_mul_workgroup(%binding0: memref<?xf32>, %binding1: memref<?xf32>, %binding2: memref<?xf32>, %dim: index, %tid: index) attributes {
          // Ensures that we try to statically link this external function and
          // pull it in from the object file.
          hal.import.static
        }

        // IREE exported function using a HAL interface.
        // At this layer of the stack all operands have been converted into
        // constants and bindings have been specified.
        func.func @simple_mul() {
          %c0 = arith.constant 0 : index

          // Push constants representing primitive operands can be loaded here.
          %dim_i32 = hal.interface.constant.load[0] : i32
          %dim = arith.index_castui %dim_i32 : i32 to index

          // This function is invoked once per workgroup so determine where this
          // particular workgroup is in the grid. In this example we use a
          // workgroup size of 64x1x1 (which is exceedingly small for CPUs but
          // useful for demonstration).
          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %tid = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]

          // Bindings are accessed by reference.
          %binding0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}
          %binding1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}
          %binding2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}

          // Call the externally defined C function with an (almost) plain C
          // calling convention (see above for details about the mess memrefs
          // turn into).
          //
          // TODO: there are ways of accessing CPU information here such as
          // active architecture and feature bits but it is not yet exposed to
          // the HAL level.
          func.call @simple_mul_workgroup(%binding0, %binding1, %binding2, %dim, %tid) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()

          // NOTE: this is code generated as normal - other MLIR ops can be used
          // here for looping/control flow, vector operations, linalg, etc.
          // This simple sample is just calling out to the external function but
          // microkernels fused with other code are possible.

          return
        }

        func.func private @simple_mul_inplace_workgroup(%binding0: memref<?xf32>, %binding1: memref<?xf32>, %dim: index, %tid: index) attributes {
          hal.import.static
        }
        func.func @simple_mul_inplace() {
          %c0 = arith.constant 0 : index

          %dim_i32 = hal.interface.constant.load[0] : i32
          %dim = arith.index_castui %dim_i32 : i32 to index

          %workgroup_id_x = hal.interface.workgroup.id[0] : index
          %tid = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]

          // Same as above but note that we're treating %binding1 as read/write.
          %binding0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}
          %binding1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) : memref<?xf32>{%dim}

          func.call @simple_mul_inplace_workgroup(%binding0, %binding1, %dim, %tid) : (memref<?xf32>, memref<?xf32>, index, index) -> ()

          return
        }
      }

    }  // hal.executable.variant

  }  // hal.executable

  // Function demonstrating a few hand-authored dispatches mixed with codegen.
  // Invoke with:
  //  --device=local-sync
  //  --function=mixed_invocation
  //  --input=8xf32=2
  //  --input=8xf32=4
  // CHECK-LABEL: EXEC @mixed_invocation
  func.func @mixed_invocation(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    // The only externally available metadata in the dispatch are the values
    // passed in as operands. Here we pass in the dynamic dimension.
    //
    // HACK: for hand-authored kernels all primitive values passed in need to
    // be i32 or a bit-castable type. This is because ABI packing of other types
    // happens inside of the PackDispatchOperandsPass that is currently not
    // usable with external functions as it changes the ABI. In the future we
    // can better define the ABI such that it's possible to match the compiler
    // expectations around padding/alignment. For now users must do the packing
    // themselves (splitting i64 into i32+i32, etc).
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %dim_i32 = arith.index_cast %dim : index to i32

    // Dispatch a basic `ret = lhs * rhs` using an external function.
    %0 = flow.dispatch @executable::@x86_64::@simple_mul[%dim](%dim_i32, %arg0, %arg1) {
      // HACK: keep the executable live through DCE. Only required when
      // using the automatic variant selection.
      // TODO(benvanik): automatically add this when required.
      hal.executable.ref = [@executable]
    } : (i32, tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}

    // Code gen some other ops - these will interleave with the hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?xf32>

    // Dispatch an in-place `rhs *= lhs` using an external function.
    // This form (@executable::@variant::@export) specifically chooses a variant
    // instead of relying on automatic selection. This can be used by frontends
    // to allow user-controlled overrides of the dispatches, custom selection
    // logic based on runtime parameters, etc. In general, though, the above
    // automatic selection should be used.
    //
    // Note that we don't declare the hal.interface.bindings and let them be
    // inferred - this only works when either specifying the variant that has
    // a pipeline layout defined or all variants have the same pipeline layouts.
    %2 = flow.dispatch @executable::@x86_64::@simple_mul_inplace[%dim](%dim_i32, %0, %1) : (i32, tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> %1{%dim}

    // CHECK: 8xf32=96 96 96 96 96 96 96 96
    return %2 : tensor<?xf32>
  }

}  // module
