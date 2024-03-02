// RUN: iree-compile %s \
// RUN:     --iree-hal-executable-object-search-path=$IREE_BINARY_DIR | \
// RUN: iree-run-module \
// RUN:     --device=vulkan \
// RUN:     --module=- \
// RUN:     --function=mixed_invocation \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 | \
// RUN: FileCheck %s

// The configuration used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced and multiple variants with differing formats
// and compilation options (architectures, etc) can be embedded for runtime
// selection.
#spirv_target = #hal.executable.target<"vulkan-spirv", "vulkan-spirv-fb", {
  spirv.target_env = #spirv.target_env<
    #spirv.vce<v1.3, [Shader, GroupNonUniform], [SPV_KHR_storage_buffer_storage_class, SPV_KHR_variable_pointers]>,
    #spirv.resource_limits<max_compute_workgroup_size = [128, 128, 64], subgroup_size = 64>
  >
}>

// The target devices that the program will run on.
// These can come from compiler flags and multiple targets can be supported
// It's possible, for example, to support targeting multiple devices in the same
// compiled binary.
#vulkan_target = #hal.device.target<"vulkan", [
  #spirv_target
]>

module @example attributes {hal.device.targets = [#vulkan_target]} {

  // Executable containing hand-authored shaders.
  // Though each executable can contain multiple exported functions SPIR-V lacks
  // decent linking support and we model them separately here. In future
  // iterations of this functionality we will allow multiple .spv object files
  // to be linked into a single executable so that we can eliminate some binary
  // size overheads and improve runtime shader compilation performance.
  //
  // TODO(#7824): make export ordinals map to objects so that we can perform
  // our own linking (until some future Vulkan extension is added for pipeline
  // libraries).
  hal.executable.source private @simple_mul attributes {
    // Object files linked into the executable.
    // Certain backends (today) support either wholesale definition or linking
    // of partial objects for imports used by generated code. Each compilation
    // target can have its own unique set of objects to link in and the target
    // keys can be generic. This allows for an object file to be linked in based
    // only on the target triple while allowing for more specialized ones
    // requiring certain CPU features to be only included when building those.
    objects = #hal.executable.objects<{
      #spirv_target = [
        #hal.executable.object<{
          // Referencing a file path on disk but could also have the data
          // embedded in order to make the MLIR file hermetic/portable across
          // compilation pipelines. In the future we'll likely use MLIR's
          // external resource functionality for this. By allowing for the
          // objects to be embedded we can support JIT scenarios where some
          // layer higher or lower may be emitting the objects to link in as
          // part of the overall compilation.
          path = "samples/custom_dispatch/vulkan/shaders/simple_mul.spv"
        }>
      ]
    }>
  } {

    // TODO(benvanik): demonstrate hal.executable.constant.block for
    // specialization via host logic. These map to specialization constants.

    // Exported function with the C name `simple_mul`.
    // The ordinal must be assigned by the user and unique for the executable.
    // The layout defines the required bindings and push constants and can be
    // thought of as the function signature.
    hal.executable.export public @main ordinal(0)
        layout(#hal.pipeline.layout<push_constants = 1, sets = [
          <0, bindings = [
              <0, storage_buffer, ReadOnly>,
              <1, storage_buffer, ReadOnly>,
              <2, storage_buffer>
          ]>
        ]>) attributes {
          // Bindings are automatically inferred when possible as part of the
          // ABI but can be overridden if the user wants to use features such as
          // sparse bindings or multiple descriptor sets. To do so the
          // `hal.interface.bindings` attribute can be added to an export op as
          // follows mapping tensor operands/results to the pipeline layout
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
      // and limits (shared memory size, etc). The other arguments are the
      // values passed in the dispatch operation (usually things like root
      // output op tensor dimensions and other abstract values).
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    }

  }  // hal.executable

  hal.executable.source private @simple_mul_inplace attributes {
    objects = #hal.executable.objects<{
      #spirv_target = [
        #hal.executable.object<{
          path = "samples/custom_dispatch/vulkan/shaders/simple_mul_inplace.spv"
        }>
      ]
    }>
  } {
    // Similar to the above but in-place by using a read/write binding.
    hal.executable.export public @main ordinal(0)
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
  }  // hal.executable

  // Function demonstrating a few hand-authored dispatches mixed with codegen.
  // Invoke with:
  //  --device=vulkan
  //  --function=mixed_invocation
  //  --input=8xf32=2
  //  --input=8xf32=4
  // CHECK-LABEL: EXEC @mixed_invocation
  func.func @mixed_invocation(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    // HACK: for hand-authored shaders all primitive values passed in need to
    // be i32 or a bit-castable type. This is because ABI packing of other types
    // happens inside of the PackDispatchOperandsPass that is currently not
    // usable with external functions as it changes the ABI. In the future we
    // can better define the ABI such that it's possible to match the compiler
    // expectations around padding/alignment. For now users must do the packing
    // themselves (splitting i64 into i32+i32, etc).
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>
    %dim_i32 = arith.index_cast %dim : index to i32

    // Dispatch a basic `ret = lhs * rhs` shader.
    %0 = flow.dispatch @simple_mul::@main[%dim](%dim_i32, %arg0, %arg1) : (i32, tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}

    // Code gen some other ops - these will interleave with the hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?xf32>

    // Dispatch an in-place `rhs *= lhs` shader.
    %2 = flow.dispatch @simple_mul_inplace::@main[%dim](%dim_i32, %0, %1) : (i32, tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> %1{%dim}

    // CHECK: 8xf32=96 96 96 96 96 96 96 96
    return %2 : tensor<?xf32>
  }

}  // module
