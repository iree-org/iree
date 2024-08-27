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
  iree.gpu.target = #iree_gpu.target<
    arch = "", features = "spirv:v1.3,cap:Shader", wgp = <
      compute = fp32|int32, storage = b32, subgroup = none,
      dot = none, mma = [], subgroup_size_choices = [64, 64],
      max_workgroup_sizes = [128, 128, 64], max_thread_count_per_workgroup = 128,
      max_workgroup_memory_bytes = 16384,
      max_workgroup_counts = [65535, 65535, 65535]>
  >
}>

// The target devices that the program will run on.
// These can come from compiler flags and multiple targets can be supported
// It's possible, for example, to support targeting multiple devices in the same
// compiled binary.
#vulkan_target = #hal.device.target<"vulkan", [#spirv_target]> : !hal.device

module @example attributes {hal.device.targets = [#vulkan_target]} {

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
    // Note that not all backends use names or the names are derived from
    // ordinals so we include that (`:ordinal`).
    %0 = hal.dispatch.extern "main"[%dim](%dim_i32, %arg0, %arg1) : (i32, tensor<?xf32>{%dim}, tensor<?xf32>{%dim}) -> tensor<?xf32>{%dim}
      // This host function is used to compute the XYZ workgroup count
      // dispatched at runtime. It can query the %device for capabilities
      // and limits (shared memory size, etc). The other arguments are the
      // values passed in the dispatch operation (usually things like root
      // output op tensor dimensions and other abstract values).
      count(%device: !hal.device, %workload: index) -> (index, index, index) {
        %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
        %c1 = arith.constant 1 : index
        hal.return %x, %c1, %c1 : index, index, index
      }
      // The layout defines the required bindings and push constants and can be
      // thought of as the function signature.
      layout(#hal.pipeline.layout<constants = 1, bindings = [
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer, ReadOnly>,
        #hal.pipeline.binding<storage_buffer>
      ]>)
      // Object files linked into the executable.
      // Certain backends (today) support either wholesale definition or linking
      // of partial objects for imports used by generated code. Each compilation
      // target can have its own unique set of objects to link in and the target
      // keys can be generic. This allows for an object file to be linked in based
      // only on the target triple while allowing for more specialized ones
      // requiring certain CPU features to be only included when building those.
      objects({
        #spirv_target ordinal(0) = [
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
      })

    // Code gen some other ops - these will interleave with the hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?xf32>

    // CHECK: 8xf32=12 12 12 12 12 12 12 12
    return %1 : tensor<?xf32>
  }

}  // module
