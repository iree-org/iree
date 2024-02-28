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
// from the IREE `stream` dialect layer. This allows for much more optimization
// opportunity as bindings and operands can be mutated by the compiler unlike
// approaches targeting the `hal` layer.
//
// In the future with improvements to lowerings into LLVM we should be able to
// do this from the `flow` layer as well and thus connect all the way to
// frontends. Currently memrefs are blocking the ability to define useful
// external functions at the higher layers.
//
// By enabling this from higher levels of the stack it's possible for users to
// hand-author what they need and have that get whole-program optimized,
// multi-device scheduled, and portably deployed with (nearly) all the features
// of IREE-generated workloads.

// The configurations used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced and multiple variants with differing formats
// and compilation options (architectures, etc) can be embedded for runtime
// selection. By fully specifying the targets here we can target multiple
// architectures and it's always possible to embed these instead of using the
// coarse command line compiler flags that only set single targets.
#arm_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-none-elf"
}>
#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf"
}>

// The target devices that the program will run on.
// These can come from compiler flags and multiple targets can be supported
// It's possible, for example, to support targeting multiple devices in the same
// compiled binary (CPU + Vulkan, etc).
#cpu_target = #hal.device.target<"llvm-cpu", [
  #arm_64_target,
  #x86_64_target
]>

module @example attributes {hal.device.targets = [#cpu_target]} {

  // Executable containing exported shims and calls to external functions.
  // Each executable can contain multiple exported functions and variants for
  // different architectures or even devices. It's also possible to mix hand-
  // authored functions with code generated ones even for the same functions
  // such that code generation is used as a fallback when the hand-authored
  // kernels aren't supported at runtime.
  stream.executable private @executable attributes {
    // Object files linked into the executable.
    // These object files are linked into the dynamic library and must meet
    // the requirements for embedded ELF linkage (no TLS, no globals, no
    // syscalls, no libc, etc). Each compilation target can have its own unique
    // set of objects to link in and the target keys can be generic. This allows
    // for an object file to be linked in based only on the target triple while
    // allowing for more specialized ones requiring certain CPU features to be
    // only included when building those. For this example we just use the
    // fully-specified targets.
    hal.executable.objects = #hal.executable.objects<{
      #arm_64_target = [
        #hal.executable.object<{
          // Referencing a file path on disk but could also have the data
          // embedded in order to make the MLIR file hermetic/portable across
          // compilation pipelines. In the future we'll likely use MLIR's
          // external resource functionality for this. By allowing for the
          // objects to be embedded we can support JIT scenarios where some
          // layer higher or lower may be emitting the objects to link in as
          // part of the overall compilation.
          path = "samples/custom_dispatch/cpu/embedded/functions_arm_64.o"
        }>
      ],
      #x86_64_target = [
        #hal.executable.object<{
          path = "samples/custom_dispatch/cpu/embedded/functions_x86_64.o"
        }>
      ]
    }>
  } {
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

    // Similar to the above but in-place by using a read/write binding.
    stream.executable.export public @simple_mul_inplace workgroups(%workload: index) -> (index, index, index) {
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      hal.return %x, %c1, %c1 : index, index, index
    }

    builtin.module {
      // External function declaration using a user-chosen calling convention.
      func.func private @simple_mul_workgroup(%binding0: memref<?xf32>, %binding1: memref<?xf32>, %binding2: memref<?xf32>, %dim: index, %tid: index) attributes {
        // Ensures that we try to statically link this external function and
        // pull it in from the object file.
        hal.import.static
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

        // Call the externally defined C function with an (almost) plain C
        // calling convention (see above for details about the mess memrefs
        // turn into).
        //
        // TODO: there are ways of accessing CPU information here such as
        // active architecture and feature bits but it is not yet exposed to
        // the stream level.
        func.call @simple_mul_workgroup(%memref0, %memref1, %memref2, %dim, %tid) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()

        // NOTE: this is code generated as normal - other MLIR ops can be used
        // here for looping/control flow, vector operations, linalg, etc.
        // This simple sample is just calling out to the external function but
        // microkernels fused with other code are possible.

        return
      }

      func.func private @simple_mul_inplace_workgroup(%binding0: memref<?xf32>, %binding1: memref<?xf32>, %dim: index, %tid: index) attributes {
        hal.import.static
      }
      func.func @simple_mul_inplace(
          %binding0: !stream.binding,
          %binding1: !stream.binding,
          %dim: index) {
        %c0 = arith.constant 0 : index

        %workgroup_id_x = stream.dispatch.workgroup.id[0] : index
        %tid = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]

        // Same as above but note that we're treating %binding1 as read/write.
        %memref0 = stream.binding.subspan %binding0[%c0] : !stream.binding -> memref<?xf32>{%dim}
        %memref1 = stream.binding.subspan %binding1[%c0] : !stream.binding -> memref<?xf32>{%dim}

        func.call @simple_mul_inplace_workgroup(%memref0, %memref1, %dim, %tid) : (memref<?xf32>, memref<?xf32>, index, index) -> ()

        return
      }
    }
  }

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
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>

    // Dispatch a basic `ret = lhs * rhs` using an external function.
    // This form (@executable::@export) allows for automatic variant selection
    // when multi-targeting.
    %0 = flow.dispatch @executable::@simple_mul[%dim](%arg0, %arg1, %dim) : (tensor<?xf32>{%dim}, tensor<?xf32>{%dim}, index) -> tensor<?xf32>{%dim}

    // Code gen some other ops - these will interleave with the hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?xf32>

    // Dispatch an in-place `rhs *= lhs` using an external function.
    %2 = flow.dispatch @executable::@simple_mul_inplace[%dim](%0, %1, %dim) : (tensor<?xf32>{%dim}, tensor<?xf32>{%dim}, index) -> %1{%dim}

    // CHECK: 8xf32=96 96 96 96 96 96 96 96
    return %2 : tensor<?xf32>
  }

}  // module
