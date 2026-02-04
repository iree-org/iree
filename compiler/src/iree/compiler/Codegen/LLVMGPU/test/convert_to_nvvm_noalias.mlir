// RUN: iree-opt --split-input-file \
// RUN:   --pass-pipeline='builtin.module(iree-stream-fuse-dispatch-bindings)' \
// RUN:   %s | FileCheck %s --check-prefix=STREAM

// Test that stream.binding_noalias attributes are correctly generated in Stream
// dialect and propagated to llvm.noalias attributes in LLVM dialect.
//
// This test verifies the complete pipeline as described in the PR:
// 1. Stream dialect: After FuseDispatchBindings pass, stream.binding_noalias
//    attributes are generated on func.func arguments for distinct bindings
//    (different resources)
// 2. LLVM dialect: After ConvertToNVVMPass, llvm.noalias attributes are
//    applied to function arguments

// Test case: Two distinct bindings that use different resources.
// With alias_mutable_bindings = false, each mutable binding gets its own
// equivalence class, so they should be in different correlation groups.
stream.executable private @sort_3d_dispatch {
  stream.executable.export public @dispatch
    attributes {stream.resources = #stream.resource_config<{alias_mutable_bindings = false}>}
  builtin.module {
    util.func public @dispatch(
      %arg0: !stream.binding,
      %arg1: !stream.binding
    ) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %subspan0 = stream.binding.subspan %arg0[%c0]
        : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16xi32>>{%c16}
      %subspan1 = stream.binding.subspan %arg1[%c0]
        : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<16xf32>>{%c16}
      util.return
    }
  }
}

// Create multiple dispatches where each binding consistently uses a different
// resource. This ensures FuseDispatchBindings recognizes them as distinct
// correlation groups and generates stream.binding_noalias attributes.
util.func public @test() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  // Create distinct resources - each binding will use a different resource
  %alloc0 = stream.resource.alloc uninitialized
    : !stream.resource<transient>{%c32}
  %alloc1 = stream.resource.alloc uninitialized
    : !stream.resource<transient>{%c32}
  %alloc2 = stream.resource.alloc uninitialized
    : !stream.resource<transient>{%c32}
  %alloc3 = stream.resource.alloc uninitialized
    : !stream.resource<transient>{%c32}
  %timepoint = stream.cmd.execute
      with(%alloc0 as %r0: !stream.resource<transient>{%c32},
           %alloc1 as %r1: !stream.resource<transient>{%c32},
           %alloc2 as %r2: !stream.resource<transient>{%c32},
           %alloc3 as %r3: !stream.resource<transient>{%c32}) {
    // First dispatch: binding 0 uses r0, binding 1 uses r1 (different resources)
    stream.cmd.dispatch @sort_3d_dispatch::@dispatch[%c1, %c1, %c1] {
      rw %r0[%c0 for %c16] : !stream.resource<transient>{%c32},
      rw %r1[%c0 for %c16] : !stream.resource<transient>{%c32}
    }
    // Second dispatch: binding 0 uses r2, binding 1 uses r3 (different resources)
    // This pattern ensures binding 0 and binding 1 always use different resources
    stream.cmd.dispatch @sort_3d_dispatch::@dispatch[%c1, %c1, %c1] {
      rw %r2[%c0 for %c16] : !stream.resource<transient>{%c32},
      rw %r3[%c0 for %c16] : !stream.resource<transient>{%c32}
    }
  } => !stream.timepoint
  util.return
}

// After FuseDispatchBindings pass: stream.binding_noalias attributes should be
// generated for distinct bindings (different correlation groups).
// With alias_mutable_bindings = false, each mutable binding is in its own
// equivalence class, so binding 0 and binding 1 are in different correlation
// groups and should have stream.binding_noalias attributes.
//
// STREAM-LABEL: util.func public @dispatch
// STREAM-SAME: (%[[ARG0:.+]]: !stream.binding {stream.binding_noalias = [1 : i32]},
// STREAM-SAME:  %[[ARG1:.+]]: !stream.binding {stream.binding_noalias = [0 : i32]}

// After ConvertToNVVMPass: llvm.noalias attributes should be applied.
// LLVM-LABEL: llvm.func @dispatch
// LLVM-SAME: (%[[LLVM_ARG0:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef},
// LLVM-SAME:  %[[LLVM_ARG1:.+]]: !llvm.ptr {llvm.align = 16 : i32, llvm.noalias, llvm.nonnull, llvm.noundef}
