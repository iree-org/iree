// RUN: iree-compile --iree-hal-target-device=cuda --iree-cuda-target=sm_86 \
// RUN:     --mlir-print-ir-after-all --mlir-disable-threading=true %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

// This is an end-to-end test that mirrors the behavior observed when compiling
// `test.mlir` with:
//   iree-compile --iree-hal-target-device=cuda --iree-cuda-target=sm_86 \
//                --mlir-print-ir-after-all test.mlir -o test.vmfb 2> ir_dump.log
//
// It verifies that:
// 1) After FuseDispatchBindingsPass (iree-stream-fuse-dispatch-bindings),
//    stream.binding_noalias attributes are attached to !stream.binding
//    arguments of the dispatch entry point.
// 2) After ConvertToNVVMPass (iree-convert-to-nvvm),
//    llvm.noalias attributes are applied to the corresponding LLVM pointer
//    arguments.

// CHECK-LABEL: // -----// IR Dump After FuseDispatchBindingsPass (iree-stream-fuse-dispatch-bindings) //----- //
// CHECK: builtin.module {
// CHECK:   func.func @sort_3d_dispatch_0_sort_DxDxDxi32(
// CHECK-SAME: %arg0: !stream.binding {{.*}}stream.binding_noalias = [1 : i32]
// CHECK-SAME: %arg1: !stream.binding {{.*}}stream.binding_noalias = [0 : i32]

// CHECK-LABEL: // -----// IR Dump After ConvertToNVVMPass (iree-convert-to-nvvm) //----- //
// CHECK: module {
// CHECK:   llvm.func @sort_3d_dispatch_0_sort_DxDxDxi32(
// CHECK-SAME: %arg0: !llvm.ptr<1> {{.*}} llvm.noalias, llvm.nonnull, llvm.noundef
// CHECK:   llvm.func @sort_3d_dispatch_0_sort_DxDxDxi32(
// CHECK-SAME: %arg1: !llvm.ptr<1> {{.*}} llvm.noalias, llvm.nonnull, llvm.noundef

// Minimal reproduction of `test.mlir` used for the end-to-end pipeline.
util.func public @sort_3d(%arg0: tensor<?x?x?xi32>, %arg1 : tensor<?x?x?xf32>)
    -> (tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
  %0, %1 = iree_linalg_ext.sort dimension(0)
      outs(%arg0, %arg1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>) {
      ^bb0(%arg2: i32, %arg3: i32, %arg4 : f32, %arg5 : f32):
        %2 = arith.cmpf ogt, %arg4, %arg5 : f32
        iree_linalg_ext.yield %2 : i1
      } -> tensor<?x?x?xi32>, tensor<?x?x?xf32>
  util.return %0, %1 : tensor<?x?x?xi32>, tensor<?x?x?xf32>
}
