// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-llvmgpu-configuration-pipeline), iree-codegen-linalg-to-nvvm-pipeline)))' --iree-gpu-test-target=sm_80 -split-input-file %s -o - | FileCheck %s

// This test checks that the lowering of nvvm includes the extraction
// and optimization of address computations.

// The main goal here is to check that the loop invariant part of
// the address computation of the first ldmatrix is hoisted outside
// of the loop.
// Couple of notes:
// - We don't actually check that the computed offset feeds the ldmatrix.
//   Instead we collect indirect evidence that it does. The rationale is
//   the check lines would get messy because we would have to check that
//   the offset is properly inserted to then extracted from the memref
//   descriptor.
// - The current check lines anchor themselves on two input values: laneid
//   and tid.y. The actual instructions these two values go through is not
//   particularly interesting, but we need to match the full def-use chain
//   nonetheless to make sure that the hoisting happened as expected.
//
// Long story short, in this test we want to match:
// ```
// entry:
//   v1 = laneid
//   v2 = tid.y
//   loop_invariant = some_math(laneid, tid.y)
// loop:
//   loop_variant_part_of_offset = some_math(loop_variant)
//   final_address = loop_variant_part_of_offset + loop_invariant
//   ... = ldmatrix final_address
// ```
// Where the important part is that loop_invariant is outside the loop
// and is contributed back to the final address with just one instruction.

// Match the interesting constants.
// CHECK-DAG: %[[C2:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK-DAG: %[[C6:.*]] = llvm.mlir.constant(6 : i32) : i32
// CHECK-DAG: %[[C16:.*]] = llvm.mlir.constant(16 : i32) : i32
// CHECK-DAG: %[[C64:.*]] = llvm.mlir.constant(64 : i32) : i32
// CHECK-DAG: %[[C4096:.*]] = llvm.mlir.constant(4096 : index) : i64
// CHECK-DAG: %[[C8192:.*]] = llvm.mlir.constant(8192 : index) : i64
//
// Match the interesting special registers.
// CHECK-DAG: %[[TID_Y:.*]] = nvvm.read.ptx.sreg.tid.y range <i32, 0, 2> : i32
// CHECK-DAG: %[[TID_Y_EXT:.*]] = llvm.sext %[[TID_Y]] : i32 to i64
// CHECK-DAG: %[[TID_Y_TRUNC:.*]] = llvm.trunc %[[TID_Y_EXT]] : i64 to i32
// CHECK-DAG: %[[LANEID:.*]] = nvvm.read.ptx.sreg.laneid range <i32, 0, 32> : i32
// CHECK-DAG: %[[LANEID_EXT:.*]] = llvm.sext %[[LANEID]] : i32 to i64
// CHECK-DAG: %[[LANEID_TRUNC:.*]] = llvm.trunc %[[LANEID_EXT]] : i64 to i32
// CHECK-DAG: %[[TID_Y_IDX:.*]] = llvm.mul %[[TID_Y_TRUNC]], %[[C64]] overflow<nsw> : i32
//
// Match the loop invariant math on the special registers.
// CHECK: %[[GRP_IDX:.*]] = llvm.add %[[TID_Y_IDX]], %[[LANEID_TRUNC]]  : i32
// CHECK: %[[GRP_IDX1:.*]] = llvm.add %[[GRP_IDX]], %{{.*}}  : i32
// CHECK: %[[GRP_IDX2:.*]] = llvm.and %[[GRP_IDX1]], %[[C6]]  : i32
// CHECK: %[[GRP_IDX3:.*]] = llvm.shl %[[GRP_IDX2]], %[[C2]]  : i32
// CHECK: %{{.*}} = llvm.xor %[[SRC:.*]], %[[GRP_IDX3]]  : i32
// CHECK: %[[ADJ_SRC:.*]] = llvm.add %[[SRC]], %[[C16]]  : i32
// CHECK: %[[INV:.*]] = llvm.xor %[[ADJ_SRC]], %[[GRP_IDX3]]  : i32
// CHECK: %[[INV_EXT:.*]] = llvm.zext %[[INV]] : i32 to i64
//
// Find the basic block boundary.
// CHECK: llvm.br ^[[LOOP_BODY:bb[0-9]+]](
//
// Grab the iv (this check is probably brittle)
// CHECK: {{^ *}}^[[LOOP_BODY]]({{.*}}, %{{[^:]*}}: !llvm.array<2 x vector<2xf16>>, %[[IV:.*]]: i64, %{{[^:]*}}: i64, %{{[^:]*}}: !llvm.array
//
// Match the loop variant part of the address computation.
// CHECK: %[[VAR:.*]] = llvm.mul %[[IV]], %[[C4096]]
//
// Add the loop invariant part.
// CHECK: %[[OFF:.*]] = llvm.add %{{.*}}, %[[INV_EXT]]
//
// Store the resulting offset in the memref descriptor.
// llvm.insert %[[OFF]], %{{.*}}[2]
//
// Just double check that we captured the IV
// CHECK: %[[IV_NEXT:.*]] = llvm.mul %[[IV]], %[[C8192]]  : i64
#executable_target_cuda_nvptx_fb = #hal.executable.target<"cuda", "cuda-nvptx-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
hal.executable private @matmul_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb target(#executable_target_cuda_nvptx_fb) {
    hal.executable.export public @matmul_dispatch_0_matmul_2560x2560x2560 ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device, %arg1: index, %arg2: index, %arg3: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root %arg1, %arg2, %arg3
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_dispatch_0_matmul_2560x2560x2560() {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f16
        %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2560x2560xf16>>
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2560x2560xf16>>
        %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2560x2560xf16>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2560, 2560], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2560x2560xf16>> -> tensor<2560x2560xf16>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2560, 2560], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2560x2560xf16>> -> tensor<2560x2560xf16>
        %5 = tensor.empty() : tensor<2560x2560xf16>
        %6 = linalg.fill ins(%cst : f16) outs(%5 : tensor<2560x2560xf16>) -> tensor<2560x2560xf16>
        %7 = linalg.matmul ins(%3, %4 : tensor<2560x2560xf16>, tensor<2560x2560xf16>) outs(%6 : tensor<2560x2560xf16>) -> tensor<2560x2560xf16>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2560, 2560], strides = [1, 1] : tensor<2560x2560xf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2560x2560xf16>>
        return
      }
    }
  }
}
