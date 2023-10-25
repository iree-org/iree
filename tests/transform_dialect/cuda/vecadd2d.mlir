!type = tensor<9x512xf32>
!type2 = tensor<512x9xf32>

#trait = { indexing_maps  = [affine_map<(d0, d1) -> (d0, d1)>],
           iterator_types = ["parallel", "parallel"] }

#trait2 = { indexing_maps  = [affine_map<(d0, d1) -> (d0, d1)>,
                              affine_map<(d0, d1) -> (d1, d0)>],
           iterator_types = ["parallel", "parallel"] }

util.global private @"lhs" {noinline} = dense<0.0> : !type2
util.global private @"rhs" {noinline} = dense<2.0> : !type

func.func @vecadd2d() -> (!type2) {
  %cst0 = arith.constant 0.000000e+00 : f32
  %cst1 = arith.constant 2.000000e+00 : f32

  %x_ptr = util.global.address @"rhs" : !util.ptr<!type>
  %x = util.global.load.indirect %x_ptr : !util.ptr<!type> -> !type
  %y_ptr = util.global.address @"lhs" : !util.ptr<!type2>
  %y = util.global.load.indirect %y_ptr : !util.ptr<!type2> -> !type2

  // Note: Two linalg.generics to fill the tensors will make IREE generate two
  // separate kernels for the above and the below. It is important to validate
  // the results.
  %2 = linalg.generic #trait2 ins(%x : !type) outs(%y : !type2) {
  ^bb0(%arg3: f32, %arg4: f32):
    %3 = arith.addf %arg3, %arg4 : f32
    linalg.yield %3 : f32
  } -> !type2

  return %2 : !type2
}

// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline  \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-use-transform-dialect-strategy=%p/vecadd2d_codegen_spec.mlir | \
// RUN: FileCheck %s --check-prefix=CHECK

// RUN: iree-opt %s --iree-hal-target-backends=cuda \
// RUN:     --iree-abi-transformation-pipeline \
// RUN:     --iree-flow-transformation-pipeline  \
// RUN:     --iree-stream-transformation-pipeline \
// RUN:     --iree-hal-configuration-pipeline | \
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-use-transform-dialect-strategy=%p/vecadd2d_codegen_spec_partial_tile.mlir | \
// RUN: FileCheck %s --check-prefix=CHECK-PARTIAL-TILE

// RUN: iree-compile %s --iree-hal-target-backends=cuda \
// RUN:     --iree-opt-const-expr-hoisting=false --iree-opt-const-eval=false \
/// Constant JIT'ing must be disabled because the transform-dialect debug
/// flags leak to the JIT session, which doesn't know what to do with them.
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-use-transform-dialect-strategy=%p/vecadd2d_codegen_spec.mlir | \
// RUN: iree-run-module --module=- --function=vecadd2d --device=cuda |\
// RUN: FileCheck %s --check-prefix=EXEC

//     CHECK:  hal.executable.export
//     CHECK:  bb0(%[[DEV:.*]]: !hal.device):
//     CHECK:  %[[C171:.*]] = arith.constant 171 : index
//     CHECK:  %[[C1:.*]] = arith.constant 1 : index
//     CHECK:  %[[C2:.*]] = arith.constant 2 : index
//     CHECK:  hal.return %[[C171]], %[[C1]], %[[C2]] : index, index, index

//     CHECK:  %[[BLKZ:.*]] = hal.interface.workgroup.id[2] : index
//     CHECK:  %[[BLKX:.*]] = hal.interface.workgroup.id[0] : index
//     CHECK:  memref.subview %0[%[[BLKZ:.*]], %[[BLKX:.*]]]

//     CHECK-PARTIAL-TILE:  hal.executable.export
//     CHECK-PARTIAL-TILE:  bb0(%[[DEV:.*]]: !hal.device):
//     CHECK-PARTIAL-TILE:  %[[C1:.*]] = arith.constant 1 : index
//     CHECK-PARTIAL-TILE:  %[[C1_2:.*]] = arith.constant 1 : index
//     CHECK-PARTIAL-TILE:  %[[C171:.*]] = arith.constant 171 : index
//     CHECK-PARTIAL-TILE:  hal.return %[[C1]], %[[C1_2]], %[[C171]] : index, index, index

//      EXEC: EXEC @vecadd2d
//      EXEC: result[0]: hal.buffer_view
//      EXEC: 512x9xf32=[2 2 2 2 2 2 2 2 2][2 2 2 2 2 2 2 2 2]
