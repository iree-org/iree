// RUN: iree-opt %s --split-input-file \
// RUN:   --pass-pipeline="builtin.module(func.func(iree-codegen-gpu-prune-dps-ops))" | FileCheck %s

// CHECK-LABEL: func.func @minimal_unused_result_removed
// CHECK-DAG: %[[SRC:.*]] = bufferization.alloc_tensor() copy(%{{.*}}) {memory_space = #gpu.address_space<private>} : tensor<1x10xf32>
// CHECK-DAG: iree_linalg_ext.sort{{.*}} dimension(1) outs(%[[SRC]], %{{.*}} : tensor<1x10xf32>, tensor<1x10xi64>)
#layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @minimal_unused_result_removed(%arg0: index, %arg1: index, %loopIn: tensor<1x10xi64>) -> tensor<1x10xi64> {
  %c0 = arith.constant 0 : index
  %9 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x10xf32>>{%arg1}
  %11 = iree_tensor_ext.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%arg1, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x10xf32>>{%arg1} -> tensor<?x10xf32>
  %extracted_slice = tensor.extract_slice %11[%arg0, 0] [1, 10] [1, 1] : tensor<?x10xf32> to tensor<1x10xf32>
  %14 = scf.forall (%arg2) in (1) shared_outs(%arg3 = %loopIn) -> (tensor<1x10xi64>) {
    %15:2 = iree_linalg_ext.sort dimension(1) outs(%extracted_slice, %arg3 : tensor<1x10xf32>, tensor<1x10xi64>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: i64, %arg7: i64):
      %16 = arith.cmpf oge, %arg4, %arg5 : f32
      iree_linalg_ext.yield %16 : i1
    } -> tensor<1x10xf32>, tensor<1x10xi64>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %15#1 into %arg3[0, 0] [1, 10] [1, 1] : tensor<1x10xi64> into tensor<1x10xi64>
    }
  }
  return %14 : tensor<1x10xi64>
}

// -----

// CHECK-LABEL: func.func @minimal_unused_result_not_removed
// CHECK-NOT: bufferization.alloc_tensor()
#layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @minimal_unused_result_not_removed(%arg0: index, %arg1: index, %loopIn: tensor<1x10xi64>) -> tensor<1x10xi64> {
  %c0 = arith.constant 0 : index
  %cst = arith.constant dense<1.000000e+00> : tensor<1x10xf32>
  %9 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x10xf32>>{%arg1}
  %11 = iree_tensor_ext.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%arg1, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x10xf32>>{%arg1} -> tensor<?x10xf32>
  %extracted_slice = tensor.extract_slice %11[%arg0, 0] [1, 10] [1, 1] : tensor<?x10xf32> to tensor<1x10xf32>
  %add = arith.addf %extracted_slice, %cst: tensor<1x10xf32>
  %14 = scf.forall (%arg2) in (1) shared_outs(%arg3 = %loopIn) -> (tensor<1x10xi64>) {
    %15:2 = iree_linalg_ext.sort dimension(1) outs(%add, %arg3 : tensor<1x10xf32>, tensor<1x10xi64>) {
    ^bb0(%arg4: f32, %arg5: f32, %arg6: i64, %arg7: i64):
      %16 = arith.cmpf oge, %arg4, %arg5 : f32
      iree_linalg_ext.yield %16 : i1
    } -> tensor<1x10xf32>, tensor<1x10xi64>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %15#1 into %arg3[0, 0] [1, 10] [1, 1] : tensor<1x10xi64> into tensor<1x10xi64>
    }
  }
  return %14 : tensor<1x10xi64>
}

// -----

// CHECK-LABEL: func.func @full_example_ready_to_compile
// CHECK-DAG: %[[SRC:.*]] = bufferization.alloc_tensor() copy(%{{.*}}) {memory_space = #gpu.address_space<private>} : tensor<1x10xf32>
// CHECK-DAG: iree_linalg_ext.sort{{.*}} dimension(1) outs(%[[SRC]], %{{.*}} : tensor<1x10xf32>, tensor<1x10xi64>)
#layout = #hal.pipeline.layout<constants = 2, bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
func.func @full_example_ready_to_compile() attributes {translation_info = #iree_codegen.translation_info<pipeline = LLVMGPUTileAndFuse workgroup_size = [128, 1, 1] subgroup_size = 64>} {
  %c0 = arith.constant 0 : index
  %c32_i64 = arith.constant 32 : i64
  %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
  %1 = hal.interface.constant.load layout(#layout) ordinal(1) : i32
  %2 = arith.extui %0 : i32 to i64
  %3 = arith.extui %1 : i32 to i64
  %4 = arith.shli %3, %c32_i64 : i64
  %5 = arith.ori %2, %4 : i64
  %6 = arith.index_castui %5 : i64 to index
  %7 = util.assume.int %6<umin = 0, umax = 9007199254740991> : index
  %8 = iree_tensor_ext.dispatch.workload.ordinal %7, 0 : index
  %9 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x10xf32>>{%8}
  %10 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x10xi64>>{%8}
  %11 = iree_tensor_ext.dispatch.tensor.load %9, offsets = [0, 0], sizes = [%8, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x10xf32>>{%8} -> tensor<?x10xf32>
  %12 = iree_tensor_ext.dispatch.tensor.load %10, offsets = [0, 0], sizes = [%8, 10], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x10xi64>>{%8} -> tensor<?x10xi64>
  %13 = scf.forall (%arg0) in (%8) shared_outs(%arg1 = %12) -> (tensor<?x10xi64>) {
    %extracted_slice = tensor.extract_slice %11[%arg0, 0] [1, 10] [1, 1] : tensor<?x10xf32> to tensor<1x10xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg0, 0] [1, 10] [1, 1] : tensor<?x10xi64> to tensor<1x10xi64>
    %14 = scf.forall (%arg2) in (1) shared_outs(%arg3 = %extracted_slice_0) -> (tensor<1x10xi64>) {
      %15:2 = iree_linalg_ext.sort {lowering_config = #iree_gpu.lowering_config<{thread = [1, 0], workgroup = [1, 0]}>} dimension(1) outs(%extracted_slice, %arg3 : tensor<1x10xf32>, tensor<1x10xi64>) {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: i64, %arg7: i64):
        %16 = arith.cmpf oge, %arg4, %arg5 : f32
        iree_linalg_ext.yield %16 : i1
      } -> tensor<1x10xf32>, tensor<1x10xi64>
      scf.forall.in_parallel {
        tensor.parallel_insert_slice %15#1 into %arg3[0, 0] [1, 10] [1, 1] : tensor<1x10xi64> into tensor<1x10xi64>
      }
    } {mapping = [#gpu.thread<linear_dim_0>]}
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %14 into %arg1[%arg0, 0] [1, 10] [1, 1] : tensor<1x10xi64> into tensor<?x10xi64>
    }
  } {mapping = [#iree_codegen.workgroup_mapping<x>]}
  iree_tensor_ext.dispatch.tensor.store %13, %10, offsets = [0, 0], sizes = [%8, 10], strides = [1, 1] : tensor<?x10xi64> -> !iree_tensor_ext.dispatch.tensor<readwrite:tensor<?x10xi64>>{%8}
  return
}
