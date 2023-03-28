// RUN: iree-opt %s --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-llvmgpu-use-transform-dialect=%s | \
// RUN: FileCheck --check-prefix=CHECK %s

hal.executable @_attention_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb, target = <"cuda", "cuda-nvptx-fb", {target_arch = "sm_35"}> {
    hal.executable.export public @_attention_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @_attention_dispatch_0() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf32>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf32>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf32>> -> tensor<192x1024x64xf32>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf32>> -> tensor<192x1024x64xf32>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf32>> -> tensor<192x1024x64xf32>
        %7 = tensor.empty() : tensor<192x1024x64xf32>
        %8 = iree_linalg_ext.attention ins(%4, %5, %6 : tensor<192x1024x64xf32>, tensor<192x1024x64xf32>, tensor<192x1024x64xf32>) outs(%7 : tensor<192x1024x64xf32>) -> tensor<192x1024x64xf32>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : tensor<192x1024x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf32>>
        return
      }
    }
  }
}

transform.sequence failures(propagate) {
  ^bb0(%variant_op: !pdl.operation):

    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!pdl.operation) -> !pdl.operation

    // Tile and distribute to workgroups
    // ==========================================
    %forall_grid, %tiled_attention =
    transform.structured.tile_to_forall_op %attention tile_sizes [1, 128]
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )

    // Tile and decompose attention
    // ==========================================
    %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    %outer_loop, %max_fill, %sum_fill, %inner_loop, %fill_op, %first_matmul, %reduce_max, %partial_softmax, %reduce_sum, %update,
    %softmax, %scale_acc, %second_matmul = tile_and_decompose_attention %attention2 :
       (!pdl.operation) -> (!pdl.operation, !pdl.operation,!pdl.operation,  !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation, !pdl.operation)

    // Vectorize function
    // ==========================================
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func {  rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
    %func_3 = transform.structured.vectorize %func

    // Bufferization
    // ==========================================
    transform.iree.apply_patterns %func_3
      { fold_reassociative_reshapes, canonicalization, tiling_canonicalization, cse } : (!pdl.operation) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
    transform.iree.apply_patterns %func_3 { erase_unnecessary_tensor_operands } : (!pdl.operation) -> ()
    %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!pdl.operation) -> (!pdl.operation)
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.erase_hal_descriptor_type_from_memref %memref_func : (!pdl.operation) -> ()

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
    transform.iree.forall_to_workgroup %func_7 : (!pdl.operation) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [128, 1, 1] : (!pdl.operation) -> ()

    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!pdl.operation) -> !pdl.operation
    transform.iree.apply_patterns %func_8 { canonicalization } : (!pdl.operation) -> ()
    transform.iree.apply_patterns %func_8 { cse } : (!pdl.operation) -> ()
    transform.iree.apply_buffer_optimizations %func_8 : (!pdl.operation) -> ()
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-DAG:  func.func @_attention_dispatch_0() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<-1.000000e+30> : vector<128xf32>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<0.000000e+00> : vector<128xf32>
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<128x128xf32>
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<192x1024x64xf32>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<192x1024x64xf32>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:        %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:      %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_Y]]]
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D3]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:     : memref<192x1024x64xf32> to memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:        %[[D5:.+]] = vector.transfer_read %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]], %[[CST_2]] {in_bounds
// CHECK-SAME:     = [true, true]} : memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>, vector<128x64xf32>
// CHECK:        %[[D6:.+]]:3 = scf.for %[[ARG0:.+]] = %[[C0]] to %[[C1024]] step %[[C128]]
// CHECK-SAME:     iter_args(%[[ARG1:.+]] = %[[CST]], %[[ARG2:.+]] = %[[CST_0]], %[[ARG3:.+]] = %[[D5]]) -> (vector<128xf32>,
// CHECK-SAME:     vector<128xf32>, vector<128x64xf32>) {
// CHECK:          %[[D7:.+]] = vector.transfer_read %[[D0]][%[[WORKGROUP_ID_X]], %[[D4]], %[[C0]]], %[[CST_2]]
// CHECK-SAME:       {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:          %[[D8:.+]] = vector.transfer_read %[[D1]][%[[WORKGROUP_ID_X]], %[[ARG0]], %[[C0]]], %[[CST_2]]
// CHECK-SAME:       {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:          %[[D9:.+]] = vector.contract {indexing_maps = [#[[MAP1]], #[[MAP2]], #[[MAP3]]], iterator_types
// CHECK-SAME:       = ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>} %[[D7]], %[[D8]],
// CHECK-SAME:       %[[CST_1]] : vector<128x64xf32>, vector<128x64xf32> into vector<128x128xf32>
// CHECK:          %[[D10:.+]] = vector.multi_reduction <maxf>, %[[D9]], %[[ARG1]] [1] : vector<128x128xf32> to
// CHECK-SAME:       vector<128xf32>
// CHECK:          %[[D11:.+]] = vector.broadcast %[[D10]] : vector<128xf32> to vector<128x128xf32>
// CHECK:          %[[D12:.+]] = vector.transpose %[[D11]], [1, 0] : vector<128x128xf32> to vector<128x128xf32>
// CHECK:          %[[D13:.+]] = arith.subf %[[D9]], %[[D12]] : vector<128x128xf32>
// CHECK:          %[[D14:.+]] = math.exp %[[D13]] : vector<128x128xf32>
// CHECK:          %[[D15:.+]] = arith.subf %[[ARG1]], %[[D10]] : vector<128xf32>
// CHECK:          %[[D16:.+]] = math.exp %[[D15]] : vector<128xf32>
// CHECK:          %[[D17:.+]] = arith.mulf %[[D16]], %[[ARG2]] : vector<128xf32>
// CHECK:          %[[D18:.+]] = vector.multi_reduction <add>, %[[D14]], %[[D17]] [1] : vector<128x128xf32> to
// CHECK-SAME:       vector<128xf32>
// CHECK:          %[[D19:.+]] = vector.broadcast %[[D18]] : vector<128xf32> to vector<128x128xf32>
// CHECK:          %[[D20:.+]] = vector.transpose %[[D19]], [1, 0] : vector<128x128xf32> to vector<128x128xf32>
// CHECK:          %[[D21:.+]] = arith.divf %[[D14]], %[[D20]] : vector<128x128xf32>
// CHECK:          %[[D22:.+]] = vector.broadcast %[[D17]] : vector<128xf32> to vector<64x128xf32>
// CHECK:          %[[D23:.+]] = vector.broadcast %[[D18]] : vector<128xf32> to vector<64x128xf32>
// CHECK:          %[[D24:.+]] = arith.divf %[[D22]], %[[D23]] : vector<64x128xf32>
// CHECK:          %[[D25:.+]] = vector.transpose %[[D24]], [1, 0] : vector<64x128xf32> to vector<128x64xf32>
// CHECK:          %[[D26:.+]] = arith.mulf %[[D25]], %[[ARG3]] : vector<128x64xf32>
// CHECK:          %[[D27:.+]] = vector.transfer_read %[[D2]][%[[WORKGROUP_ID_X]], %[[ARG0]], %[[C0]]], %[[CST_2]]
// CHECK-SAME:       {in_bounds = [true, true]} : memref<192x1024x64xf32>, vector<128x64xf32>
// CHECK:          %[[D28:.+]] = vector.contract {indexing_maps = [#[[MAP1]], #[[MAP4]], #[[MAP3]]],
// CHECK-SAME:       iterator_types = ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>}
// CHECK-SAME:       %[[D21]], %[[D27]], %[[D26]] : vector<128x128xf32>, vector<128x64xf32> into
// CHECK-SAME:       vector<128x64xf32>
// CHECK:          scf.yield %[[D10]], %[[D18]], %[[D28]] : vector<128xf32>, vector<128xf32>, vector<128x64xf32>
// CHECK:        }
// CHECK:        vector.transfer_write %[[D6]]#[[D2:.+]], %[[SUBVIEW]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds =
// CHECK-SAME:     [true, true]} : vector<128x64xf32>, memref<1x128x64xf32, strided<[65536, 64, 1], offset: ?>>
// CHECK:        return
