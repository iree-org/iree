// RUN: iree-opt %s --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-codegen-materialize-user-configs, iree-llvmgpu-select-lowering-strategy, iree-llvmgpu-lower-executable-target)))' \
// RUN:     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
// RUN:     --iree-codegen-use-transform-dialect-strategy=%s | \
// RUN: FileCheck --check-prefix=CHECK %s

hal.executable @_attention_dispatch_0 {
  hal.executable.variant public @cuda_nvptx_fb target(<"cuda", "cuda-nvptx-fb", {target_arch = "sm_60"}>) {
    hal.executable.export public @_attention_dispatch_0 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) {
    ^bb0(%arg0: !hal.device, %arg1: index, %arg2: index):
      %x, %y, %z = flow.dispatch.workgroup_count_from_dag_root %arg1, %arg2
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @_attention_dispatch_0() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<192x1024x64xf16>> -> tensor<192x1024x64xf16>
        %7 = tensor.empty() : tensor<192x1024x64xf16>
        %8 = iree_linalg_ext.attention ins(%4, %5, %6 : tensor<192x1024x64xf16>, tensor<192x1024x64xf16>, tensor<192x1024x64xf16>) outs(%7 : tensor<192x1024x64xf16>) -> tensor<192x1024x64xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [192, 1024, 64], strides = [1, 1, 1] : tensor<192x1024x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<192x1024x64xf16>>
        return
      }
    }
  }
}

transform.sequence failures(propagate) {
  ^bb0(%variant_op: !transform.any_op):

  // Get attention op
  // ==========================================
  %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // Tile and distribute to workgroups
  // ==========================================
  %tiled_attention, %forall_grid =
  transform.structured.tile_using_forall %attention tile_sizes [1, 128]
    ( mapping = [#gpu.block<x>, #gpu.block<y>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_grid : (!transform.any_op) -> ()

  // Tile batch dimensions of attention
  // ==========================================
  %attention2 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %batch_tiled_attn, %loop = transform.structured.tile_using_for %attention2 [1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %top_level_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %top_level_func {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_cse %top_level_func : !transform.any_op

  // Promote query and output operands
  // ==========================================
  %attention3 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %promoted_attention, %alloc_a0, %alloc_a1 = transform.iree.promote_operands %attention3 [0, 3]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

  // Tile and decompose attention
  // ==========================================
  %attention4 = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  %acc_fill, %max_fill, %sum_fill, %inner_loop,
  %fill_op, %first_matmul, %reduce_max, %partial_softmax, %update, %reduce_sum, %reciprocal_sum, %softmax, %truncate, %scale_acc, %second_matmul, %last_truncate
      = tile_and_decompose_attention %attention4 :
     (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

  // Promote key and value operands
  // ==========================================
  %promoted_first_matmul, %alloc0 = transform.iree.promote_operands %first_matmul [1]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %promoted_second_matmul, %alloc1 = transform.iree.promote_operands %second_matmul [1]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Tile and fuse attention ops
  // ==========================================
  %tiled_matmul, %forall = transform.structured.tile_using_forall %promoted_second_matmul tile_sizes [32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %f0, %loop0 = transform.structured.fuse_into_containing_op %scale_acc into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %f1, %loop1 = transform.structured.fuse_into_containing_op %truncate into %loop0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %f2, %loop2 = transform.structured.fuse_into_containing_op %softmax into %loop1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_cse %func : !transform.any_op

  %f3, %loop3 = transform.structured.fuse_into_containing_op %reciprocal_sum into %loop2 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %f4, %loop4 = transform.structured.fuse_into_containing_op %reduce_sum into %loop3 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.apply_cse %func : !transform.any_op

  %f5, %loop5 = transform.structured.fuse_into_containing_op %update into %loop4 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %f6, %loop6 = transform.structured.fuse_into_containing_op %partial_softmax into %loop5 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.apply_cse %func : !transform.any_op

  %f7, %loop7 = transform.structured.fuse_into_containing_op %reduce_max into %loop6 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %f8, %loop8 = transform.structured.fuse_into_containing_op %promoted_first_matmul into %loop7 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %f9, %loop9 = transform.structured.fuse_into_containing_op %fill_op into %loop8 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.apply_cse %func : !transform.any_op

  // Distribute fills and last truncate
  // ==========================================
  %fills = transform.merge_handles %acc_fill, %max_fill, %sum_fill, %last_truncate : !transform.any_op
  %tiled_fill, %fill_grid = transform.structured.tile_using_forall %fills tile_sizes[32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Vectorize function
  // ==========================================
  transform.apply_patterns to %func {
    transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
    transform.apply_patterns.linalg.fold_unit_extent_dims_via_slices
    transform.apply_patterns.vector.cast_away_vector_leading_one_dim
  } : !transform.any_op
  %func_3 = transform.structured.vectorize_children_and_apply_patterns %func : (!transform.any_op) -> (!transform.any_op)

  // Bufferization
  // ==========================================
  transform.apply_patterns to %func_3 {
     transform.apply_patterns.tensor.reassociative_reshape_folding
     transform.apply_patterns.canonicalization
     transform.apply_patterns.iree.fold_fill_into_pad
     transform.apply_patterns.linalg.tiling_canonicalization
     transform.apply_patterns.scf.for_loop_canonicalization
  } : !transform.any_op
  transform.iree.apply_cse %func_3 : !transform.any_op
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op : (!transform.any_op) -> (!transform.any_op)

  // Step 5. Pre-process the contract and transfer ops to put it in the right form.
  // ===========================================================================
  %func_2 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func_2 {
    transform.apply_patterns.iree.prepare_vector_to_mma
  } : !transform.any_op

  // Step 6. Post-bufferization vector distribution
  // ===========================================================================
  %func_7 = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
  transform.iree.forall_to_workgroup %func_7 : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_7 workgroup_dims = [4, 8, 4] subgroup_size = 32 : (!transform.any_op) -> ()

  transform.apply_patterns to %func_7 {
     transform.apply_patterns.memref.fold_memref_alias_ops
  } : !transform.any_op
  transform.iree.apply_licm %func_7 : !transform.any_op
  transform.apply_patterns to %func_7 {
     transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_cse %func_7 : !transform.any_op
  %func_8 = transform.structured.hoist_redundant_vector_transfers %func_7
  : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %func_8 {
    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.iree.apply_cse %func_8 : !transform.any_op
  transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()
}

// CHECK-DAG:  #[[MAP:.+]] = affine_map<()[s0] -> (s0 * 128)>
// CHECK-DAG:  #[[MAP1:.+]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-DAG:  #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s2 * 32 + ((s0 + s1 * 4) floordiv 32) * 32)>
// CHECK-DAG:  #[[MAP3:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:  #[[MAP4:.+]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG:  #[[MAP5:.+]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG:  #[[MAP6:.+]] = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK:      func.func @_attention_dispatch_0() {
// CHECK-DAG:    %[[CST:.+]] = arith.constant dense<0.000000e+00> : vector<32x64xf32>
// CHECK-DAG:    %[[CST_0:.+]] = arith.constant dense<-1.000000e+30> : vector<32xf32>
// CHECK-DAG:    %[[CST_1:.+]] = arith.constant dense<0.000000e+00> : vector<32xf32>
// CHECK-DAG:    %[[CST_2:.+]] = arith.constant dense<0.000000e+00> : vector<32x128xf32>
// CHECK-DAG:    %[[CST_3:.+]] = arith.constant dense<1.000000e+00> : vector<32xf32>
// CHECK-DAG:    %[[CST_4:.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:    %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:    %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:    %[[C1024:.+]] = arith.constant 1024 : index
// CHECK-DAG:    %[[CST_5:.+]] = arith.constant 0.000000e+00 : f32
// CHECK:        %[[D0:.+]] = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D0]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[D1:.+]] = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D1]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[D2:.+]] = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) flags(ReadOnly) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D2]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[D3:.+]] = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64)
// CHECK-SAME:     offset(%[[C0]]) : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        memref.assume_alignment %[[D3]], 64 : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[WORKGROUP_ID_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:        %[[WORKGROUP_ID_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:    %[[D4:.+]] = affine.apply #[[MAP]]()[%[[WORKGROUP_ID_Y]]]
// CHECK:        %[[SUBVIEW:.+]] = memref.subview %[[D0]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:     memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[SUBVIEW_6:.+]] = memref.subview %[[D3]][%[[WORKGROUP_ID_X]], %[[D4]], 0] [1, 128, 64] [1, 1, 1] :
// CHECK-SAME:     memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:        %[[ALLOC:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU:.+]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[SUBVIEW]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
// CHECK-SAME:     outs(%[[ALLOC]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
// CHECK:        %[[ALLOC_7:.+]] = memref.alloc() {alignment = 64 : i64} : memref<1x128x64xf16,
// CHECK-SAME:     #[[GPU]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[SUBVIEW_6]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
// CHECK-SAME:     outs(%[[ALLOC_7]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
// CHECK-DAG:    %[[D5:.+]] = gpu.thread_id  x
// CHECK-DAG:    %[[D6:.+]] = gpu.thread_id  y
// CHECK-DAG:    %[[D7:.+]] = gpu.thread_id  z
// CHECK-DAG:    %[[D8:.+]] = affine.apply #[[MAP2]]()[%[[D5]], %[[D6]], %[[D7]]]
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK:        %[[D9:.+]] = vector.transfer_read %[[ALLOC]][%[[C0]], %[[D8]], %[[C0]]], %[[CST_4]] {in_bounds = [true,
// CHECK-SAME:     true]} : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>, vector<32x64xf16>
// CHECK:        %[[D10:.+]] = arith.extf %[[D9]] : vector<32x64xf16> to vector<32x64xf32>
// CHECK:        %[[D11:.+]]:3 = scf.for %[[ARG0:[a-zA-Z0-9_]+]] = %[[C0]] to %[[C1024]] step %[[C128]]
// CHECK-SAME:     iter_args(%[[ARG1:[a-zA-Z0-9_]+]] = %[[CST_0]], %[[ARG2:[a-zA-Z0-9_]+]] = %[[CST_1]],
// CHECK-SAME:     %[[ARG3:[a-zA-Z0-9_]+]] = %[[CST]]) -> (vector<32xf32>, vector<32xf32>, vector<32x64xf32>) {
// CHECK:          %[[SUBVIEW_8:.+]] = memref.subview %[[D1]][%[[WORKGROUP_ID_X]], %[[ARG0]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:       : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:          %[[SUBVIEW_9:.+]] = memref.subview %[[D2]][%[[WORKGROUP_ID_X]], %[[ARG0]], 0] [1, 128, 64] [1, 1, 1]
// CHECK-SAME:       : memref<192x1024x64xf16, #hal.descriptor_type<storage_buffer>> to memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
// CHECK:          %[[ALLOC_10:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16,
// CHECK-SAME:       #[[GPU]].address_space<workgroup>>
// CHECK:          gpu.barrier
// CHECK:          linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:       ins(%[[SUBVIEW_8]] : memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%[[ALLOC_10]] :
// CHECK-SAME:       memref<128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:          ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:            linalg.yield %[[IN]] : f16
// CHECK:          }
// CHECK:          gpu.barrier
// CHECK:          %[[ALLOC_11:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128x64xf16,
// CHECK-SAME:       #[[GPU]].address_space<workgroup>>
// CHECK:          gpu.barrier
// CHECK:          linalg.generic {indexing_maps = [#[[MAP3]], #[[MAP3]]], iterator_types = ["parallel", "parallel"]}
// CHECK-SAME:       ins(%[[SUBVIEW_9]] : memref<128x64xf16, strided<[64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%[[ALLOC_11]] :
// CHECK-SAME:       memref<128x64xf16, #[[GPU]].address_space<workgroup>>) {
// CHECK:          ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:            linalg.yield %[[IN]] : f16
// CHECK:          }
// CHECK:          gpu.barrier
// CHECK:          %[[D13:.+]] = vector.transfer_read %[[ALLOC_10]][%[[C0]], %[[C0]]], %[[CST_4]] {in_bounds = [true,
// CHECK-SAME:       true]} : memref<128x64xf16, #[[GPU]].address_space<workgroup>>, vector<128x64xf16>
// CHECK:          %[[D14:.+]] = arith.extf %[[D13]] : vector<128x64xf16> to vector<128x64xf32>
// CHECK:          %[[D15:.+]] = vector.contract {indexing_maps = [#[[MAP4]], #[[MAP5]], #[[MAP6]]], iterator_types =
// CHECK-SAME:       ["parallel", "parallel", "reduction"], kind = #[[VECTOR:.+]].kind<add>} %[[D10]], %[[D14]],
// CHECK-SAME:       %[[CST_2]] : vector<32x64xf32>, vector<128x64xf32> into vector<32x128xf32>
// CHECK:          %[[D16:.+]] = vector.multi_reduction <maximumf>, %[[D15]], %[[ARG1]] [1] : vector<32x128xf32> to
// CHECK-SAME:       vector<32xf32>
// CHECK:          %[[D17:.+]] = vector.broadcast %[[D16]] : vector<32xf32> to vector<128x32xf32>
// CHECK:          %[[D18:.+]] = vector.transpose %[[D17]], [1, 0] : vector<128x32xf32> to vector<32x128xf32>
// CHECK:          %[[D19:.+]] = arith.subf %[[D15]], %[[D18]] : vector<32x128xf32>
// CHECK:          %[[D20:.+]] = math.exp2 %[[D19]] : vector<32x128xf32>
// CHECK:          %[[D21:.+]] = arith.subf %[[ARG1]], %[[D16]] : vector<32xf32>
// CHECK:          %[[D22:.+]] = math.exp2 %[[D21]] : vector<32xf32>
// CHECK:          %[[D23:.+]] = arith.mulf %[[D22]], %[[ARG2]] : vector<32xf32>
// CHECK:          %[[D24:.+]] = vector.multi_reduction <add>, %[[D20]], %[[D23]] [1] : vector<32x128xf32> to
// CHECK-SAME:       vector<32xf32>
// CHECK:          %[[D25:.+]] = arith.divf %[[CST_3]], %[[D24]] : vector<32xf32>
// CHECK:          %[[D26:.+]] = vector.broadcast %[[D25]] : vector<32xf32> to vector<128x32xf32>
// CHECK:          %[[D27:.+]] = vector.transpose %[[D26]], [1, 0] : vector<128x32xf32> to vector<32x128xf32>
// CHECK:          %[[D28:.+]] = arith.mulf %[[D20]], %[[D27]] : vector<32x128xf32>
// CHECK:          %[[D29:.+]] = arith.truncf %[[D28]] : vector<32x128xf32> to vector<32x128xf16>
// CHECK:          %[[D30:.+]] = arith.mulf %[[D23]], %[[D25]] : vector<32xf32>
// CHECK:          %[[D31:.+]] = vector.broadcast %[[D30]] : vector<32xf32> to vector<64x32xf32>
// CHECK:          %[[D33:.+]] = vector.transpose %[[D31]], [1, 0] : vector<64x32xf32> to vector<32x64xf32>
// CHECK:          %[[D34:.+]] = arith.mulf %[[D33]], %[[ARG3]] : vector<32x64xf32>
// CHECK:          %[[D35:.+]] = vector.transfer_read %[[ALLOC_11]][%[[C0]], %[[C0]]], %[[CST_4]] {in_bounds = [true,
// CHECK-SAME:       true]} : memref<128x64xf16, #[[GPU]].address_space<workgroup>>, vector<128x64xf16>
// CHECK:          %[[D36:.+]] = arith.extf %[[D29]] : vector<32x128xf16> to vector<32x128xf32>
// CHECK:          %[[D37:.+]] = arith.extf %[[D35]] : vector<128x64xf16> to vector<128x64xf32>
// CHECK:          %[[D38:.+]] = vector.transpose %[[D37]], [1, 0] : vector<128x64xf32> to vector<64x128xf32>
// CHECK:          %[[D39:.+]] = vector.contract {indexing_maps = [#[[MAP4]], #[[MAP5]], #[[MAP6]]], iterator_types =
// CHECK-SAME:       ["parallel", "parallel", "reduction"], kind = #[[VECTOR]].kind<add>} %[[D36]], %[[D38]], %[[D34]] :
// CHECK-SAME:       vector<32x128xf32>, vector<64x128xf32> into vector<32x64xf32>
// CHECK:          gpu.barrier
// CHECK:          scf.yield %[[D16]], %[[D24]], %[[D39]] : vector<32xf32>, vector<32xf32>, vector<32x64xf32>
// CHECK:        }
// CHECK:        %[[D12:.+]] = arith.truncf %[[D11]]#[[D2:.+]] : vector<32x64xf32> to vector<32x64xf16>
// CHECK:        vector.transfer_write %[[D12]], %[[ALLOC_7]][%[[C0]], %[[D8]], %[[C0]]] {in_bounds = [true, true]} :
// CHECK-SAME:     vector<32x64xf16>, memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>
// CHECK:        gpu.barrier
// CHECK:        gpu.barrier
// CHECK:        linalg.generic {indexing_maps = [#[[MAP1]], #[[MAP1]]], iterator_types = ["parallel", "parallel",
// CHECK-SAME:     "parallel"]} ins(%[[ALLOC_7]] : memref<1x128x64xf16, #[[GPU]].address_space<workgroup>>)
// CHECK-SAME:     outs(%[[SUBVIEW_6]] : memref<1x128x64xf16, strided<[65536, 64, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) {
// CHECK:        ^bb0(%[[IN:.+]]: f16, %[[OUT:.+]]: f16):
// CHECK:          linalg.yield %[[IN]] : f16
// CHECK:        }
// CHECK:        gpu.barrier
