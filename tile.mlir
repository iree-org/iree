#intrinsic = #iree_gpu.mma_layout<MFMA_F16_32x32x8_F32>

hal.executable private @attention_dispatch_0 {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree.gpu.target = #iree_gpu.target<arch = "gfx908", features = "", wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8, storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic, dot =  dp4xi8toi32, mma = [<MFMA_F16_16x16x16_F32>, <MFMA_F16_32x32x8_F32>], subgroup_size_choices = [64], max_workgroup_sizes = [1024, 1024, 1024], max_thread_count_per_workgroup = 1024, max_workgroup_memory_bytes = 65536>>, ukernels = "none"}>) {
    hal.executable.export public @attention_dispatch_0_attention_20x4096x64xf16 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer, ReadOnly>, <3, storage_buffer>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>, #hal.interface.binding<0, 1>, #hal.interface.binding<0, 2>, #hal.interface.binding<0, 3>]} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @attention_dispatch_0_attention_20x4096x64xf16() {
        %cst = arith.constant 1.000000e+00 : f16
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>>
        %3 = hal.interface.binding.subspan set(0) binding(3) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        %4 = flow.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %5 = flow.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %6 = flow.dispatch.tensor.load %2, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<20x4096x64xf16>> -> tensor<20x4096x64xf16>
        %7 = tensor.empty() : tensor<20x4096x64xf16>
        %8 = iree_linalg_ext.attention ins(%4, %5, %6, %cst : tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, tensor<20x4096x64xf16>, f16) outs(%7 : tensor<20x4096x64xf16>) -> tensor<20x4096x64xf16>
        flow.dispatch.tensor.store %8, %3, offsets = [0, 0, 0], sizes = [20, 4096, 64], strides = [1, 1, 1] : tensor<20x4096x64xf16> -> !flow.dispatch.tensor<writeonly:tensor<20x4096x64xf16>>
        return
      }
    }
  }
}

module attributes { transform.with_named_sequence } {

  // Utility matching for finding all undistributed fills.
  transform.named_sequence @matcher(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %arg0 ["linalg.fill"] : !transform.any_op
    %0 = transform.get_parent_op %arg0 {allow_empty_results, nth_parent = 2 : i64, op_name = "scf.forall"} : (!transform.any_op) -> !transform.any_op
    transform.match.operation_empty %0 : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }

  transform.named_sequence @get_undistributed_fills(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
    %0 = transform.collect_matching @matcher in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.yield %0 : !transform.any_op
  }

  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
    // Get attention op
    // ==========================================
    %attention = transform.structured.match ops{["iree_linalg_ext.attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // Tile and distribute to workgroups
    // ==========================================
    %blocked_att, %wg_forall =
    transform.structured.tile_using_forall %attention tile_sizes [1, 128, 0, 0, 0]
      ( mapping = [#gpu.block<linear_dim_0>, #gpu.block<linear_dim_1>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Cleanup
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Convert to online attention
    // ==========================================
    transform.iree.convert_to_online_attention %blocked_att : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Tile along K2
    %online_att = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %tiled_att, %k2_for = transform.structured.tile_using_for %online_att tile_sizes [0, 0, 0, 64, 0]: (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Promote key and value operands
    // ==========================================
    %promoted_att, %alloc0, %alloc1 = transform.iree.promote_operands %tiled_att [1, 2]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // This is a hack... We distribute the loop and the merging seperately and just assume they are fused.
    %warp_attention = transform.structured.match ops{["iree_linalg_ext.online_attention"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %warp_merge = transform.structured.match ops{["linalg.generic"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %warp_attention tile_sizes[0, 32, 0, 0, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.structured.tile_using_forall %warp_merge tile_sizes[0, 32, 0] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    %fills = transform.include @get_undistributed_fills failures(propagate) (%variant_op)  : (!transform.any_op) -> !transform.any_op
    transform.structured.tile_using_forall %fills tile_sizes[0, 32] (mapping = [#gpu.warp<linear_dim_0>]) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.apply_patterns to %func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func : !transform.any_op

    // Decompose attention
    %func2 = transform.apply_registered_pass "iree-linalg-ext-decompose-attention" to %func : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func2 : !transform.any_op

    // Vectorize function
    // ==========================================
    transform.apply_patterns to %func2 {
      transform.apply_patterns.iree.fold_reshape_into_tensor_hal_interface
      transform.apply_patterns.vector.cast_away_vector_leading_one_dim
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op
    %func_3 = transform.structured.vectorize_children_and_apply_patterns %func2 : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %func_3 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op

    // Bufferization
    // ==========================================
    transform.apply_patterns to %func_3 {
      transform.apply_patterns.tensor.reassociative_reshape_folding
      transform.apply_patterns.canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.scf.for_loop_canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_3 : !transform.any_op
    transform.iree.eliminate_empty_tensors %func_3 : (!transform.any_op) -> ()
    transform.apply_patterns to %func_3 { transform.apply_patterns.linalg.erase_unnecessary_inputs } : !transform.any_op
    %memref_func = transform.iree.bufferize { target_gpu } %func_3 : (!transform.any_op) -> (!transform.any_op)

    // Step 5. Pre-process the contract and transfer ops to put it in the right form.
    // ===========================================================================
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.vector.fold_arith_extension
    } : !transform.any_op

    // Step 6. Post-bufferization vector distribution
    // ===========================================================================
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [64, 4, 1] subgroup_size = 64 sync_after_distribution = false : (!transform.any_op) -> ()

    transform.apply_patterns to %memref_func {
      transform.apply_patterns.memref.fold_memref_alias_ops
    } : !transform.any_op
    transform.iree.apply_licm %memref_func : !transform.any_op
    transform.apply_patterns to %memref_func {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %memref_func : !transform.any_op
    %func_8 = transform.structured.hoist_redundant_vector_transfers %memref_func
    : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %func_8 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %func_8 : !transform.any_op
    transform.memref.erase_dead_alloc_and_stores %func_8 : (!transform.any_op) -> ()

    // Apply chained matmul optimization.
    %func_9 = transform.apply_registered_pass "iree-amdgpu-prepare-chained-matmul" to %func_8 : (!transform.any_op) -> (!transform.any_op)

    %intrinsic = transform.param.constant #intrinsic -> !transform.any_param

    %mma = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %mma1, %mma2 = transform.split_handle %mma : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.annotate %mma "iree.amdgpu.mma" = %intrinsic : !transform.any_op, !transform.any_param

    transform.apply_registered_pass "iree-llvmgpu-cast-type-to-fit-mma" to %func_9 : (!transform.any_op) -> (!transform.any_op)

    %contracts = transform.structured.match ops{["vector.contract"]} in %variant_op :  (!transform.any_op) -> !transform.any_op
    %contract1, %contract2 = transform.split_handle %contracts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.set_contraction_layout_attributes %contract1, %intrinsic { read_layout_indices = array<i64: 0, 1> } : !transform.any_op, !transform.any_param
    transform.iree.set_contraction_layout_attributes %contract2, %intrinsic : !transform.any_op, !transform.any_param

    %distribute_func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %distribute_func_2 = transform.iree.amdgpu_distribute_vectors %distribute_func : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %distribute_func_2 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    // Distribute shared memory copies
    // ==========================================
    transform.iree.gpu_distribute_shared_memory_copy %distribute_func_2 : (!transform.any_op) -> ()
    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    %forop = transform.structured.match ops{["scf.for"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    %prefetched_forop = transform.iree.prefetch_shared_memory_copies %forop : (!transform.any_op) -> (!transform.any_op)

    transform.apply_patterns to %distribute_func_2 {
        transform.apply_patterns.memref.fold_memref_alias_ops
        transform.apply_patterns.canonicalization
        transform.apply_patterns.linalg.tiling_canonicalization
      } : !transform.any_op
    transform.apply_cse to %distribute_func_2 : !transform.any_op

    %func_11 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
    transform.iree.reduce_shared_memory_bank_conflicts %func_11 : (!transform.any_op) -> ()

    transform.yield
  }

} ////  module
