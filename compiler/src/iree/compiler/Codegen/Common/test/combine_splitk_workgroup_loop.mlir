// RUN: iree-opt %s --pass-pipeline="builtin.module(func.func(iree-codegen-combine-splitk-workgroup-loop))" \
// RUN:  --allow-unregistered-dialect --mlir-print-local-scope --split-input-file | FileCheck %s

#hal_descriptor_type = #hal.descriptor_type<storage_buffer>

func.func @combine_splitk_reduction() attributes {translation_info = #iree_codegen.translation_info<pipeline = CPUDoubleTilingExpert>} {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %c-1 = arith.constant -1 : index
  %c128 = arith.constant 128 : index
  %c8 = arith.constant 8 : index
  %0 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x131072xf32>>
  %1 = hal.interface.binding.subspan layout(<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<128x1024xf32, #hal_descriptor_type>
  %2 = tensor.empty() : tensor<128x1024xf32>
  %3 = scf.forall (%arg0) = (0) to (131072) step (128) shared_outs(%arg1 = %2) -> (tensor<128x1024xf32>) {
    %4 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, %arg0], sizes = [128, 128], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<128x131072xf32>> -> tensor<128x128xf32>
    %5 = arith.cmpi slt, %arg0, %c0 : index
    %6 = arith.subi %c-1, %arg0 : index
    %7 = arith.select %5, %6, %arg0 : index
    %8 = arith.divsi %7, %c128 : index
    %9 = arith.subi %c-1, %8 : index
    %10 = arith.select %5, %9, %8 : index
    %extracted_slice = tensor.extract_slice %arg1[0, %10] [128, 1] [1, 1] : tensor<128x1024xf32> to tensor<128xf32>
    iree_codegen.workgroup_count_hint(%c8)
    %11 = pcf.loop scope(#iree_codegen.workgroup<linearize>) count(%c8)
      execute(%ref = %extracted_slice)[%id: index]
           : (!pcf.sref<128xf32, #iree_codegen.workgroup<linearize>, #pcf.sync_on_parent>)
          -> (tensor<128xf32>) {
      %12 = affine.apply affine_map<()[s0] -> (s0 * 16)>()[%id]
      %extracted_slice_0 = tensor.extract_slice %4[%12, 0] [16, 128] [1, 1] : tensor<128x128xf32> to tensor<16x128xf32>
      %extracted_slice_1 = tensor.extract_slice %extracted_slice[%12] [16] [1] : tensor<128xf32> to tensor<16xf32>
      %13 = linalg.fill {lowering_config = #iree_cpu.lowering_config<vector_common_parallel = [4]>} ins(%cst : f32) outs(%extracted_slice_1 : tensor<16xf32>) -> tensor<16xf32>
      %14 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%extracted_slice_0 : tensor<16x128xf32>) outs(%13 : tensor<16xf32>) attrs =  {iree_linalg_ext.split_reduction = [128], lowering_config = #iree_cpu.lowering_config<distribution = [16, 0], vector_common_parallel = [4, 0], vector_reduction = [0, 4]>} {
      ^bb0(%in: f32, %out: f32):
        %15 = arith.addf %in, %out : f32
        linalg.yield %15 : f32
      } -> tensor<16xf32>
      pcf.write_slice %14 into %ref[%12] [16] [1] : tensor<16xf32> into !pcf.sref<128xf32, #iree_codegen.workgroup<linearize>, #pcf.sync_on_parent>
      pcf.return
    }
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %11 into %arg1[0, %10] [128, 1] [1, 1] : tensor<128xf32> into tensor<128x1024xf32>
    }
  } {mapping = [#iree_linalg_ext.split_reduction_mapping<0>]}
  iree_codegen.store_to_buffer %3, %1 : tensor<128x1024xf32> into memref<128x1024xf32, #hal_descriptor_type>
  return
}

// CHECK-LABEL: func.func @combine_splitk_reduction
//       CHECK:   %[[EMPTY:.+]] = tensor.empty() : tensor<128x1024xf32>
//       CHECK:   %[[FORALL_COUNT:.+]] = arith.ceildivsi
//       CHECK:   %[[TOTAL_WG_COUNT:.+]] = arith.muli %[[FORALL_COUNT]], %c8
//       CHECK:   iree_codegen.workgroup_count_hint(%[[TOTAL_WG_COUNT]])
//       CHECK:   %[[GENERIC:.+]] = pcf.generic scope(#iree_codegen.workgroup<linearize>)
//  CHECK-NEXT:       execute(%{{.+}} = %[[EMPTY]])[%[[LINEAR_ID:.+]]: index, %[[TOTAL_WORKERS:.+]]: index]
//  CHECK-NEXT:            : (!pcf.sref<128x1024xf32, sync(#iree_codegen.workgroup<linearize>)>)
//       CHECK:     %[[TOTAL_ITERS:.+]] = arith.ceildivsi
//       CHECK:     %[[FORALL_ID:.+]] = arith.divsi %[[LINEAR_ID]], %c8
//       CHECK:     %[[LOOP_ID:.+]] = arith.remsi %[[LINEAR_ID]], %c8
//       CHECK:     %[[OUTER_STEP:.+]] = arith.ceildivsi %[[TOTAL_WORKERS]], %c8
//       CHECK:     scf.for %[[OUTER_IV:.+]] = %[[FORALL_ID]] to %[[TOTAL_ITERS]] step %[[OUTER_STEP]]
//       CHECK:       %[[FORALL_IV_DELIN:.+]] = affine.delinearize_index %[[OUTER_IV]]
//       CHECK:       %[[PRODUCT:.+]] = arith.muli %c8, %[[FORALL_ID]]
//       CHECK:       %[[DIFF:.+]] = arith.subi %[[TOTAL_WORKERS]], %[[PRODUCT]]
//       CHECK:       %[[INNER_STEP:.+]] = arith.minsi %[[DIFF]], %c8
//       CHECK:       scf.for %{{.+}} = %[[LOOP_ID]] to %c8 step %[[INNER_STEP]]
//       CHECK:         linalg.fill
//       CHECK:         linalg.generic
//       CHECK:         pcf.write_slice
//       CHECK:     pcf.return
