// RUN: iree-opt --split-input-file  --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(builtin.module(iree-codegen-lower-to-rocm-gpu))))" %s | FileCheck %s

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer>
]>
#target_info = #iree_gpu.target<arch = "gfx942",
  features = "",
  wgp = <compute =  fp64|fp32|fp16|int64|int32|int16|int8,
    storage =  b64|b32|b16|b8, subgroup =  shuffle|arithmetic,
    dot =  dp4xi8toi32, subgroup_size_choices = [64],
    max_workgroup_sizes = [1024, 1024, 1024],
    max_thread_count_per_workgroup = 1024,
    max_workgroup_memory_bytes = 65536,
    max_workgroup_counts = [2147483647, 2147483647, 2147483647]
  >
>
hal.executable public @main {
  hal.executable.variant public @rocm_hsaco_fb target(<"rocm", "rocm-hsaco-fb", {iree_codegen.target_info = #target_info}>) {
    hal.executable.export public @no_merge_basic_blocks ordinal(0) layout(#pipeline_layout) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    } attributes {subgroup_size = 64 : index, workgroup_size = [128 : index, 1 : index, 1 : index]}
    builtin.module {
      func.func @no_merge_basic_blocks() {
        %c8 = arith.constant 8 : index
        %0 = ub.poison : i8
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c0_i8 = arith.constant 0 : i8
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() : memref<1x1x16x40xi8, #gpu.address_space<workgroup>>
        %alloc_0 = memref.alloc() : memref<1x32x40xi8, #gpu.address_space<workgroup>>
        %thread_id_x = gpu.thread_id  x upper_bound 128
        %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x32x16x16xi8, #hal.descriptor_type<storage_buffer>>
        %3 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x32x32x3x3xi8, #hal.descriptor_type<storage_buffer>>
        %5:2 = affine.delinearize_index %thread_id_x into (16, 8) : index, index
        %6 = affine.min affine_map<()[s0] -> (7, s0)>()[%5#0]
        %7 = affine.min affine_map<()[s0] -> (-s0 + 7, 1)>()[%6]
        %8 = vector.create_mask %c1, %c1, %5#0, %c4 : vector<1x1x1x4xi1>
        %10 = arith.cmpi sgt, %7, %c0 : index
        %11 = vector.extract %8[0, 0, 0] : vector<4xi1> from vector<1x1x1x4xi1>
        %workgroup_id_x = hal.interface.workgroup.id[0] : index
        %12:2 = affine.delinearize_index %workgroup_id_x into (32, 7) : index, index
        %subview = memref.subview %3[%12#0, 0, 0, 0, 0] [1, 32, 32, 3, 3] [1, 1, 1, 1, 1] : memref<32x32x32x3x3xi8, #hal.descriptor_type<storage_buffer>> to memref<1x32x32x3x3xi8, strided<[9216, 288, 9, 3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %collapse_shape = memref.collapse_shape %subview [[0], [1], [2, 3, 4]] : memref<1x32x32x3x3xi8, strided<[9216, 288, 9, 3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> into memref<1x32x288xi8, strided<[9216, 288, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        gpu.barrier memfence [#gpu.address_space<workgroup>]
        %13 = vector.transfer_read %collapse_shape[%c0, %c0, %c0], %0 {in_bounds = [true]} : memref<1x32x288xi8, strided<[9216, 288, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, vector<8xi8>
        %alloca = memref.alloca(%7) : memref<?x1x1x1x1xi8, #gpu.address_space<private>>
        scf.if %10 {
          %15 = vector.transfer_read %1[%c0, %c0, %c0, %c0], %0 {in_bounds = [true]} :  memref<32x32x16x16xi8, #hal.descriptor_type<storage_buffer>>, vector<1xi8>
          vector.transfer_write %15, %alloca[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true]} : vector<1xi8>, memref<?x1x1x1x1xi8, #gpu.address_space<private>>
        }
        vector.transfer_write %13, %alloc_0[%c0, %c0, %c0] {in_bounds = [true]} : vector<8xi8>, memref<1x32x40xi8, #gpu.address_space<workgroup>>
        %alloca_2 = memref.alloca(%7) : memref<1x1x?x4xi8, #gpu.address_space<private>>
        scf.if %10 {
          %15 = vector.transfer_read %alloca[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true]} : memref<?x1x1x1x1xi8, #gpu.address_space<private>>, vector<1xi8>
          vector.transfer_write %15, %alloca_2[%c0, %c0, %c0, %c0] {in_bounds = [true]} : vector<1xi8>, memref<1x1x?x4xi8, #gpu.address_space<private>>
        }
        %14 = vector.transfer_read %alloca_2[%c0, %c0, %c0, %c0], %c0_i8, %11 {in_bounds = [true]} : memref<1x1x?x4xi8, #gpu.address_space<private>>, vector<4xi8>
        vector.transfer_write %14, %alloc[%c0, %c0, %c0, %c0] {in_bounds = [true]} : vector<4xi8>, memref<1x1x16x40xi8, #gpu.address_space<workgroup>>
        scf.for %arg0 = %c0 to %c8 step %c1 {
          %15 = arith.addi %arg0, %c1 : index
          %alloca_3 = memref.alloca(%7) : memref<?x1x1x1x1xi8, #gpu.address_space<private>>
          scf.if %10 {
            %17 = vector.transfer_read %1[%12#0, %15, %c0, %c0], %0 {in_bounds = [true]} : memref<32x32x16x16xi8, #hal.descriptor_type<storage_buffer>>, vector<1xi8>
            vector.transfer_write %17, %alloca_3[%c0, %c0, %c0, %c0, %c0] {in_bounds = [true]} : vector<1xi8>, memref<?x1x1x1x1xi8, #gpu.address_space<private>>
          }
          %alloca_4 = memref.alloca(%7) : memref<1x1x?x4xi8, #gpu.address_space<private>>
          scf.if %10 {
            %17 = vector.transfer_read %alloca_3[%c0, %c0, %c0, %c0, %c0], %0 {in_bounds = [true]} : memref<?x1x1x1x1xi8, #gpu.address_space<private>>, vector<1xi8>
            vector.transfer_write %17, %alloca_4[%c0, %c0, %c0, %c0] {in_bounds = [true]} : vector<1xi8>, memref<1x1x?x4xi8, #gpu.address_space<private>>
          }
          %16 = vector.transfer_read %alloca_4[%c0, %c0, %c0, %c0], %c0_i8, %11 {in_bounds = [true]} : memref<1x1x?x4xi8, #gpu.address_space<private>>, vector<4xi8>
          vector.transfer_write %16, %alloc[%c0, %c0, %5#0, %5#1] {in_bounds = [true]} : vector<4xi8>, memref<1x1x16x40xi8, #gpu.address_space<workgroup>>
        }
        return
      }
    }
  }
}
// The purpose of this test is to make sure that we dont merge basic blocks
// If some pass accidentaly does this then we would have arguments to
// ^bb4 and also have 4 preds instead of 2 due to back edges.
// CHECK-LABEL: func @no_merge_basic_blocks(
//       CHECK:   ^bb4:
//  CHECK-SAME: 2 preds: ^bb2, ^bb3
