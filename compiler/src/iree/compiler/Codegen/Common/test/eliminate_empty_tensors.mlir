// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-eliminate-empty-tensors)" %s | FileCheck %s

func.func @eliminate_empty_tensors_with_store_op() {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %0 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
  %1 = tensor.empty() : tensor<32x384xf32>
  scf.for %arg0 = %c0 to %c128 step %c32 {
    %2 = scf.for %arg1 = %c0 to %c32 step %c8 iter_args(%arg2 = %1) -> (tensor<32x384xf32>) {
      scf.yield %arg2 : tensor<32x384xf32>
    }
    flow.dispatch.tensor.store %2, %0, offsets = [%arg0, 0], sizes = [32, 384], strides = [1, 1] : tensor<32x384xf32> -> !flow.dispatch.tensor<writeonly:tensor<128x384xf32>>
  }
  return
}

// CHECK-LABEL: @eliminate_empty_tensors_with_store_op
// CHECK: %[[C0:.+]] = arith.constant 0 : index
// CHECK: %[[C8:.+]] = arith.constant 8 : index
// CHECK: %[[C32:.+]] = arith.constant 32 : index
// CHECK: %[[C128:.+]] = arith.constant 128 : index
// CHECK: %[[SPAN:.+]] = hal.interface.binding.subspan
// CHECK: scf.for %[[ARG0:.+]] = %[[C0]] to %[[C128]] step %[[C32]]
// CHECK:   %[[LOAD:.+]] = flow.dispatch.tensor.load %[[SPAN]], offsets = [%[[ARG0]], 0]
// CHECK:   %[[RES:.+]] = scf.for %{{.+}} = %[[C0]] to %[[C32]] step %[[C8]] iter_args(%{{.+}} = %[[LOAD]])
// CHECK:   flow.dispatch.tensor.store %[[RES]], %[[SPAN]]

// -----

func.func @pad_only_dispatch() {
  %cst = arith.constant dense<0.000000e+00> : vector<4xf32>
  %c64 = arith.constant 64 : index
  %c19 = arith.constant 19 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c634816 = arith.constant 634816 : index
  %c3846080 = arith.constant 3846080 : index
  %cst_0 = arith.constant 0.000000e+00 : f32
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c634816) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<1x112x112x64xf32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c3846080) : !flow.dispatch.tensor<writeonly:tensor<1x114x114x64xf32>>
  %workgroup_id_x = hal.interface.workgroup.id[0] : index
  %2 = affine.min affine_map<()[s0] -> ((s0 floordiv 6) floordiv 6, 1)>()[%workgroup_id_x]
  %3 = affine.min affine_map<()[s0] -> ((s0 floordiv 6) floordiv 6 + 1, 1)>()[%workgroup_id_x]
  %4 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%3, %2]
  %5 = arith.cmpi eq, %4, %c0 : index
  %6 = affine.max affine_map<()[s0] -> (0, (s0 floordiv 6) * -19 + ((s0 floordiv 6) floordiv 6) * 114 + 1)>()[%workgroup_id_x]
  %7 = affine.max affine_map<()[s0] -> ((s0 floordiv 6) * 19 - ((s0 floordiv 6) floordiv 6) * 114 - 1, 0)>()[%workgroup_id_x]
  %8 = affine.min affine_map<()[s0] -> (112, s0)>()[%7]
  %9 = affine.max affine_map<()[s0] -> ((s0 floordiv 6) * 19 - ((s0 floordiv 6) floordiv 6) * 114 + 18, 0)>()[%workgroup_id_x]
  %10 = affine.min affine_map<()[s0] -> (112, s0)>()[%9]
  %11 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%10, %8]
  %12 = arith.cmpi eq, %11, %c0 : index
  %13 = arith.ori %12, %5 : i1
  %14 = affine.max affine_map<()[s0] -> (0, s0 * -19 + (s0 floordiv 6) * 114 + 1)>()[%workgroup_id_x]
  %15 = affine.max affine_map<()[s0] -> (s0 * 19 - (s0 floordiv 6) * 114 - 1, 0)>()[%workgroup_id_x]
  %16 = affine.min affine_map<()[s0] -> (112, s0)>()[%15]
  %17 = affine.max affine_map<()[s0] -> (s0 * 19 - (s0 floordiv 6) * 114 + 18, 0)>()[%workgroup_id_x]
  %18 = affine.min affine_map<()[s0] -> (112, s0)>()[%17]
  %19 = affine.apply affine_map<()[s0, s1] -> (s0 - s1)>()[%18, %16]
  %20 = arith.cmpi eq, %19, %c0 : index
  %21 = arith.ori %20, %13 : i1
  scf.if %21 {
    %generated = tensor.generate  {
    ^bb0(%arg0: index, %arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_0 : f32
    } : tensor<1x19x19x64xf32>
    %22 = affine.apply affine_map<()[s0] -> ((s0 floordiv 6) floordiv 6)>()[%workgroup_id_x]
    %23 = affine.apply affine_map<()[s0] -> ((s0 floordiv 6) * 19 - ((s0 floordiv 6) floordiv 6) * 114)>()[%workgroup_id_x]
    %24 = affine.apply affine_map<()[s0] -> (s0 * 19 - (s0 floordiv 6) * 114)>()[%workgroup_id_x]
    flow.dispatch.tensor.store %generated, %1, offsets = [%22, %23, %24, 0], sizes = [1, 19, 19, 64], strides = [1, 1, 1, 1] : tensor<1x19x19x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x114x114x64xf32>>
  } else {
    %22 = flow.dispatch.tensor.load %0, offsets = [%2, %8, %16, 0], sizes = [%4, %11, %19, 64], strides = [1, 1, 1, 1] : !flow.dispatch.tensor<readonly:tensor<1x112x112x64xf32>> -> tensor<?x?x?x64xf32>
    %23 = tensor.empty() : tensor<1x19x19x64xf32>
    %24 = affine.min affine_map<()[s0, s1] -> (0, s0 - s1)>()[%3, %2]
    %25 = tensor.empty() : tensor<1x1x1x4xf32>
    %26 = scf.for %arg0 = %c0 to %c19 step %c1 iter_args(%arg1 = %23) -> (tensor<1x19x19x64xf32>) {
      %30 = affine.max affine_map<(d0)[s0] -> (-d0 + s0, 0)>(%arg0)[%6]
      %31 = affine.max affine_map<(d0)[s0] -> (0, d0 - s0)>(%arg0)[%6]
      %32 = affine.min affine_map<(d0)[s0, s1] -> (s0 - s1, d0)>(%31)[%10, %8]
      %33 = affine.max affine_map<(d0)[s0] -> (0, d0 - s0 + 1)>(%arg0)[%6]
      %34 = affine.min affine_map<(d0)[s0, s1] -> (s0 - s1, d0)>(%33)[%10, %8]
      %35 = affine.apply affine_map<(d0, d1)[s0] -> (d0 - d1 + s0)>(%34, %32)[%30]
      %36 = arith.cmpi sle, %30, %c0 : index
      %37 = arith.cmpi sgt, %35, %c0 : index
      %38 = arith.andi %36, %37 : i1
      %39 = affine.apply affine_map<()[s0] -> (-s0)>()[%30]
      %40 = scf.for %arg2 = %c0 to %c19 step %c1 iter_args(%arg3 = %arg1) -> (tensor<1x19x19x64xf32>) {
        %41 = affine.max affine_map<(d0)[s0] -> (-d0 + s0, 0)>(%arg2)[%14]
        %42 = affine.max affine_map<(d0)[s0] -> (0, d0 - s0)>(%arg2)[%14]
        %43 = affine.min affine_map<(d0)[s0, s1] -> (s0 - s1, d0)>(%42)[%18, %16]
        %44 = affine.max affine_map<(d0)[s0] -> (0, d0 - s0 + 1)>(%arg2)[%14]
        %45 = affine.min affine_map<(d0)[s0, s1] -> (s0 - s1, d0)>(%44)[%18, %16]
        %46 = affine.apply affine_map<(d0, d1)[s0] -> (d0 - d1 + s0)>(%45, %43)[%41]
        %47 = arith.cmpi sle, %41, %c0 : index
        %48 = arith.cmpi sgt, %46, %c0 : index
        %49 = arith.andi %47, %48 : i1
        %50 = arith.andi %38, %49 : i1
        %51 = affine.apply affine_map<()[s0] -> (-s0)>()[%41]
        %52 = scf.for %arg4 = %c0 to %c64 step %c4 iter_args(%arg5 = %arg3) -> (tensor<1x19x19x64xf32>) {
          %53 = scf.if %50 -> (vector<4xf32>) {
            %55 = arith.addi %39, %32 : index
            %56 = arith.addi %51, %43 : index
            %57 = vector.transfer_read %22[%24, %55, %56, %arg4], %cst_0 {in_bounds = [true]} : tensor<?x?x?x64xf32>, vector<4xf32>
            scf.yield %57 : vector<4xf32>
          } else {
            scf.yield %cst : vector<4xf32>
          }
          %54 = vector.transfer_write %53, %25[%c0, %c0, %c0, %c0] {in_bounds = [true]} : vector<4xf32>, tensor<1x1x1x4xf32>
          %inserted_slice = tensor.insert_slice %54 into %arg5[0, %arg0, %arg2, %arg4] [1, 1, 1, 4] [1, 1, 1, 1] : tensor<1x1x1x4xf32> into tensor<1x19x19x64xf32>
          scf.yield %inserted_slice : tensor<1x19x19x64xf32>
        }
        scf.yield %52 : tensor<1x19x19x64xf32>
      }
      scf.yield %40 : tensor<1x19x19x64xf32>
    }
    %27 = affine.apply affine_map<()[s0] -> ((s0 floordiv 6) floordiv 6)>()[%workgroup_id_x]
    %28 = affine.apply affine_map<()[s0] -> ((s0 floordiv 6) * 19 - ((s0 floordiv 6) floordiv 6) * 114)>()[%workgroup_id_x]
    %29 = affine.apply affine_map<()[s0] -> (s0 * 19 - (s0 floordiv 6) * 114)>()[%workgroup_id_x]
    flow.dispatch.tensor.store %26, %1, offsets = [%27, %28, %29, 0], sizes = [1, 19, 19, 64], strides = [1, 1, 1, 1] : tensor<1x19x19x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<1x114x114x64xf32>>
  }
  return
}
// CHECK-LABEL: func @pad_only_dispatch()
// CHECK:         %[[SRC:.+]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK:         %[[DEST:.+]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK:         scf.if
// CHECK:           %[[OFFSET0:.+]] = affine.apply
// CHECK:           %[[OFFSET1:.+]] = affine.apply
// CHECK:           %[[OFFSET2:.+]] = affine.apply
// CHECK:           %[[SUB_DEST:.+]] = flow.dispatch.tensor.load %[[DEST]]
// CHECK:           %{{.+}} = linalg.generic
// CHECK-SAME:        outs(%[[SUB_DEST]]
// CHECK:         } else {
// CHECK:           %[[OFFSET0:.+]] = affine.apply
// CHECK:           %[[OFFSET1:.+]] = affine.apply
// CHECK:           %[[OFFSET2:.+]] = affine.apply
// CHECK:           %[[SUB_DEST:.+]] = flow.dispatch.tensor.load %[[DEST]]
// CHECK:           %{{.+}} = scf.for {{.+}} iter_args(%{{.+}} = %[[SUB_DEST]])
