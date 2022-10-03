// RUN: iree-opt --split-input-file --pass-pipeline='hal.executable(hal.executable.variant(builtin.module(func.func(iree-spirv-vector-to-cooperative-ops,cse))))' %s | FileCheck %s

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

hal.executable private @matmul_contract  {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
        [Shader, CooperativeMatrixNV, Int8, StorageBuffer8BitAccess],
        [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage]>,
        #spirv.resource_limits<max_compute_workgroup_invocations = 128, max_compute_workgroup_size = [128, 128, 64]>>}> {
    builtin.module {
      // CHECK-LABEL: func.func @matmul_contract
      //  CHECK-SAME: %[[ARG0:.+]]: memref<8x32xi8>, %[[ARG1:.+]]: memref<32x8xi8>, %[[ARG2:.+]]: memref<8x8xi32>
      func.func @matmul_contract(%arg0: memref<8x32xi8>, %arg1: memref<32x8xi8>, %arg2: memref<8x8xi32>) {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0 : i32
        %cst_i8 = arith.constant 0 : i8
        // CHECK: %[[ARG0_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG0]] : memref<8x32xi8> to !spirv.ptr<!spirv.struct<(!spirv.rtarray<i8, stride=1> [0])>, StorageBuffer>
        // CHECK: %[[ARG1_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG1]] : memref<32x8xi8> to !spirv.ptr<!spirv.struct<(!spirv.rtarray<i8, stride=1> [0])>, StorageBuffer>
        // CHECK: %[[ARG2_CAST:.+]] = builtin.unrealized_conversion_cast %[[ARG2]] : memref<8x8xi32> to !spirv.ptr<!spirv.struct<(!spirv.rtarray<i32, stride=4> [0])>, StorageBuffer>
        // CHECK: %[[C32:.+]] = spirv.Constant 32 : i32
        // CHECK: %[[COL_MAJOR:.+]] = spirv.Constant false
        // CHECK: %[[ADDR0:.+]] = spirv.AccessChain %[[ARG0_CAST]]
        // CHECK: %[[A:.+]] = spirv.NV.CooperativeMatrixLoad %[[ADDR0]], %[[C32]], %[[COL_MAJOR]]
        %0 = vector.transfer_read %arg0[%c0, %c0], %cst_i8 : memref<8x32xi8>, vector<8x32xi8>
        // CHECK: %[[C8:.+]] = spirv.Constant 8 : i32
        // CHECK: %[[ADDR1:.+]] = spirv.AccessChain %[[ARG1_CAST]]
        // CHECK: %[[B:.+]] = spirv.NV.CooperativeMatrixLoad %[[ADDR1]], %[[C8]], %[[COL_MAJOR]]
        %1 = vector.transfer_read %arg1[%c0, %c0], %cst_i8 : memref<32x8xi8>, vector<32x8xi8>
        // CHECK: %[[ADDR2:.+]] = spirv.AccessChain %[[ARG2_CAST]]
        // CHECK: %[[C:.+]] = spirv.NV.CooperativeMatrixLoad %[[ADDR2]], %[[C8]], %[[COL_MAJOR]]
        %2 = vector.transfer_read %arg2[%c0, %c0], %cst : memref<8x8xi32>, vector<8x8xi32>
        // CHECK: %[[R:.+]] = spirv.NV.CooperativeMatrixMulAdd %[[A]], %[[B]], %[[C]]
        %3 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %0, %1, %2 : vector<8x32xi8>, vector<32x8xi8> into vector<8x8xi32>
        // CHECK: spirv.NV.CooperativeMatrixStore %[[ADDR2]], %[[R]], %[[C8]], %[[COL_MAJOR]]
        vector.transfer_write %3, %arg2[%c0, %c0] : vector<8x8xi32>, memref<8x8xi32>
        return
      }
    }
  }
}

// -----

#map0 = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

hal.executable private @matmul_contract_licm  {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
          [Shader, CooperativeMatrixNV, Int8, StorageBuffer8BitAccess],
          [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage]>,
          #spirv.resource_limits<max_compute_workgroup_invocations = 128, max_compute_workgroup_size = [128, 128, 64]>>}> {
    builtin.module {
      // CHECK-LABEL: func.func @matmul_contract_licm
      func.func @matmul_contract_licm(%arg0: memref<4096x4096xi8>, %arg1: memref<4096x4096xi8>, %arg2: memref<4096x4096xi32>) {
        %c32 = arith.constant 32 : index
        %c4096 = arith.constant 4096 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %c0_i8 = arith.constant 0 : i8
        // CHECK: %[[C:.+]] = spirv.NV.CooperativeMatrixLoad
        %4 = vector.transfer_read %arg2[%c0, %c0], %c0_i32 {in_bounds = [true, true]} : memref<4096x4096xi32>, vector<16x16xi32>
        // CHECK: %[[INIT:.+]] = builtin.unrealized_conversion_cast %[[C]] : !spirv.coopmatrix<16x16xi32, Subgroup> to vector<16x16xi32>
        // CHECK: %[[LOOP:.+]] = scf.for
        // CHECK-SAME: iter_args(%[[ARG:.+]] = %[[INIT]])
        // CHECK: %[[C1:.+]] = builtin.unrealized_conversion_cast %[[ARG]] : vector<16x16xi32> to !spirv.coopmatrix<16x16xi32, Subgroup>
        %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4) -> (vector<16x16xi32>) {
          // CHECK: %[[A:.+]] = spirv.NV.CooperativeMatrixLoad
          %6 = vector.transfer_read %arg0[%c0, %arg3], %c0_i8 {in_bounds = [true, true]} : memref<4096x4096xi8>, vector<16x32xi8>
          // CHECK: %[[B:.+]] = spirv.NV.CooperativeMatrixLoad
          %7 = vector.transfer_read %arg1[%arg3, %c0], %c0_i8 {in_bounds = [true, true]} : memref<4096x4096xi8>, vector<32x16xi8>
          // CHECK: %[[R:.+]] = spirv.NV.CooperativeMatrixMulAdd %[[A]], %[[B]], %[[C1]]
          %8 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %6, %7, %arg4 : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
          // CHECK: %[[YIELD:.+]] = builtin.unrealized_conversion_cast %[[R]] : !spirv.coopmatrix<16x16xi32, Subgroup> to vector<16x16xi32>
          // CHECK: scf.yield %[[YIELD]]
          scf.yield %8 : vector<16x16xi32>
        }
        // CHECK: %[[ACCv:.+]] = builtin.unrealized_conversion_cast %[[LOOP]] : vector<16x16xi32> to !spirv.coopmatrix<16x16xi32, Subgroup>
        // CHECK: spirv.NV.CooperativeMatrixStore %{{.*}}, %[[ACCv]], %{{.*}}, %{{.*}}
        vector.transfer_write %5, %arg2[%c0, %c0] : vector<16x16xi32>, memref<4096x4096xi32>
        return
      }
    }
  }
}
// -----

#map0 = affine_map<(d0, d1) -> (d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>

hal.executable private @matmul_contract_vector_memref  {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
      spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
        [Shader, CooperativeMatrixNV, Int8, StorageBuffer8BitAccess],
        [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix, SPV_KHR_8bit_storage]>,
        #spirv.resource_limits<max_compute_workgroup_invocations = 128, max_compute_workgroup_size = [128, 128, 64]>>}> {
    builtin.module {
      // CHECK-LABEL: func.func @matmul_contract_vector_memref
      func.func @matmul_contract_vector_memref(%arg0: memref<4096x256xvector<4xi32>>, %arg1: memref<4096x256xvector<4xi32>>, %arg2: memref<4096x1024xvector<4xi32>>) {
        %c32 = arith.constant 32 : index
        %c4096 = arith.constant 4096 : index
        %c0 = arith.constant 0 : index
        %cst = arith.constant dense<0> : vector<4xi32>
        // CHECK: %[[C:.+]] = spirv.NV.CooperativeMatrixLoad
        %4 = vector.transfer_read %arg2[%c0, %c0], %cst : memref<4096x1024xvector<4xi32>>, vector<16x16xi32>
        // CHECK: scf.for
        %5 = scf.for %arg3 = %c0 to %c4096 step %c32 iter_args(%arg4 = %4) -> (vector<16x16xi32>) {
          // CHECK: %[[A:.+]] = spirv.NV.CooperativeMatrixLoad
          %6 = vector.transfer_read %arg0[%c0, %arg3], %cst : memref<4096x256xvector<4xi32>>, vector<16x32xi8>
          // CHECK: %[[B:.+]] = spirv.NV.CooperativeMatrixLoad
          %7 = vector.transfer_read %arg1[%arg3, %c0], %cst : memref<4096x256xvector<4xi32>>, vector<32x16xi8>
          // CHECK: %[[R:.+]] = spirv.NV.CooperativeMatrixMulAdd %[[A]], %[[B]], %{{.*}}
          %8 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} %6, %7, %arg4 : vector<16x32xi8>, vector<32x16xi8> into vector<16x16xi32>
          scf.yield %8 : vector<16x16xi32>
        }
        // CHECK: spirv.NV.CooperativeMatrixStore
        vector.transfer_write %5, %arg2[%c0, %c0] : vector<16x16xi32>, memref<4096x1024xvector<4xi32>>
        return
      }
    }
  }
}

// -----

hal.executable private @const_elementwise_ops  {
  hal.executable.variant @vulkan, target = <"vulkan-spirv", "vulkan-spirv-fb", {
    spirv.target_env = #spirv.target_env<#spirv.vce<v1.0,
        [Shader, CooperativeMatrixNV, Float16],
        [SPV_KHR_storage_buffer_storage_class, SPV_NV_cooperative_matrix]>,
        #spirv.resource_limits<max_compute_workgroup_invocations = 128, max_compute_workgroup_size = [128, 128, 64]>>}> {
    builtin.module {
      // CHECK-LABEL: func.func @const_elementwise_ops
      func.func @const_elementwise_ops(%add_val: vector<16x16xf16>, %sub_val: vector<16x16xf16>, %div_val: vector<16x16xf16>) -> vector<16x16xf16> {
        // CHECK: %[[SPLAT:.+]] = spirv.Constant 8.000000e+00 : f16
        // CHECK: %[[CST:.+]] = spirv.CompositeConstruct %[[SPLAT]] : (f16) -> !spirv.coopmatrix<16x16xf16, Subgroup>
        %eight = arith.constant dense<8.0> : vector<16x16xf16>
        // CHECK: %{{.+}} = spirv.FAdd %[[CST]], %{{.+}} : !spirv.coopmatrix<16x16xf16, Subgroup>
        %add = arith.addf %eight, %add_val: vector<16x16xf16>
        // CHECK: %{{.+}} = spirv.FSub %{{.+}}, %{{.+}} !spirv.coopmatrix<16x16xf16, Subgroup>
        %sub = arith.subf %add, %sub_val : vector<16x16xf16>
        // CHECK: %{{.+}} = spirv.FDiv %{{.+}}, %{{.+}} !spirv.coopmatrix<16x16xf16, Subgroup>
        %div = arith.divf %sub, %div_val : vector<16x16xf16>
        return %div: vector<16x16xf16>
      }
    }
  }
}
