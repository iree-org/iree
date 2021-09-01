// RUN: iree-opt -pass-pipeline='hal.executable(hal.executable.variant(builtin.module(builtin.func(iree-llvmgpu-distribute-shared-memory-copy))))' -cse %s | IreeFileCheck %s

// CHECK-DAG: #[[$MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4)>
// CHECK-DAG: #[[$MAP1:.*]] = affine_map<()[s0] -> (s0 * 4 - (s0 floordiv 4) * 16)>
// CHECK-DAG: #[[$MAP2:.*]] = affine_map<()[s0, s1, s2] -> (s1 * 8 + s2 * 32 + s0 floordiv 4 + 32)>
// CHECK-DAG: #[[$MAP3:.*]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 128)>
// CHECK-DAG: #[[$MAP4:.*]] = affine_map<()[s0, s1, s2] -> (s0 + s1 * 32 + s2 * 128 + 128)>
// CHECK-DAG: #[[$MAP5:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 4 + s1 * 128 + s2 * 512)>

hal.executable @shared_mem_cpy attributes {sym_visibility = "private"} {
  hal.executable.variant @cuda, target = #hal.executable.target<"cuda", "cuda-nvptx-fb"> {
    hal.executable.entry_point @shared_mem_cpy attributes {
      interface = @io,
      ordinal = 0 : index,
      workgroup_size = [32: index, 4: index, 1:index]}
    builtin.module  {
      memref.global "private" @__shared_memory___1 : memref<3x512xf32, 3>
      memref.global "private" @__shared_memory___0 : memref<256x4xf32, 3>
      memref.global "private" @__shared_memory__ : memref<64x16xf32, 3>
    // CHECK-LABEL: @shared_mem_cpy(
      builtin.func @shared_mem_cpy(
        %m0 : memref<64x16xf32>, %m1 : memref<256x4xf32>, %m2 : memref<3x512xf32>) {
        %sm0 = memref.get_global @__shared_memory__ : memref<64x16xf32, 3>
        %sm1 = memref.get_global @__shared_memory___0 : memref<256x4xf32, 3>
        %sm2 = memref.get_global @__shared_memory___1 : memref<3x512xf32, 3>
        gpu.barrier
    // CHECK-DAG: %[[C2:.*]] = constant 2 : index
    // CHECK-DAG: %[[C1:.*]] = constant 1 : index
    // CHECK-DAG: %[[C0:.*]] = constant 0 : index
    // CHECK-DAG: %[[TX:.*]] = "gpu.thread_id"() {dimension = "x"} : () -> index
    // CHECK-DAG: %[[TY:.*]] = "gpu.thread_id"() {dimension = "y"} : () -> index
    // CHECK-DAG: %[[TZ:.*]] = "gpu.thread_id"() {dimension = "z"} : () -> index

    // CHECK-DAG: %[[Y0:.*]] = affine.apply #[[$MAP0]]()[%[[TX]], %[[TY]], %[[TZ]]]
    // CHECK-DAG: %[[X0:.*]] = affine.apply #[[$MAP1]]()[%[[TX]]]
    //     CHECK: %[[R0:.*]] = vector.transfer_read %{{.*}}[%[[Y0]], %[[X0]]], %{{.*}} {in_bounds = [true, true]} : memref<64x16xf32>, vector<1x4xf32>
    // CHECK-DAG: %[[Y1:.*]] = affine.apply #[[$MAP2]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R1:.*]] = vector.transfer_read %{{.*}}[%[[Y1]], %[[X0]]], %{{.*}} {in_bounds = [true, true]} : memref<64x16xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R0]], %{{.*}}[%[[Y0]], %[[X0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<64x16xf32, 3>
    //     CHECK: vector.transfer_write %[[R1]], %{{.*}}[%[[Y1]], %[[X0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<64x16xf32, 3>

        linalg.copy(%m0, %sm0) {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<64x16xf32>, memref<64x16xf32, 3>

    //     CHECK: %[[Y1:.*]] = affine.apply #[[$MAP3]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R2:.*]] = vector.transfer_read %{{.*}}[%[[Y1]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : memref<256x4xf32>, vector<1x4xf32>
    //     CHECK: %[[Y2:.*]] = affine.apply #[[$MAP4]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R3:.*]] = vector.transfer_read %{{.*}}[%[[Y2]], %[[C0]]], %{{.*}} {in_bounds = [true, true]} : memref<256x4xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R2]], %{{.*}}[%[[Y1]], %[[C0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<256x4xf32, 3>
    //     CHECK: vector.transfer_write %[[R3]], %{{.*}}[%[[Y2]], %[[C0]]] {in_bounds = [true, true]} : vector<1x4xf32>, memref<256x4xf32, 3>

        linalg.copy(%m1, %sm1) {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<256x4xf32>, memref<256x4xf32, 3>

    //     CHECK: %[[X1:.*]] = affine.apply #[[$MAP5]]()[%[[TX]], %[[TY]], %[[TZ]]]
    //     CHECK: %[[R4:.*]] = vector.transfer_read %{{.*}}[%[[C0]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: %[[R5:.*]] = vector.transfer_read %{{.*}}[%[[C1]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: %[[R6:.*]] = vector.transfer_read %{{.*}}[%[[C2]], %[[X1]]], %{{.*}} {in_bounds = [true, true]} : memref<3x512xf32>, vector<1x4xf32>
    //     CHECK: vector.transfer_write %[[R4]], %{{.*}}[%c0, %15] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>
    //     CHECK: vector.transfer_write %[[R5]], %{{.*}}[%c1, %15] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>
    //     CHECK: vector.transfer_write %[[R6]], %{{.*}}[%c2, %15] {in_bounds = [true, true]} : vector<1x4xf32>, memref<3x512xf32, 3>

        linalg.copy(%m2, %sm2) {__internal_linalg_transform__ = "copy_to_workgroup_memory"} : memref<3x512xf32>, memref<3x512xf32, 3>
        gpu.barrier
        return
      }
    }
  }
}
