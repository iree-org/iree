// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-codegen-fold-memref-copy-into-dps-ops))" %s | FileCheck %s

#map = affine_map<(d0) -> (d0)>

func.func @fold_linalg_generic_temp() {
  %source = memref.alloc() : memref<4xf32>
  %target = memref.alloc() : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]
  } outs(%temp : memref<4xf32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @fold_linalg_generic_temp(
// CHECK-DAG: %[[SOURCE:.+]] = memref.alloc
// CHECK-DAG: %[[TARGET:.+]] = memref.alloc
// CHECK: memref.copy %[[SOURCE]], %[[TARGET]]
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[TARGET]] : memref<4xf32>)
// CHECK-NOT: memref.copy

// -----

#map = affine_map<(d0) -> (d0)>

func.func @no_fold_copy_source_aliases_target(%target: memref<4xf32>) {
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %target, %temp : memref<4xf32> to memref<4xf32>
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]
  } outs(%temp : memref<4xf32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @no_fold_copy_source_aliases_target(
// CHECK-SAME:    %[[TARGET:.+]]: memref<4xf32>
// CHECK: %[[TEMP:.+]] = memref.alloc
// CHECK: memref.copy %[[TARGET]], %[[TEMP]]
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[TEMP]] : memref<4xf32>)
// CHECK: memref.copy %[[TEMP]], %[[TARGET]]

// -----

func.func @elide_linalg_fill_temp() {
  %c0 = arith.constant 0.0 : f32
  %source = memref.alloc() : memref<4xf32>
  %target = memref.alloc() : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  linalg.fill ins(%c0 : f32) outs(%temp : memref<4xf32>)
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @elide_linalg_fill_temp(
// CHECK-DAG: %[[TARGET:.+]] = memref.alloc
// CHECK-NOT: memref.copy
// CHECK: linalg.fill
// CHECK-SAME: outs(%[[TARGET]] : memref<4xf32>)
// CHECK-NOT: memref.copy

// -----

#map = affine_map<(d0) -> (d0)>

func.func @elide_linalg_generic_full_overwrite() {
  %source = memref.alloc() : memref<4xf32>
  %target = memref.alloc() : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]
  } outs(%temp : memref<4xf32>) {
  ^bb0(%out: f32):
    %c1 = arith.constant 1.0 : f32
    linalg.yield %c1 : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @elide_linalg_generic_full_overwrite(
// CHECK-DAG: %[[TARGET:.+]] = memref.alloc
// CHECK-NOT: memref.copy
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[TARGET]] : memref<4xf32>)
// CHECK-NOT: memref.copy

// -----

memref.global "private" constant @full_scatter_indices : memref<4x1xi32> = dense<[[0], [1], [2], [3]]>

func.func @elide_static_full_scatter_temp() {
  %source = memref.alloc() : memref<4xf32>
  %target = memref.alloc() : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  %updates = memref.alloc() : memref<4xf32>
  %indices = memref.get_global @full_scatter_indices : memref<4x1xi32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%updates, %indices : memref<4xf32>, memref<4x1xi32>)
      outs(%temp : memref<4xf32>) {
  ^bb0(%update: f32, %old: f32):
    iree_linalg_ext.yield %update : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @elide_static_full_scatter_temp(
// CHECK-DAG: %[[TARGET:.+]] = memref.alloc
// CHECK-NOT: memref.copy
// CHECK: iree_linalg_ext.scatter
// CHECK-SAME: outs(%[[TARGET]] : memref<4xf32>)
// CHECK-NOT: memref.copy

// -----

memref.global "private" constant @partial_scatter_indices : memref<2x1xi32> = dense<[[0], [2]]>

func.func @preserve_copy_for_partial_scatter() {
  %source = memref.alloc() : memref<4xf32>
  %target = memref.alloc() : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  %updates = memref.alloc() : memref<2xf32>
  %indices = memref.get_global @partial_scatter_indices : memref<2x1xi32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%updates, %indices : memref<2xf32>, memref<2x1xi32>)
      outs(%temp : memref<4xf32>) {
  ^bb0(%update: f32, %old: f32):
    iree_linalg_ext.yield %update : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @preserve_copy_for_partial_scatter(
// CHECK-DAG: %[[SOURCE:.+]] = memref.alloc
// CHECK-DAG: %[[TARGET:.+]] = memref.alloc
// CHECK: memref.copy %[[SOURCE]], %[[TARGET]]
// CHECK: iree_linalg_ext.scatter
// CHECK-SAME: outs(%[[TARGET]] : memref<4xf32>)
// CHECK-NOT: memref.copy

// -----

memref.global "private" constant @old_value_scatter_indices : memref<4x1xi32> = dense<[[0], [1], [2], [3]]>

func.func @preserve_copy_for_scatter_combiner_read() {
  %source = memref.alloc() : memref<4xf32>
  %target = memref.alloc() : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  %updates = memref.alloc() : memref<4xf32>
  %indices = memref.get_global @old_value_scatter_indices : memref<4x1xi32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  iree_linalg_ext.scatter dimension_map = [0] unique_indices(true)
      ins(%updates, %indices : memref<4xf32>, memref<4x1xi32>)
      outs(%temp : memref<4xf32>) {
  ^bb0(%update: f32, %old: f32):
    %sum = arith.addf %update, %old : f32
    iree_linalg_ext.yield %sum : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @preserve_copy_for_scatter_combiner_read(
// CHECK-DAG: %[[SOURCE:.+]] = memref.alloc
// CHECK-DAG: %[[TARGET:.+]] = memref.alloc
// CHECK: memref.copy %[[SOURCE]], %[[TARGET]]
// CHECK: iree_linalg_ext.scatter
// CHECK-SAME: outs(%[[TARGET]] : memref<4xf32>)
// CHECK-NOT: memref.copy

// -----

#map = affine_map<(d0) -> (d0)>

func.func @no_fold_dps_input_aliases_target(%source: memref<4xf32>,
                                            %target: memref<4xf32>) {
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  linalg.generic {
    indexing_maps = [#map, #map],
    iterator_types = ["parallel"]
  } ins(%target : memref<4xf32>) outs(%temp : memref<4xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @no_fold_dps_input_aliases_target(
// CHECK-SAME:    %[[SOURCE:.+]]: memref<4xf32>, %[[TARGET:.+]]: memref<4xf32>
// CHECK: %[[TEMP:.+]] = memref.alloc
// CHECK: memref.copy %[[SOURCE]], %[[TEMP]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[TARGET]] : memref<4xf32>)
// CHECK-SAME: outs(%[[TEMP]] : memref<4xf32>)
// CHECK: memref.copy %[[TEMP]], %[[TARGET]]

// -----

#map = affine_map<(d0) -> (d0)>

func.func @no_fold_intervening_target_access(%source: memref<4xf32>,
                                             %target: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  %read = memref.load %target[%c0] : memref<4xf32>
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]
  } outs(%temp : memref<4xf32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return %read : f32
}

// CHECK-LABEL: func.func @no_fold_intervening_target_access(
// CHECK-SAME:    %[[SOURCE:.+]]: memref<4xf32>, %[[TARGET:.+]]: memref<4xf32>
// CHECK: %[[TEMP:.+]] = memref.alloc
// CHECK: memref.copy %[[SOURCE]], %[[TEMP]]
// CHECK: memref.load %[[TARGET]]
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[TEMP]] : memref<4xf32>)
// CHECK: memref.copy %[[TEMP]], %[[TARGET]]

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0) -> (d0)>

func.func @fold_distinct_hal_bindings() {
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) : memref<4xf32>
  %target = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]
  } outs(%temp : memref<4xf32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @fold_distinct_hal_bindings(
// CHECK-DAG: %[[SOURCE:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG: %[[TARGET:.+]] = hal.interface.binding.subspan {{.+}} binding(1)
// CHECK: memref.copy %[[SOURCE]], %[[TARGET]]
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[TARGET]] : memref<4xf32>)
// CHECK-NOT: memref.copy

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>
]>
#map = affine_map<(d0) -> (d0)>

func.func @no_fold_same_hal_binding() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %source = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c0) : memref<4xf32>
  %target = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) offset(%c64) : memref<4xf32>
  %temp = memref.alloc() : memref<4xf32>
  memref.copy %source, %temp : memref<4xf32> to memref<4xf32>
  linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]
  } outs(%temp : memref<4xf32>) {
  ^bb0(%out: f32):
    linalg.yield %out : f32
  }
  memref.copy %temp, %target : memref<4xf32> to memref<4xf32>
  return
}

// CHECK-LABEL: func.func @no_fold_same_hal_binding(
// CHECK-DAG: %[[SOURCE:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK-DAG: %[[TARGET:.+]] = hal.interface.binding.subspan {{.+}} binding(0)
// CHECK: %[[TEMP:.+]] = memref.alloc
// CHECK: memref.copy %[[SOURCE]], %[[TEMP]]
// CHECK: linalg.generic
// CHECK-SAME: outs(%[[TEMP]] : memref<4xf32>)
// CHECK: memref.copy %[[TEMP]], %[[TARGET]]
