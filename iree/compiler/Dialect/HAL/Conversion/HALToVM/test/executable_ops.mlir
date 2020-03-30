// RUN: iree-opt -split-input-file -iree-convert-hal-to-vm %s | IreeFileCheck %s

hal.executable @exe {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.entry_point @entry attributes {
    interface = @interface,
    ordinal = 0 : i32,
    signature = (tensor<4xf32>) -> tensor<4xf32>,
    workgroup_size = [32 : index, 1 : index, 1 : index]
  }
  hal.executable.entry_point @entry_alias attributes {
    interface = @interface,
    ordinal = 0 : i32,
    signature = (tensor<4xf32>) -> tensor<4xf32>,
    workgroup_size = [32 : index, 1 : index, 1 : index]
  }
  hal.executable.binary attributes {
    data = dense<[0, 1, 2, 3]> : vector<4xi8>,
    format = 1230128453 : i32
  }
  hal.executable.binary attributes {
    data = dense<[4, 5, 6, 7]> : vector<4xi8>,
    format = 1397773893 : i32
  }
}

// CHECK-LABEL: @executableCachePrepare
func @executableCachePrepare(%arg0 : !hal.executable_cache, %arg1 : !hal.executable_layout) -> !hal.executable {
  %0 = hal.executable_cache.prepare %arg0, layout = %arg1, caching_mode = "AllowOptimization", @exe : !hal.executable
  return %0 : !hal.executable
}
