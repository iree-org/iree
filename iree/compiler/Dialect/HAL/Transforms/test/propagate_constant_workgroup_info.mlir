// RUN: iree-opt -allow-unregistered-dialect -split-input-file -pass-pipeline='hal.executable(hal.executable.variant(iree-hal-propagate-constant-workgroup-info))' %s | IreeFileCheck %s

hal.executable @exe {
  hal.interface @interface {
    hal.interface.binding @s0b0, set=0, binding=0, type="StorageBuffer", access="Read"
    hal.interface.binding @s0b1, set=0, binding=1, type="StorageBuffer", access="Read|Write"
  }
  hal.executable.variant @target, target="target" {
    hal.executable.entry_point @entry attributes {
      interface = @interface,
      ordinal = 0 : index,
      workgroup_size = [32 : index, 4 : index, 8 : index]
    }
    module {
      // CHECK: func @entry()
      func @entry() {
        // CHECK-DAG: constant 32 : index
        %workgroup_size_x = hal.interface.workgroup.size[0] : index
        // CHECK-DAG: constant 4 : index
        %workgroup_size_y = hal.interface.workgroup.size[1] : index
        return
      }
    }
  }
}
