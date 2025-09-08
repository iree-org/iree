// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx942 --iree-hip-encoding-layout-resolver=pad %s | FileCheck %s --check-prefix=PAD
//
// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx90a --iree-hip-encoding-layout-resolver=pad %s | FileCheck %s --check-prefix=PAD

// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx90a --iree-hip-encoding-layout-resolver=data-tiling %s | FileCheck %s --check-prefix=DATA-TILING

// RUN: iree-opt --pass-pipeline='builtin.module(iree-hal-assign-target-devices{targetDevices=hip},iree-hal-transformation-pipeline{serialize-executables=false})' \
// RUN:   --iree-hip-target=gfx90a --iree-hip-encoding-layout-resolver=none %s | FileCheck %s --check-prefix=NONE

// PAD:      #hal.executable.target<"rocm"
// PAD-SAME:   iree.encoding.resolver = #iree_gpu.gpu_padding_resolver<>

// DATA-TILING:      #hal.executable.target<"rocm"
// DATA-TILING-SAME:   iree.encoding.resolver = #iree_gpu.gpu_encoding_resolver<>

// NONE:      #hal.executable.target<"rocm"
// NONE-NOT:    iree.encoding.resolver

stream.executable public @main {
  stream.executable.export @main workgroups(%arg0: index) -> (index, index, index) {
    %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_dag_root(%arg0)
    stream.return %x, %y, %z : index, index, index
  }
  builtin.module {
    func.func @main() {
      return
    }
  }
}
