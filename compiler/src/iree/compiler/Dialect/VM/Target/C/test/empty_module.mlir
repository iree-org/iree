// RUN: iree-translate -iree-vm-ir-to-c-module %s | FileCheck %s

// CHECK: #include "iree/vm/ops.h"
vm.module @empty_module {
}
