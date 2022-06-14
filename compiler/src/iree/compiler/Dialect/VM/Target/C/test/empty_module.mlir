// RUN: iree-compile --compile-mode=vm --output-format=vm-c %s | FileCheck %s

// CHECK: #include "iree/vm/ops.h"
vm.module @empty_module {
}
