// clang-format off
// RUN: mkdir -p %t
// RUN: clang -fPIC -shared -o %t/libvalid_plugin.so %s
// RUN: iree-opt --iree-load-plugin=sample_plugin=%t/libvalid_plugin.so --help
// clang-format on

#include <stdbool.h>

bool iree_register_compiler_plugin_sample_plugin(void *registrar) {
  (void)registrar;
  return true;
}
