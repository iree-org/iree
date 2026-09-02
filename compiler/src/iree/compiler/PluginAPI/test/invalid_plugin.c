// clang-format off
// RUN: mkdir -p %t
// RUN: clang -fPIC -shared -o %t/libinvalid_plugin.so %s
// RUN: (iree-opt --iree-load-plugin=%t/libinvalid_plugin.so 2>&1 || true) | FileCheck %s --check-prefix=PLUGIN_LOAD_FAIL
// clang-format on

#include <stdbool.h>

bool some_function() { return false; }

// clang-format off
// PLUGIN_LOAD_FAIL: [IREE Dynamic Plugin ERROR]: plugin{{.*}}defines no iree_get_compiler_plugin_info
// clang-format on
