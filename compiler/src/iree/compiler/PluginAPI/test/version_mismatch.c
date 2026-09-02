// clang-format off
// RUN: mkdir -p %t
// RUN: clang -fPIC -shared -o %t/libversion_mismatch.so %s
// RUN: (iree-opt --iree-load-plugin=%t/libversion_mismatch.so 2>&1 || true) | FileCheck %s --check-prefix=VERSION_MISMATCH
// clang-format on

// A plugin from a future ABI must be refused before its registration function
// is called, not left to fail on a virtual call.

#include <stdbool.h>

struct IreeCompilerPluginInfo {
  int apiVersion;
  const char *pluginId;
  bool (*registerPlugin)(void *registrar);
};

static bool register_future_plugin(void *registrar) {
  (void)registrar;
  return true;
}

struct IreeCompilerPluginInfo iree_get_compiler_plugin_info(void) {
  struct IreeCompilerPluginInfo info = {999, "from_the_future",
                                        register_future_plugin};
  return info;
}

// clang-format off
// VERSION_MISMATCH: [IREE Dynamic Plugin ERROR]: plugin{{.*}}was built against plugin API version 999
// clang-format on
