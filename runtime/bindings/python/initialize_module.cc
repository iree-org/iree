// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "./binding.h"
#include "./hal.h"
#include "./invoke.h"
#include "./io.h"
#include "./loop.h"
#include "./numpy_interop.h"
#include "./py_module.h"
#include "./status_utils.h"
#include "./vm.h"
#include "iree/base/internal/flags.h"
#include "iree/hal/drivers/init.h"

namespace {
// Stable storage for flag processing.  Flag handling uses string views,
// expecting the caller to keep the original strings around for as long
// as the flags are in use.  This object holds one set of flag strings
// for each invocation of parse_flags.
std::vector<std::unique_ptr<std::vector<std::string>>> alloced_flag_cache;
}  // namespace

namespace iree {
namespace python {

NB_MODULE(_runtime, m) {
  numpy::InitializeNumPyInterop();
  IREE_TRACE_APP_ENTER();

  IREE_CHECK_OK(iree_hal_register_all_available_drivers(
      iree_hal_driver_registry_default()));

  m.doc() = "IREE Binding Backend Helpers";
  SetupHalBindings(m);
  SetupInvokeBindings(m);
  SetupIoBindings(m);
  SetupLoopBindings(m);
  SetupPyModuleBindings(m);
  SetupVmBindings(m);

  // Adds the given set of strings to the global flags.  These new flags
  // take effect upon the next creation of a driver.  They do not affect
  // drivers already created.
  m.def("parse_flags", [](py::args py_flags) {
    // Make a new set of strings at the back of the cache
    alloced_flag_cache.emplace_back(
        std::make_unique<std::vector<std::string>>(std::vector<std::string>()));
    auto &alloced_flags = *alloced_flag_cache.back();

    // Add the given python strings to the std::string set.
    alloced_flags.push_back("python");
    for (py::handle py_flag : py_flags) {
      alloced_flags.push_back(py::cast<std::string>(py_flag));
    }

    // As the flags-processing mechanism of the C API requires long-lived
    // char * strings, create a set of char * strings from the std::strings,
    // with the std::strings responsible for maintaining the storage.
    // Must build pointer vector after filling std::strings so pointers are
    // stable.
    std::vector<char *> flag_ptrs;
    for (auto &alloced_flag : alloced_flags) {
      flag_ptrs.push_back(const_cast<char *>(alloced_flag.c_str()));
    }

    // Send the flags to the C API
    char **argv = &flag_ptrs[0];
    int argc = flag_ptrs.size();
    CheckApiStatus(iree_flags_parse(IREE_FLAGS_PARSE_MODE_CONTINUE_AFTER_HELP,
                                    &argc, &argv),
                   "Error parsing flags");
  });

  m.def("disable_leak_checker", []() { py::set_leak_warnings(false); });
}

}  // namespace python
}  // namespace iree
