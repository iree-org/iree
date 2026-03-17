// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// JS backend registration for CTS tests.
//
// The JS proactor uses the JavaScript event loop as its kernel. When running
// under the worker test runner, timers fire via setTimeout on the event host
// thread and completions are delivered through a SharedArrayBuffer ring.
//
// Currently supports NOP and timer operations. Socket, event, and notification
// operations return UNAVAILABLE.

#include "iree/async/cts/util/registry.h"
#include "iree/async/platform/js/proactor.h"
#include "iree/async/proactor.h"

namespace iree::async::cts {

static iree::StatusOr<iree_async_proactor_t*> CreateJSProactor() {
  iree_async_proactor_t* proactor = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_async_proactor_create_js(iree_async_proactor_options_default(),
                                    iree_allocator_system(), &proactor));
  return proactor;
}

// JS worker-mode backend: timers via JS setTimeout, completions via SAB ring.
static bool js_registered_ = (CtsRegistry::RegisterBackend({
                                  "js",
                                  {"js", CreateJSProactor},
                                  {"portable"},
                              }),
                              true);

}  // namespace iree::async::cts
