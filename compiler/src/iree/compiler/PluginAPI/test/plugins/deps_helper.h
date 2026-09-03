// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINAPI_TEST_PLUGINS_DEPS_HELPER_H_
#define IREE_COMPILER_PLUGINAPI_TEST_PLUGINS_DEPS_HELPER_H_

namespace iree_plugin_test {
// In a library of its own, and in nothing the compiler links, so a plugin link
// that reaches only the registration target leaves this undefined.
bool helperSucceeded();
} // namespace iree_plugin_test

#endif // IREE_COMPILER_PLUGINAPI_TEST_PLUGINS_DEPS_HELPER_H_
