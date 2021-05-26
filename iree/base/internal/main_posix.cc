// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/main.h"
#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_ANDROID) || defined(IREE_PLATFORM_APPLE) || \
    defined(IREE_PLATFORM_LINUX)

namespace iree {
namespace {

extern "C" int main(int argc, char** argv) { return iree_main(argc, argv); }

}  // namespace
}  // namespace iree

#endif  // IREE_PLATFORM_*
