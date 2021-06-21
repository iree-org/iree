// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_STATUS_H_
#define IREE_BASE_STATUS_H_

//===----------------------------------------------------------------------===//
//                                                                            //
//  (             (      (                                             (      //
//  )\ )          )\ )   )\ )           (       (        *   )         )\ )   //
// (()/(    (    (()/(  (()/(   (       )\      )\     ` )  /(   (    (()/(   //
//  /(_))   )\    /(_))  /(_))  )\    (((_)  ((((_)(    ( )(_))  )\    /(_))  //
// (_))_   ((_)  (_))   (_))   ((_)   )\___   )\ _ )\  (_(_())  ((_)  (_))_   //
//  |   \  | __| | _ \  | _ \  | __| ((/ __|  (_)_\(_) |_   _|  | __|  |   \  //
//  | |) | | _|  |  _/  |   /  | _|   | (__    / _ \     | |    | _|   | |) | //
//  |___/  |___| |_|    |_|_\  |___|   \___|  /_/ \_\    |_|    |___|  |___/  //
//                                                                            //
//===----------------------------------------------------------------------===//
// TODO(#3848): this is a C++ wrapper that we shouldn't have - it's a large
// amount of code with a tight coupling to abseil and it's just not needed in
// the core (as we are almost exclusively C now). This header file should be
// replaced with the C iree_status_t implementation currently shoved in
// iree/base/api.h in order to not conflict with this file.
// We may still want a simple Status wrapper just to get RAII, but definitely
// not what StatusOr is doing (an std::pair<Status, T> would be more than
// enough).

#ifdef __cplusplus
#include "iree/base/internal/status.h"
#endif  // __cplusplus

#endif  // IREE_BASE_STATUS_H_
