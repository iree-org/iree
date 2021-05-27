// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
#include "iree/base/internal/statusor.h"
#endif  // __cplusplus

#endif  // IREE_BASE_STATUS_H_
