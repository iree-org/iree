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

#ifndef IREE_BASE_PLATFORM_HEADERS_H_
#define IREE_BASE_PLATFORM_HEADERS_H_

#include "iree/base/target_platform.h"

#if defined(IREE_PLATFORM_WINDOWS)

#if defined(_MSC_VER)
// Abseil compatibility: don't include incompatible winsock versions.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
// Abseil compatibility: don't define min and max macros.
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#endif  // _MSC_VER

#include <windows.h>

// WinGDI.h defines `ERROR`, undef to avoid conflict naming.
#undef ERROR

#endif  // IREE_PLATFORM_WINDOWS

#endif  // IREE_BASE_PLATFORM_HEADERS_H_
