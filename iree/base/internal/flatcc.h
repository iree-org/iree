#// Copyright 2020 Google LLC
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

#ifndef IREE_BASE_INTERNAL_FLATCC_H_
#define IREE_BASE_INTERNAL_FLATCC_H_

//===----------------------------------------------------------------------===//
// flatcc include order fixes
//===----------------------------------------------------------------------===//
//
// This header merely wraps the flatcc headers that are generally useful to
// include in various places that may not know the specific messages they are
// working with.
//
// If using flatcc prefer to include this file over any hard-to-handle flatcc
// file such as flatbuffers_common_reader.h or flatbuffers_common_builder.h.
//
// NOTE: order matters for these includes so stop clang from messing with it:
// clang-format off

#include "flatcc/reflection/flatbuffers_common_reader.h"
#include "iree/base/internal/flatcc_reader.h"

#include "flatcc/flatcc_verifier.h"
#include "iree/base/internal/flatcc_verifier.h"

#include "flatcc/flatcc_builder.h"
#include "flatcc/reflection/flatbuffers_common_builder.h"
#include "iree/base/internal/flatcc_builder.h"

#include "flatcc/flatcc_json_parser.h"
#include "iree/base/internal/flatcc_json_parser.h"

#include "flatcc/flatcc_json_printer.h"
#include "iree/base/internal/flatcc_json_printer.h"

// clang-format on

#endif  // IREE_BASE_INTERNAL_FLATCC_H_
