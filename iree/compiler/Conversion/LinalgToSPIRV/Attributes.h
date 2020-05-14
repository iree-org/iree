// Copyright 2020 Google LLC
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

#ifndef IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_ATTRIBUTES_H_
#define IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_ATTRIBUTES_H_

#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace iree_compiler {

/// Attribute on a module op to denote the scheduling order of entry points.
/// The attribute value is expected to be an array of entry point name strings.
inline llvm::StringRef getEntryPointScheduleAttrName() {
  return "vkspv.entry_point_schedule";
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_ATTRIBUTES_H_
