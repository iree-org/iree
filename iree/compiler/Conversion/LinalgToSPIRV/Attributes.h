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

/// Enumerates the methods used to compute the number of workgroups to use with
/// an entry point function. The lowering to SPIR-V sets an integer attribute on
/// the entry point function with one of these values. It is later used by
/// `recordDispatch` to compute the number of workgroups for the entry point
/// function.
enum class WorkgroupCountMethodology {
  // TODO(#2134): Remove the `Default` option. This is only a fallback used for
  // convolution/pooling cases that are currently not working as intended, as
  // described in the bug.
  Default = 0,               // Use the default mechanism used by IREE
  LinearizeResultShape = 1,  // Use the linearized shape of the result of the
                             // dispatch region
  ResultShape = 2            // Use the shape of the dispatch region.
};

/// Returns the name of the attribute to use that propagates the method to use
/// to compute the number of workgroups to use with an entry point function. The
/// attribute used is an IntegerAttr with value being one of the enum entries of
/// WorkgroupCountMethodology.
// TODO(ravishankarm): The approach to use attributes to propagate the
// methodology to use to compute number of workgroups is to convoluted. Ideally,
// the lowering should produce a function that should then just be inlined at
// the point this is needed.
inline llvm::StringRef getWorkgroupCountAttrName() {
  return "vkspv.workgroup_count_from_result_shape";
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_LINALGTOSPIRV_ATTRIBUTES_H_
