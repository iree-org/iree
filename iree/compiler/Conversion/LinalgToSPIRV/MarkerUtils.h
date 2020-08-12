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

//===- MarkerUtils.h - Methods for manipulating markers on Linalg op ------===//
//
// Method that set markers on Linalg operations that determine which processor
// heirarchy to use for partitioning
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_MARKERUTILS_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_MARKERUTILS_H_

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
namespace iree_compiler {

/// Marker to denote that a linalg operation is to be partitioned to
/// workitems. No assumption can be made about the number of woritems in the
/// workgroup and number of iterations, i.e. a cyclic distribution is required.
StringRef getWorkgroupMarker();
StringRef getWorkgroupMemoryMarker();

/// Marker to denote that a linalg operation is to be partitioned to workitems
/// with the assumption that the number of workitems in the workgroup is greater
/// than equal to the number of iterations.
StringRef getWorkgroupNumItemsGENumItersMarker();
StringRef getWorkgroupMemoryNumItemsGENumItersMarker();

/// Marker for copy operations that are moving data from StorageClass to
/// Workgroup memory.
StringRef getCopyToWorkgroupMemoryMarker();

/// Returns true if an operation has the specified `marker`. When `marker` is
/// empty, returns true if the operation has any marker.
bool hasMarker(Operation *, ArrayRef<StringRef> markers = {});

/// Sets a given marker on an operation.
void setMarker(Operation *, StringRef);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_MARKERUTILS_H_
