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

#ifndef IREE_COMPILER_SERIALIZATION_VM_SOURCE_MAP_BUILDER_H_
#define IREE_COMPILER_SERIALIZATION_VM_SOURCE_MAP_BUILDER_H_

#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "schemas/source_map_def_generated.h"

namespace mlir {
namespace iree_compiler {

struct VMFunctionSourceMap {
  std::vector<std::pair<int, Location>> locations;
};

class VMSourceMapBuilder {
 public:
  explicit VMSourceMapBuilder(::flatbuffers::FlatBufferBuilder *fbb);

  LogicalResult AddFunction(int functionOrdinal,
                            VMFunctionSourceMap functionSourceMap);

  ::flatbuffers::Offset<iree::SourceMapDef> Finish(int maxFunctionOrdinal);

 private:
  struct LocationOffsetTable {
    std::vector<::flatbuffers::Offset<iree::LocationDef>> locationDefs;
    llvm::DenseMap<Location, int> locationMap;
  };

  int GetUniqueString(std::string value);

  ::flatbuffers::Offset<iree::FunctionSourceMapDef>
  SerializeVMFunctionSourceMap(const VMFunctionSourceMap &functionMap);
  int SerializeLocation(const Location &location,
                        LocationOffsetTable *locationOffsetTable);

  ::flatbuffers::FlatBufferBuilder *fbb_;
  std::vector<std::string> stringTable_;
  llvm::StringMap<int> stringTableMap_;
  std::vector<VMFunctionSourceMap> functionMaps_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_SERIALIZATION_VM_SOURCE_MAP_BUILDER_H_
