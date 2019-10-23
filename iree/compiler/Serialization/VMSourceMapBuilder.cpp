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

#include "iree/compiler/Serialization/VMSourceMapBuilder.h"

#include "flatbuffers/flatbuffers.h"
#include "iree/schemas/source_map_def_generated.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace iree_compiler {

VMSourceMapBuilder::VMSourceMapBuilder(::flatbuffers::FlatBufferBuilder *fbb)
    : fbb_(fbb) {}

int VMSourceMapBuilder::GetUniqueString(std::string value) {
  auto it = stringTableMap_.find(value);
  if (it != stringTableMap_.end()) {
    return it->second;
  }
  int stringIndex = stringTable_.size();
  stringTableMap_.insert({value, stringIndex});
  stringTable_.push_back(std::move(value));
  return stringIndex;
}

LogicalResult VMSourceMapBuilder::AddFunction(
    int functionOrdinal, VMFunctionSourceMap functionSourceMap) {
  if (functionMaps_.size() <= functionOrdinal) {
    functionMaps_.resize(functionOrdinal + 1);
  }
  functionMaps_[functionOrdinal] = std::move(functionSourceMap);
  return success();
}

::flatbuffers::Offset<iree::SourceMapDef> VMSourceMapBuilder::Finish(
    int maxFunctionOrdinal) {
  // NOTE: we always ensure the source map table is the same size as the
  // function table so that lookups at runtime can be validated once at load
  // time (ensuring the tables match up) instead of on each lookup.
  if (maxFunctionOrdinal < functionMaps_.size()) {
    llvm::errs() << "Max function ordinal defined as " << maxFunctionOrdinal
                 << " but there are " << functionMaps_.size()
                 << " function source maps present";
    return {};
  }
  functionMaps_.resize(maxFunctionOrdinal);

  std::vector<::flatbuffers::Offset<iree::FunctionSourceMapDef>> functionDefs;
  functionDefs.resize(maxFunctionOrdinal);
  for (int i = 0; i < functionMaps_.size(); ++i) {
    const auto &functionMap = functionMaps_[i];
    functionDefs[i] = SerializeVMFunctionSourceMap(functionMap);
    if (functionDefs[i].IsNull()) return {};
  }

  auto functionTableOffset = fbb_->CreateVector(functionDefs);
  auto stringTableOffset = fbb_->CreateVectorOfStrings(stringTable_);
  iree::SourceMapDefBuilder smdb(*fbb_);
  smdb.add_function_table(functionTableOffset);
  smdb.add_string_table(stringTableOffset);
  return smdb.Finish();
}

::flatbuffers::Offset<iree::FunctionSourceMapDef>
VMSourceMapBuilder::SerializeVMFunctionSourceMap(
    const VMFunctionSourceMap &functionMap) {
  if (functionMap.locations.empty()) {
    // Empty table. This ensures that we still have a non-null value in the
    // function table but doesn't waste much space.
    iree::FunctionSourceMapDefBuilder fsmdb(*fbb_);
    return fsmdb.Finish();
  }

  LocationOffsetTable locationOffsetTable;
  std::vector<iree::BytecodeSourceLocation> bytecodeMap;
  for (const auto &offset_location : functionMap.locations) {
    int locationIndex =
        SerializeLocation(offset_location.second, &locationOffsetTable);
    bytecodeMap.push_back({offset_location.first, locationIndex});
  }
  auto locationTableOffset =
      fbb_->CreateVector(locationOffsetTable.locationDefs);
  auto bytecodeMapOffset = fbb_->CreateVectorOfStructs(bytecodeMap);

  iree::FunctionSourceMapDefBuilder fsmdb(*fbb_);
  fsmdb.add_location_table(locationTableOffset);
  fsmdb.add_bytecode_map(bytecodeMapOffset);
  return fsmdb.Finish();
}

int VMSourceMapBuilder::SerializeLocation(
    const Location &location, LocationOffsetTable *locationOffsetTable) {
  auto existingIt = locationOffsetTable->locationMap.find(location);
  if (existingIt != locationOffsetTable->locationMap.end()) {
    return existingIt->getSecond();
  }

  iree::LocationDefUnion locationUnionType;
  ::flatbuffers::Offset<void> locationUnionOffset;
  if (auto fileLoc = location.dyn_cast<FileLineColLoc>()) {
    locationUnionType = iree::LocationDefUnion::FileLocationDef;
    int filenameIndex = GetUniqueString(fileLoc.getFilename().str());
    iree::FileLocationDefBuilder lb(*fbb_);
    lb.add_filename(filenameIndex);
    lb.add_line(fileLoc.getLine());
    lb.add_column(fileLoc.getColumn());
    locationUnionOffset = lb.Finish().Union();
  } else if (auto nameLoc = location.dyn_cast<NameLoc>()) {
    locationUnionType = iree::LocationDefUnion::NameLocationDef;
    int nameIndex = GetUniqueString(nameLoc.getName().str());
    iree::NameLocationDefBuilder lb(*fbb_);
    lb.add_name(nameIndex);
    locationUnionOffset = lb.Finish().Union();
  } else if (auto callSiteLoc = location.dyn_cast<CallSiteLoc>()) {
    locationUnionType = iree::LocationDefUnion::CallSiteLocationDef;
    int calleeIndex =
        SerializeLocation(callSiteLoc.getCallee(), locationOffsetTable);
    int callerIndex =
        SerializeLocation(callSiteLoc.getCaller(), locationOffsetTable);
    iree::CallSiteLocationDefBuilder lb(*fbb_);
    lb.add_callee_location(calleeIndex);
    lb.add_caller_location(callerIndex);
    locationUnionOffset = lb.Finish().Union();
  } else if (auto fusedLoc = location.dyn_cast<FusedLoc>()) {
    locationUnionType = iree::LocationDefUnion::FusedLocationDef;
    std::vector<int> locationIndices;
    locationIndices.reserve(fusedLoc.getLocations().size());
    for (const auto &child_loc : fusedLoc.getLocations()) {
      int child_index = SerializeLocation(child_loc, locationOffsetTable);
      locationIndices.push_back(child_index);
    }
    auto locationIndicesOffset = fbb_->CreateVector(locationIndices);
    iree::FusedLocationDefBuilder lb(*fbb_);
    lb.add_locations(locationIndicesOffset);
    locationUnionOffset = lb.Finish().Union();
  } else {
    llvm_unreachable("Unimplemented location kind");
  }

  iree::LocationDefBuilder ldb(*fbb_);
  ldb.add_location_union_type(locationUnionType);
  ldb.add_location_union(locationUnionOffset);
  int locationIndex = locationOffsetTable->locationDefs.size();
  locationOffsetTable->locationDefs.push_back(ldb.Finish());
  locationOffsetTable->locationMap.insert({location, locationIndex});
  return locationIndex;
}

}  // namespace iree_compiler
}  // namespace mlir
