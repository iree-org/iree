// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Target/Bytecode/DebugDatabaseBuilder.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::iree_compiler::IREE::VM {

void DebugDatabaseBuilder::addFunctionSourceMap(IREE::VM::FuncOp funcOp,
                                                FunctionSourceMap sourceMap) {
  uint64_t ordinal = funcOp.getOrdinal().value_or(APInt(64, 0)).getZExtValue();
  if (functionSourceMaps.size() <= ordinal) {
    functionSourceMaps.resize(ordinal + 1);
  }
  functionSourceMaps[ordinal] = std::move(sourceMap);
}

struct LocationTable {
  explicit LocationTable(FlatbufferBuilder &fbb) : fbb(fbb) {}

  FlatbufferBuilder &fbb;

  // String table.
  DenseMap<StringRef, flatbuffers_string_ref_t> strings;
  // All serialized location entries.
  SmallVector<iree_vm_LocationTypeDef_union_ref_t> entries;
  // Map of uniqued location to the entry in the table.
  DenseMap<Location, int32_t> map;

  // Inserts a string into the location table string subtable if needed.
  flatbuffers_string_ref_t insert(StringRef value) {
    auto it = strings.find(value);
    if (it != strings.end())
      return it->second;
    auto stringRef = fbb.createString(value);
    strings[value] = stringRef;
    return stringRef;
  }

  // Inserts a location into the location table if it does not already exist.
  // Returns the ordinal of the location in the table.
  int32_t insert(Location baseLoc) {
    auto it = map.find(baseLoc);
    if (it != map.end())
      return it->second;
    auto locationRef =
        llvm::TypeSwitch<Location, iree_vm_LocationTypeDef_union_ref_t>(baseLoc)
            .Case([&](CallSiteLoc loc) {
              auto callee = insert(loc.getCallee());
              auto caller = insert(loc.getCaller());
              return iree_vm_LocationTypeDef_as_CallSiteLocDef(
                  iree_vm_CallSiteLocDef_create(fbb, callee, caller));
            })
            .Case([&](FileLineColLoc loc) {
              return iree_vm_LocationTypeDef_as_FileLineColLocDef(
                  iree_vm_FileLineColLocDef_create(
                      fbb, insert(loc.getFilename()), loc.getLine(),
                      loc.getColumn()));
            })
            .Case([&](FusedLoc loc) {
              flatbuffers_string_ref_t metadataRef = 0;
              if (loc.getMetadata()) {
                std::string str;
                llvm::raw_string_ostream os(str);
                loc.getMetadata().print(os);
                metadataRef = insert(os.str());
              }
              SmallVector<int32_t> childLocs;
              childLocs.reserve(loc.getLocations().size());
              for (auto childLoc : loc.getLocations()) {
                childLocs.push_back(insert(childLoc));
              }
              auto childLocsRef = flatbuffers_int32_vec_create(
                  fbb, childLocs.data(), childLocs.size());
              iree_vm_FusedLocDef_start(fbb);
              iree_vm_FusedLocDef_metadata_add(fbb, metadataRef);
              iree_vm_FusedLocDef_locations_add(fbb, childLocsRef);
              return iree_vm_LocationTypeDef_as_FusedLocDef(
                  iree_vm_FusedLocDef_end(fbb));
            })
            .Case([&](NameLoc loc) {
              return iree_vm_LocationTypeDef_as_NameLocDef(
                  iree_vm_NameLocDef_create(fbb, insert(loc.getName()),
                                            insert(loc.getChildLoc())));
            })
            .Default(
                [](Location loc) { return iree_vm_LocationTypeDef_as_NONE(); });
    int32_t ordinal = static_cast<int32_t>(entries.size());
    map[baseLoc] = ordinal;
    entries.push_back(locationRef);
    return ordinal;
  }

  iree_vm_LocationTypeDef_union_vec_ref_t finish() {
    return iree_vm_LocationTypeDef_vec_create(fbb, entries.data(),
                                              entries.size());
  }
};

iree_vm_DebugDatabaseDef_ref_t
DebugDatabaseBuilder::build(FlatbufferBuilder &fbb) {
  if (functionSourceMaps.empty())
    return 0;

  LocationTable locationTable(fbb);

  // functions:[FunctionSourceMapDef]
  SmallVector<iree_vm_FunctionSourceMapDef_ref_t> functionRefs;
  for (auto &sourceMap : functionSourceMaps) {
    SmallVector<iree_vm_BytecodeLocationDef_t> locationDefs;
    locationDefs.resize(sourceMap.locations.size());
    for (size_t i = 0; i < sourceMap.locations.size(); ++i) {
      locationDefs[i].bytecode_offset = sourceMap.locations[i].bytecodeOffset;
      locationDefs[i].location =
          locationTable.insert(sourceMap.locations[i].location);
    }
    auto locationsRef = iree_vm_BytecodeLocationDef_vec_create(
        fbb, locationDefs.data(), locationDefs.size());
    auto localNameRef = fbb.createString(sourceMap.localName);
    functionRefs.push_back(
        iree_vm_FunctionSourceMapDef_create(fbb, localNameRef, locationsRef));
  }
  auto functionsRef = iree_vm_FunctionSourceMapDef_vec_create(
      fbb, functionRefs.data(), functionRefs.size());

  // location_table:[LocationTypeDef]
  auto locationTableRef = locationTable.finish();

  // DebugDatabaseDef
  iree_vm_DebugDatabaseDef_start(fbb);
  iree_vm_DebugDatabaseDef_location_table_add(fbb, locationTableRef);
  iree_vm_DebugDatabaseDef_functions_add(fbb, functionsRef);
  return iree_vm_DebugDatabaseDef_end(fbb);
}

} // namespace mlir::iree_compiler::IREE::VM
