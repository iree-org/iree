// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Utils/ExecutableDebugInfoUtils.h"

#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/schemas/executable_debug_info_builder.h"
#include "mlir/IR/DialectResourceBlobManager.h"

namespace mlir::iree_compiler::IREE::HAL {

flatbuffers_vec_ref_t createSourceFilesVec(int debugLevel,
                                           DictionaryAttr sourcesAttr,
                                           FlatbufferBuilder &fbb) {
  if (debugLevel < 1) {
    // No debug information.
    return 0;
  } else if (!sourcesAttr || sourcesAttr.empty()) {
    // No sources embedded in the IR.
    return 0;
  }
  SmallVector<iree_hal_debug_SourceFileDef_ref_t> sourceFileRefs;
  for (auto sourceAttr : llvm::reverse(sourcesAttr.getValue())) {
    if (auto resourceAttr = dyn_cast_if_present<DenseResourceElementsAttr>(
            sourceAttr.getValue())) {
      auto filenameRef = fbb.createString(sourceAttr.getName());
      auto contentRef = fbb.streamUint8Vec([&](llvm::raw_ostream &os) {
        auto blobData = resourceAttr.getRawHandle().getBlob()->getData();
        os.write(blobData.data(), blobData.size());
        return true;
      });
      sourceFileRefs.push_back(
          iree_hal_debug_SourceFileDef_create(fbb, filenameRef, contentRef));
    }
  }
  std::reverse(sourceFileRefs.begin(), sourceFileRefs.end());
  return fbb.createOffsetVecDestructive(sourceFileRefs);
}

SmallVector<flatbuffers_ref_t>
createExportDefs(int debugLevel,
                 ArrayRef<IREE::HAL::ExecutableExportOp> exportOps,
                 FlatbufferBuilder &fbb) {
  SmallVector<flatbuffers_ref_t> exportDefs;
  exportDefs.resize(exportOps.size(), 0);

  if (debugLevel < 1) {
    // No debug information.
    return exportDefs;
  }

  for (auto exportOp : exportOps) {
    auto ordinalAttr = exportOp.getOrdinalAttr();
    assert(ordinalAttr && "ordinals must be assigned");
    int64_t ordinal = ordinalAttr.getInt();

    flatbuffers_ref_t nameRef = 0;
    if (debugLevel >= 1) {
      nameRef = fbb.createString(exportOp.getName());
    }

    flatbuffers_ref_t locationRef = 0;
    if (debugLevel >= 1) {
      if (auto loc = findFirstFileLoc(exportOp.getLoc())) {
        auto filenameRef = fbb.createString(loc->getFilename());
        locationRef = iree_hal_debug_FileLineLocDef_create(fbb, filenameRef,
                                                           loc->getLine());
      }
    }

    flatbuffers_vec_ref_t stageLocationsRef = 0;
    if (debugLevel >= 3) {
      SmallVector<iree_hal_debug_StageLocationDef_ref_t> stageLocationRefs;
      if (auto locsAttr = exportOp.getSourceLocsAttr()) {
        for (auto locAttr : locsAttr.getValue()) {
          if (auto loc =
                  findFirstFileLoc(cast<LocationAttr>(locAttr.getValue()))) {
            auto stageNameRef = fbb.createString(locAttr.getName());
            auto filenameRef = fbb.createString(loc->getFilename());
            stageLocationRefs.push_back(iree_hal_debug_StageLocationDef_create(
                fbb, stageNameRef,
                iree_hal_debug_FileLineLocDef_create(fbb, filenameRef,
                                                     loc->getLine())));
          }
        }
      }
      if (!stageLocationRefs.empty()) {
        stageLocationsRef = fbb.createOffsetVecDestructive(stageLocationRefs);
      }
    }

    iree_hal_debug_ExportDef_start(fbb);
    iree_hal_debug_ExportDef_name_add(fbb, nameRef);
    iree_hal_debug_ExportDef_location_add(fbb, locationRef);
    iree_hal_debug_ExportDef_stage_locations_add(fbb, stageLocationsRef);
    exportDefs[ordinal] = iree_hal_debug_ExportDef_end(fbb);
  }

  return exportDefs;
}

} // namespace mlir::iree_compiler::IREE::HAL
