// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVMCPU/LibraryBuilder.h"

#include "llvm/IR/IRBuilder.h"

// =============================================================================
//
// NOTE: these structures model 1:1 those in iree/hal/local/executable_library.h
//
// This file must always track the latest version. Backwards compatibility with
// existing runtimes using older versions of the header is maintained by
// emitting variants of the structs matching those in the older headers and
// selecting between them in the query function based on the requested version.
//
// =============================================================================

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static inline int64_t RoundUpToAlignment(int64_t value, int64_t alignment) {
  return (value + (alignment - 1)) & ~(alignment - 1);
}

//===----------------------------------------------------------------------===//
// iree/hal/local/executable_library.h structure types
//===----------------------------------------------------------------------===//
// The IR snippets below were pulled from clang running with `-S -emit-llvm`
// on the executable_library.h header: https://godbolt.org/z/6bMv5jfvf

// %struct.iree_hal_executable_import_table_v0_t = type {
//   i32,
//   i8**
// }
static llvm::StructType *makeImportTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_import_table_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *i8PtrType = llvm::IntegerType::getInt8PtrTy(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type,
                                            i8PtrType->getPointerTo(),
                                        },
                                        "iree_hal_executable_import_table_v0_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_environment_v0_t = type {
//   ...
// }
static llvm::StructType *makeEnvironmentType(llvm::LLVMContext &context) {
  auto *type = llvm::StructType::getTypeByName(
      context, "iree_hal_executable_environment_v0_t");
  if (!type) {
    type = llvm::StructType::create(context,
                                    "iree_hal_executable_environment_v0_t");
  }
  return type;
}

// %struct.iree_hal_executable_dispatch_state_v0_t = type {
//   ...
// }
static llvm::StructType *makeDispatchStateType(llvm::LLVMContext &context) {
  auto *type = llvm::StructType::getTypeByName(
      context, "iree_hal_executable_dispatch_state_v0_t");
  if (!type) {
    type = llvm::StructType::create(context,
                                    "iree_hal_executable_dispatch_state_v0_t");
  }
  return type;
}

// %struct.iree_hal_executable_workgroup_state_v0_t = type {
//   ...
// }
static llvm::StructType *makeWorkgroupStateType(llvm::LLVMContext &context) {
  auto *type = llvm::StructType::getTypeByName(
      context, "iree_hal_executable_workgroup_state_v0_t");
  if (!type) {
    type = llvm::StructType::create(context,
                                    "iree_hal_executable_workgroup_state_v0_t");
  }
  return type;
}

// i32 (%struct.iree_hal_executable_environment_v0_t*,
//      %struct.iree_hal_executable_dispatch_state_v0_t*,
//      i8*)
static llvm::FunctionType *makeDispatchFunctionType(
    llvm::LLVMContext &context) {
  auto *environmentType = makeEnvironmentType(context);
  auto *dispatchStateType = makeDispatchStateType(context);
  auto *workgroupStateType = makeWorkgroupStateType(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  return llvm::FunctionType::get(i32Type,
                                 {
                                     environmentType->getPointerTo(),
                                     dispatchStateType->getPointerTo(),
                                     workgroupStateType->getPointerTo(),
                                 },
                                 /*isVarArg=*/false);
}

// %struct.iree_hal_executable_dispatch_attrs_v0_t = type {
//   i16,
//   i16
// }
static llvm::StructType *makeDispatchAttrsType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_dispatch_attrs_v0_t")) {
    return existingType;
  }
  auto *i16Type = llvm::IntegerType::getInt16Ty(context);
  auto *type =
      llvm::StructType::create(context,
                               {
                                   i16Type,
                                   i16Type,
                               },
                               "iree_hal_executable_dispatch_attrs_v0_t",
                               /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_src_loc_v0_t = type {
//   i32,
//   i32,
//   i8*
// }
static llvm::StructType *makeSrcLocType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_src_loc_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *i8PtrType = llvm::IntegerType::getInt8PtrTy(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type,
                                            i32Type,
                                            i8PtrType,
                                        },
                                        "iree_hal_executable_src_loc_v0_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_export_table_v0_t = type {
//   i32,
//   i32*,
//   %struct.iree_hal_executable_dispatch_attrs_v0_t*,
//   i8**,
//   i8**,
//   %struct.iree_hal_executable_src_loc_v0_t*,
// }
static llvm::StructType *makeExportTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_export_table_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *dispatchFunctionType = makeDispatchFunctionType(context);
  auto *dispatchAttrsType = makeDispatchAttrsType(context);
  auto *i8PtrType = llvm::IntegerType::getInt8PtrTy(context);
  auto *srcLocType = makeSrcLocType(context);
  auto *type = llvm::StructType::create(
      context,
      {
          i32Type,
          dispatchFunctionType->getPointerTo()->getPointerTo(),
          dispatchAttrsType->getPointerTo(),
          i8PtrType->getPointerTo(),
          i8PtrType->getPointerTo(),
          srcLocType->getPointerTo(),
      },
      "iree_hal_executable_export_table_v0_t",
      /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_constant_table_v0_t = type {
//   i32
// }
static llvm::StructType *makeConstantTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_constant_table_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *type =
      llvm::StructType::create(context,
                               {
                                   i32Type,
                               },
                               "iree_hal_executable_constant_table_v0_t",
                               /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_library_header_t = type {
//   i32,
//   i8*,
//   i32,
//   i32
// }
static llvm::StructType *makeLibraryHeaderType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_library_header_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *i8PtrType = llvm::IntegerType::getInt8PtrTy(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type,
                                            i8PtrType,
                                            i32Type,
                                            i32Type,
                                        },
                                        "iree_hal_executable_library_header_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_library_v0_t = type {
//   %struct.iree_hal_executable_library_header_t*,
//   %struct.iree_hal_executable_import_table_v0_t,
//   %struct.iree_hal_executable_export_table_v0_t,
// }
static llvm::StructType *makeLibraryType(llvm::StructType *libraryHeaderType) {
  auto &context = libraryHeaderType->getContext();
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_library_v0_t")) {
    return existingType;
  }
  auto *importTableType = makeImportTableType(context);
  auto *exportTableType = makeExportTableType(context);
  auto *constantTableType = makeConstantTableType(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            libraryHeaderType->getPointerTo(),
                                            importTableType,
                                            exportTableType,
                                            constantTableType,
                                        },
                                        "iree_hal_executable_library_v0_t",
                                        /*isPacked=*/false);
  return type;
}

//===----------------------------------------------------------------------===//
// IR construction utilities
//===----------------------------------------------------------------------===//

// Creates a global NUL-terminated string constant.
//
// Example:
//   @.str.2 = private unnamed_addr constant [6 x i8] c"lib_a\00", align 1
static llvm::Constant *getStringConstant(StringRef value,
                                         llvm::Module *module) {
  auto i8Type = llvm::IntegerType::getInt8Ty(module->getContext());
  auto i32Type = llvm::IntegerType::getInt32Ty(module->getContext());
  auto *stringType = llvm::ArrayType::get(i8Type, value.size() + /*NUL*/ 1);
  auto *literal =
      llvm::ConstantDataArray::getString(module->getContext(), value);
  auto *global = new llvm::GlobalVariable(*module, stringType,
                                          /*isConstant=*/true,
                                          llvm::GlobalVariable::PrivateLinkage,
                                          literal, /*Name=*/"");
  global->setAlignment(llvm::MaybeAlign(1));
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);
  return llvm::ConstantExpr::getInBoundsGetElementPtr(
      stringType, global, ArrayRef<llvm::Constant *>{zero, zero});
}

//===----------------------------------------------------------------------===//
// Builder interface
//===----------------------------------------------------------------------===//

llvm::Function *LibraryBuilder::build(StringRef queryFuncName) {
  auto &context = module->getContext();
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *environmentType = makeEnvironmentType(context)->getPointerTo();
  auto *libraryHeaderType = makeLibraryHeaderType(context);

  // %struct.iree_hal_executable_library_header_t**
  // @iree_hal_library_query(i32, %struct.iree_hal_executable_environment_v0_t*)
  auto *queryFuncType =
      llvm::FunctionType::get(libraryHeaderType->getPointerTo(),
                              {
                                  i32Type,
                                  environmentType,
                              },
                              /*isVarArg=*/false);
  auto *func =
      llvm::Function::Create(queryFuncType, llvm::GlobalValue::InternalLinkage,
                             queryFuncName, *module);

  auto *entryBlock = llvm::BasicBlock::Create(context, "entry", func);
  llvm::IRBuilder<> builder(entryBlock);

  // Build out the header for each version and select it at runtime.
  // NOTE: today there is just one version so this is rather simple:
  //   return max_version == 0 ? &library : NULL;
  auto *v0 = buildLibraryV0((queryFuncName + "_v0").str());
  builder.CreateRet(builder.CreateSelect(
      builder.CreateICmpEQ(func->getArg(0),
                           llvm::ConstantInt::get(
                               i32Type, static_cast<int64_t>(Version::LATEST))),
      builder.CreatePointerCast(v0, libraryHeaderType->getPointerTo()),
      llvm::ConstantPointerNull::get(libraryHeaderType->getPointerTo())));

  return func;
}

llvm::Constant *LibraryBuilder::buildLibraryV0ImportTable(
    std::string libraryName) {
  auto &context = module->getContext();
  auto *importTableType = makeImportTableType(context);
  auto *i8Type = llvm::IntegerType::getInt8Ty(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);

  llvm::Constant *symbolNames =
      llvm::Constant::getNullValue(i8Type->getPointerTo());
  if (!imports.empty()) {
    SmallVector<llvm::Constant *, 4> symbolNameValues;
    for (auto &import : imports) {
      auto symbolName = import.symbol_name;
      if (import.weak) {
        symbolName = "?" + symbolName;
      }
      symbolNameValues.push_back(getStringConstant(symbolName, module));
    }
    auto *symbolNamesType =
        llvm::ArrayType::get(i8Type->getPointerTo(), symbolNameValues.size());
    auto *global = new llvm::GlobalVariable(
        *module, symbolNamesType, /*isConstant=*/true,
        llvm::GlobalVariable::PrivateLinkage,
        llvm::ConstantArray::get(symbolNamesType, symbolNameValues),
        /*Name=*/libraryName + "_import_names");
    symbolNames = llvm::ConstantExpr::getInBoundsGetElementPtr(
        symbolNamesType, global, ArrayRef<llvm::Constant *>{zero, zero});
  }

  return llvm::ConstantStruct::get(
      importTableType, {
                           // count=
                           llvm::ConstantInt::get(i32Type, imports.size()),
                           // symbols=
                           symbolNames,
                       });
}

llvm::Constant *LibraryBuilder::buildLibraryV0ExportTable(
    std::string libraryName) {
  auto &context = module->getContext();
  auto *exportTableType = makeExportTableType(context);
  auto *dispatchFunctionType = makeDispatchFunctionType(context);
  auto *dispatchAttrsType = makeDispatchAttrsType(context);
  auto *srcLocType = makeSrcLocType(context);
  auto *i8Type = llvm::IntegerType::getInt8Ty(context);
  auto *i16Type = llvm::IntegerType::getInt16Ty(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);

  // iree_hal_executable_export_table_v0_t::ptrs
  SmallVector<llvm::Constant *, 4> exportPtrValues;
  for (auto dispatch : exports) {
    exportPtrValues.push_back(dispatch.func);
  }
  auto *exportPtrsType = llvm::ArrayType::get(
      dispatchFunctionType->getPointerTo(), exportPtrValues.size());
  llvm::Constant *exportPtrs = new llvm::GlobalVariable(
      *module, exportPtrsType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantArray::get(exportPtrsType, exportPtrValues),
      /*Name=*/libraryName + "_funcs");
  // TODO(benvanik): force alignment (16? natural pointer width *2?)
  exportPtrs = llvm::ConstantExpr::getInBoundsGetElementPtr(
      exportPtrsType, exportPtrs, ArrayRef<llvm::Constant *>{zero, zero});

  // iree_hal_executable_export_table_v0_t::attrs
  llvm::Constant *exportAttrs =
      llvm::Constant::getNullValue(i32Type->getPointerTo());
  bool hasNonDefaultAttrs =
      llvm::find_if(exports, [](const Dispatch &dispatch) {
        return !dispatch.attrs.isDefault();
      }) != exports.end();
  if (!hasNonDefaultAttrs) {
    SmallVector<llvm::Constant *, 4> exportAttrValues;
    for (auto dispatch : exports) {
      exportAttrValues.push_back(llvm::ConstantStruct::get(
          dispatchAttrsType,
          {
              // local_memory_pages=
              llvm::ConstantInt::get(
                  i16Type, RoundUpToAlignment(dispatch.attrs.localMemorySize,
                                              kWorkgroupLocalMemoryPageSize) /
                               kWorkgroupLocalMemoryPageSize),
              // reserved=
              llvm::ConstantInt::get(i16Type, 0),
          }));
    }
    auto *exportAttrsType =
        llvm::ArrayType::get(dispatchAttrsType, exportAttrValues.size());
    auto *global = new llvm::GlobalVariable(
        *module, exportAttrsType, /*isConstant=*/true,
        llvm::GlobalVariable::PrivateLinkage,
        llvm::ConstantArray::get(exportAttrsType, exportAttrValues),
        /*Name=*/libraryName + "_attrs");
    // TODO(benvanik): force alignment (16? natural pointer width?)
    exportAttrs = llvm::ConstantExpr::getInBoundsGetElementPtr(
        exportAttrsType, global, ArrayRef<llvm::Constant *>{zero, zero});
  }

  // iree_hal_executable_export_table_v0_t::names
  llvm::Constant *exportNames =
      llvm::Constant::getNullValue(i8Type->getPointerTo()->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *, 4> exportNameValues;
    for (auto dispatch : exports) {
      exportNameValues.push_back(getStringConstant(dispatch.name, module));
    }
    auto *exportNamesType =
        llvm::ArrayType::get(i8Type->getPointerTo(), exportNameValues.size());
    auto *global = new llvm::GlobalVariable(
        *module, exportNamesType, /*isConstant=*/true,
        llvm::GlobalVariable::PrivateLinkage,
        llvm::ConstantArray::get(exportNamesType, exportNameValues),
        /*Name=*/libraryName + "_names");
    // TODO(benvanik): force alignment (16? natural pointer width *2?)
    exportNames = llvm::ConstantExpr::getInBoundsGetElementPtr(
        exportNamesType, global, ArrayRef<llvm::Constant *>{zero, zero});
  }

  // iree_hal_executable_export_table_v0_t::tags
  llvm::Constant *exportTags =
      llvm::Constant::getNullValue(i8Type->getPointerTo()->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *, 4> exportTagValues;
    for (auto dispatch : exports) {
      exportTagValues.push_back(getStringConstant(dispatch.tag, module));
    }
    auto *exportTagsType =
        llvm::ArrayType::get(i8Type->getPointerTo(), exportTagValues.size());
    auto *global = new llvm::GlobalVariable(
        *module, exportTagsType, /*isConstant=*/true,
        llvm::GlobalVariable::PrivateLinkage,
        llvm::ConstantArray::get(exportTagsType, exportTagValues),
        /*Name=*/libraryName + "_tags");
    // TODO(benvanik): force alignment (16? natural pointer width *2?)
    exportTags = llvm::ConstantExpr::getInBoundsGetElementPtr(
        exportTagsType, global, ArrayRef<llvm::Constant *>{zero, zero});
  }

  // iree_hal_executable_export_table_v0_t::src_locs
  llvm::Constant *exportSrcLocs =
      llvm::Constant::getNullValue(srcLocType->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *, 4> exportSrcLocValues;
    for (auto dispatch : exports) {
      exportSrcLocValues.push_back(llvm::ConstantStruct::get(
          srcLocType,
          {
              // line=
              llvm::ConstantInt::get(i32Type, dispatch.sourceLoc),
              // path_length=
              llvm::ConstantInt::get(i32Type, dispatch.sourceFile.length()),
              // path=
              getStringConstant(dispatch.sourceFile, module),
          }));
    }
    auto *exportSrcLocsType =
        llvm::ArrayType::get(srcLocType, exportSrcLocValues.size());
    auto *global = new llvm::GlobalVariable(
        *module, exportSrcLocsType, /*isConstant=*/true,
        llvm::GlobalVariable::PrivateLinkage,
        llvm::ConstantArray::get(exportSrcLocsType, exportSrcLocValues),
        /*Name=*/libraryName + "_src_locs");
    // TODO(benvanik): force alignment (16? natural pointer width?)
    exportSrcLocs = llvm::ConstantExpr::getInBoundsGetElementPtr(
        exportSrcLocsType, global, ArrayRef<llvm::Constant *>{zero, zero});
  }

  return llvm::ConstantStruct::get(
      exportTableType, {
                           // count=
                           llvm::ConstantInt::get(i32Type, exports.size()),
                           // ptrs=
                           exportPtrs,
                           // attrs=
                           exportAttrs,
                           // names=
                           exportNames,
                           // tags=
                           exportTags,
                           // src_locs=
                           exportSrcLocs,
                       });
}

llvm::Constant *LibraryBuilder::buildLibraryV0ConstantTable(
    std::string libraryName) {
  auto &context = module->getContext();
  auto *constantTableType = makeConstantTableType(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);

  return llvm::ConstantStruct::get(
      constantTableType, {
                             // count=
                             llvm::ConstantInt::get(i32Type, constantCount),
                         });
}

llvm::Constant *LibraryBuilder::buildLibraryV0(std::string libraryName) {
  auto &context = module->getContext();
  auto *libraryHeaderType = makeLibraryHeaderType(context);
  auto *libraryType = makeLibraryType(libraryHeaderType);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);

  // ----- Header -----

  auto *libraryHeader = new llvm::GlobalVariable(
      *module, libraryHeaderType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantStruct::get(
          libraryHeaderType,
          {
              // version=
              llvm::ConstantInt::get(i32Type,
                                     static_cast<int64_t>(Version::LATEST)),
              // name=
              getStringConstant(module->getName(), module),
              // features=
              llvm::ConstantInt::get(i32Type, static_cast<int64_t>(features)),
              // sanitizer=
              llvm::ConstantInt::get(i32Type,
                                     static_cast<int64_t>(sanitizerKind)),
          }),
      /*Name=*/libraryName + "_header");
  // TODO(benvanik): force alignment (8? natural pointer width?)

  // ----- Library -----

  auto *library = new llvm::GlobalVariable(
      *module, libraryType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantStruct::get(libraryType,
                                {
                                    // header=
                                    libraryHeader,
                                    // imports=
                                    buildLibraryV0ImportTable(libraryName),
                                    // exports=
                                    buildLibraryV0ExportTable(libraryName),
                                    // constants=
                                    buildLibraryV0ConstantTable(libraryName),
                                }),
      /*Name=*/libraryName);
  // TODO(benvanik): force alignment (8? natural pointer width?)

  return library;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
