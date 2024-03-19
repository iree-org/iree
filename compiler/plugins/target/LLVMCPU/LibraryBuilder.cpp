// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/LLVMCPU/LibraryBuilder.h"

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

namespace mlir::iree_compiler::IREE::HAL {

static inline int64_t roundUpToAlignment(int64_t value, int64_t alignment) {
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
  auto *i8PtrType = llvm::PointerType::getUnqual(context);
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
static llvm::FunctionType *
makeDispatchFunctionType(llvm::LLVMContext &context) {
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

// %struct.iree_hal_executable_source_location_v0_t = type {
//   i32,
//   i32,
//   i8*
// }
static llvm::StructType *makeSourceLocationType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_source_location_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *i8PtrType = llvm::PointerType::getUnqual(context);
  auto *type =
      llvm::StructType::create(context,
                               {
                                   i32Type,
                                   i32Type,
                                   i8PtrType,
                               },
                               "iree_hal_executable_source_location_v0_t",
                               /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_stage_location_table_v0_t = type {
//   i32,
//   i8**,
//   %struct.iree_hal_executable_source_location_v0_t*,
// }
static llvm::StructType *
makeStageLocationTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_stage_location_table_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *i8PtrType = llvm::PointerType::getUnqual(context);
  auto *sourceLocationType = makeSourceLocationType(context);
  auto *type =
      llvm::StructType::create(context,
                               {
                                   i32Type,
                                   i8PtrType->getPointerTo(),
                                   sourceLocationType->getPointerTo(),
                               },
                               "iree_hal_executable_stage_location_table_v0_t",
                               /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_export_table_v0_t = type {
//   i32,
//   i32*,
//   %struct.iree_hal_executable_dispatch_attrs_v0_t*,
//   i8**,
//   i8**,
//   %struct.iree_hal_executable_source_location_v0_t*,
//   %struct.iree_hal_executable_stage_location_table_v0_t*,
// }
static llvm::StructType *makeExportTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_export_table_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *dispatchFunctionType = makeDispatchFunctionType(context);
  auto *dispatchAttrsType = makeDispatchAttrsType(context);
  auto *i8PtrType = llvm::PointerType::getUnqual(context);
  auto *sourceLocationType = makeSourceLocationType(context);
  auto *stageLocationTableType = makeStageLocationTableType(context);
  auto *type = llvm::StructType::create(
      context,
      {
          i32Type,
          dispatchFunctionType->getPointerTo()->getPointerTo(),
          dispatchAttrsType->getPointerTo(),
          i8PtrType->getPointerTo(),
          i8PtrType->getPointerTo(),
          sourceLocationType->getPointerTo(),
          stageLocationTableType->getPointerTo(),
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

// %struct.iree_hal_executable_source_file_v0_t = type {
//   i32,
//   i8*,
//   i32,
//   i8*
// }
static llvm::StructType *makeSourceFileType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_source_file_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *i8PtrType = llvm::PointerType::getUnqual(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type,
                                            i8PtrType,
                                            i32Type,
                                            i8PtrType,
                                        },
                                        "iree_hal_executable_source_file_v0_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_source_file_table_v0_t = type {
//   i32,
//   %struct.iree_hal_executable_source_file_v0_t*,
// }
static llvm::StructType *makeSourceTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_source_file_table_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *sourceFileType = makeSourceFileType(context);
  auto *type =
      llvm::StructType::create(context,
                               {
                                   i32Type,
                                   sourceFileType->getPointerTo(),
                               },
                               "iree_hal_executable_source_file_table_v0_t",
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
  auto *i8PtrType = llvm::PointerType::getUnqual(context);
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
  auto *sourceTableType = makeSourceTableType(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            libraryHeaderType->getPointerTo(),
                                            importTableType,
                                            exportTableType,
                                            constantTableType,
                                            sourceTableType,
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
static llvm::Constant *createStringConstant(StringRef value,
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

// Creates a global NUL-terminated string constant or NULL if the string is
// empty.
static llvm::Constant *createStringConstantOrNull(StringRef value,
                                                  llvm::Module *module) {
  if (value.empty()) {
    auto i8Type = llvm::IntegerType::getInt8Ty(module->getContext());
    return llvm::ConstantPointerNull::get(i8Type->getPointerTo());
  }
  return createStringConstant(value, module);
}

// Creates a global serialized buffer constant (or string without NUL).
//
// Example:
//   @.data = private unnamed_addr constant [5 x i8] c"lib_a", align 1
static llvm::Constant *createBufferConstant(StringRef name,
                                            ArrayRef<char> value,
                                            llvm::Module *module) {
  auto i8Type = llvm::IntegerType::getInt8Ty(module->getContext());
  auto i32Type = llvm::IntegerType::getInt32Ty(module->getContext());
  auto *bufferType = llvm::ArrayType::get(i8Type, value.size());
  auto *literal = llvm::ConstantDataArray::get(module->getContext(), value);
  auto *global = new llvm::GlobalVariable(
      *module, bufferType,
      /*isConstant=*/true, llvm::GlobalVariable::PrivateLinkage, literal, name);
  global->setAlignment(llvm::MaybeAlign(1));
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);
  return llvm::ConstantExpr::getInBoundsGetElementPtr(
      bufferType, global, ArrayRef<llvm::Constant *>{zero, zero});
}

// Creates a global constant with the given elements.
static llvm::Constant *createArrayConstant(StringRef name,
                                           llvm::Type *elementType,
                                           ArrayRef<llvm::Constant *> elements,
                                           llvm::Module *module) {
  auto *i32Type = llvm::IntegerType::getInt32Ty(module->getContext());
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);
  auto *arrayType = llvm::ArrayType::get(elementType, elements.size());
  auto *global = new llvm::GlobalVariable(
      *module, arrayType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantArray::get(arrayType, elements), name);
  return llvm::ConstantExpr::getInBoundsGetElementPtr(
      arrayType, global, ArrayRef<llvm::Constant *>{zero, zero});
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

llvm::Constant *
LibraryBuilder::buildLibraryV0ImportTable(std::string libraryName) {
  auto &context = module->getContext();
  auto *importTableType = makeImportTableType(context);
  auto *i8Type = llvm::IntegerType::getInt8Ty(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  llvm::Constant *symbolNames =
      llvm::Constant::getNullValue(i8Type->getPointerTo());
  if (!imports.empty()) {
    SmallVector<llvm::Constant *> symbolNameValues;
    for (auto &import : imports) {
      auto symbolName = import.symbol_name;
      if (import.weak)
        symbolName = "?" + symbolName;
      symbolNameValues.push_back(createStringConstant(symbolName, module));
    }
    symbolNames =
        createArrayConstant(libraryName + "_import_names",
                            i8Type->getPointerTo(), symbolNameValues, module);
  }
  return llvm::ConstantStruct::get(
      importTableType, {
                           // count=
                           llvm::ConstantInt::get(i32Type, imports.size()),
                           // symbols=
                           symbolNames,
                       });
}

llvm::Constant *
LibraryBuilder::buildLibraryV0ExportTable(std::string libraryName) {
  auto &context = module->getContext();
  auto *exportTableType = makeExportTableType(context);
  auto *dispatchFunctionType = makeDispatchFunctionType(context);
  auto *dispatchAttrsType = makeDispatchAttrsType(context);
  auto *sourceLocationType = makeSourceLocationType(context);
  auto *stageLocationTableType = makeStageLocationTableType(context);
  auto *i8Type = llvm::IntegerType::getInt8Ty(context);
  auto *i16Type = llvm::IntegerType::getInt16Ty(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);

  // iree_hal_executable_export_table_v0_t::ptrs
  SmallVector<llvm::Constant *> exportPtrValues;
  for (auto dispatch : exports)
    exportPtrValues.push_back(dispatch.func);
  llvm::Constant *exportPtrs = createArrayConstant(
      libraryName + "_funcs", dispatchFunctionType->getPointerTo(),
      exportPtrValues, module);

  // iree_hal_executable_export_table_v0_t::attrs
  llvm::Constant *exportAttrs =
      llvm::Constant::getNullValue(i32Type->getPointerTo());
  bool hasNonDefaultAttrs = llvm::any_of(exports, [](const auto &dispatch) {
    return !dispatch.attrs.isDefault();
  });
  if (!hasNonDefaultAttrs) {
    SmallVector<llvm::Constant *> exportAttrValues;
    for (auto dispatch : exports) {
      exportAttrValues.push_back(llvm::ConstantStruct::get(
          dispatchAttrsType,
          {
              // local_memory_pages=
              llvm::ConstantInt::get(
                  i16Type, roundUpToAlignment(dispatch.attrs.localMemorySize,
                                              kWorkgroupLocalMemoryPageSize) /
                               kWorkgroupLocalMemoryPageSize),
              // reserved=
              llvm::ConstantInt::get(i16Type, 0),
          }));
    }
    exportAttrs = createArrayConstant(libraryName + "_attrs", dispatchAttrsType,
                                      exportAttrValues, module);
  }

  // iree_hal_executable_export_table_v0_t::names
  llvm::Constant *exportNames =
      llvm::Constant::getNullValue(i8Type->getPointerTo()->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *> exportNameValues;
    for (auto dispatch : exports)
      exportNameValues.push_back(createStringConstant(dispatch.name, module));
    exportNames =
        createArrayConstant(libraryName + "_names", i8Type->getPointerTo(),
                            exportNameValues, module);
  }

  // iree_hal_executable_export_table_v0_t::tags
  llvm::Constant *exportTags =
      llvm::Constant::getNullValue(i8Type->getPointerTo()->getPointerTo());
  bool hasAnyTags = llvm::any_of(
      exports, [](auto &dispatch) { return !dispatch.tag.empty(); });
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS && hasAnyTags) {
    SmallVector<llvm::Constant *> exportTagValues;
    for (auto dispatch : exports)
      exportTagValues.push_back(
          createStringConstantOrNull(dispatch.tag, module));
    exportTags = createArrayConstant(
        libraryName + "_tags", i8Type->getPointerTo(), exportTagValues, module);
  }

  // iree_hal_executable_export_table_v0_t::source_locations
  llvm::Constant *exportSourceLocations =
      llvm::Constant::getNullValue(sourceLocationType->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *> exportSourceLocationValues;
    for (auto dispatch : exports) {
      exportSourceLocationValues.push_back(llvm::ConstantStruct::get(
          sourceLocationType,
          {
              // line=
              llvm::ConstantInt::get(i32Type, dispatch.sourceLocation.line),
              // path_length=
              llvm::ConstantInt::get(i32Type,
                                     dispatch.sourceLocation.path.size()),
              // path=
              createStringConstant(dispatch.sourceLocation.path, module),
          }));
    }
    exportSourceLocations = createArrayConstant(
        libraryName + "_source_locations", sourceLocationType,
        exportSourceLocationValues, module);
  }

  // iree_hal_executable_export_table_v0_t::stage_locations
  llvm::Constant *exportStageLocations =
      llvm::Constant::getNullValue(stageLocationTableType->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *> exportStageTableValues;
    for (auto dispatch : exports) {
      SmallVector<llvm::Constant *> exportStageNameValues;
      SmallVector<llvm::Constant *> exportSourceLocationValues;
      for (auto &stageLocation : dispatch.stageLocations) {
        exportStageNameValues.push_back(
            createStringConstant(stageLocation.stage, module));
        exportSourceLocationValues.push_back(llvm::ConstantStruct::get(
            sourceLocationType,
            {
                // line=
                llvm::ConstantInt::get(i32Type, stageLocation.line),
                // path_length=
                llvm::ConstantInt::get(i32Type, stageLocation.path.size()),
                // path=
                createStringConstant(stageLocation.path, module),
            }));
      }
      llvm::Constant *stageNamesPtr = createArrayConstant(
          libraryName + "_" + dispatch.name + "_stage_names",
          i8Type->getPointerTo(), exportStageNameValues, module);
      llvm::Constant *sourceLocationsPtr = createArrayConstant(
          libraryName + "_" + dispatch.name + "_stage_source_locations",
          sourceLocationType, exportSourceLocationValues, module);
      exportStageTableValues.push_back(llvm::ConstantStruct::get(
          stageLocationTableType,
          {
              // count=
              llvm::ConstantInt::get(i32Type, exportStageNameValues.size()),
              // names=
              stageNamesPtr,
              // locations=
              sourceLocationsPtr,
          }));
    }
    if (!exportStageTableValues.empty()) {
      exportStageLocations = createArrayConstant(
          libraryName + "_stage_location_tables", stageLocationTableType,
          exportStageTableValues, module);
    }
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
                           // source_locations=
                           exportSourceLocations,
                           // stage_locations=
                           exportStageLocations,
                       });
}

llvm::Constant *
LibraryBuilder::buildLibraryV0ConstantTable(std::string libraryName) {
  auto &context = module->getContext();
  auto *constantTableType = makeConstantTableType(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  return llvm::ConstantStruct::get(
      constantTableType, {
                             // count=
                             llvm::ConstantInt::get(i32Type, constantCount),
                         });
}

llvm::Constant *
LibraryBuilder::buildLibraryV0SourceTable(std::string libraryName) {
  auto &context = module->getContext();
  auto *sourceFileType = makeSourceFileType(context);
  auto *sourceTableType = makeSourceTableType(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  llvm::Constant *sourceFilesValue =
      llvm::Constant::getNullValue(sourceFileType->getPointerTo());
  if (!sourceFiles.empty()) {
    SmallVector<llvm::Constant *> sourceFileValues;
    for (auto &sourceFile : sourceFiles) {
      sourceFileValues.push_back(llvm::ConstantStruct::get(
          sourceFileType,
          {
              // path_length=
              llvm::ConstantInt::get(i32Type, sourceFile.path.size()),
              // path=
              createStringConstant(sourceFile.path, module),
              // content_length=
              llvm::ConstantInt::get(i32Type, sourceFile.contents.size()),
              // content=
              createBufferConstant(sourceFile.path, sourceFile.contents,
                                   module),
          }));
    }
    sourceFilesValue = createArrayConstant(
        libraryName + "_sources", sourceFileType, sourceFileValues, module);
  }
  return llvm::ConstantStruct::get(
      sourceTableType, {
                           // count=
                           llvm::ConstantInt::get(i32Type, sourceFiles.size()),
                           // files=
                           sourceFilesValue,
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
              createStringConstant(module->getName(), module),
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
                                    // sources=
                                    buildLibraryV0SourceTable(libraryName),
                                }),
      /*Name=*/libraryName);
  // TODO(benvanik): force alignment (8? natural pointer width?)

  return library;
}

} // namespace mlir::iree_compiler::IREE::HAL
