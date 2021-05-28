// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LLVM/LibraryBuilder.h"

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

//===----------------------------------------------------------------------===//
// iree/hal/local/executable_library.h structure types
//===----------------------------------------------------------------------===//
// The IR snippets below were pulled from clang running with `-S -emit-llvm`
// on the executable_library.h header: https://godbolt.org/z/6bMv5jfvf

// %struct.iree_hal_executable_import_table_v0_t = type {
//   i64,
//   i8*
// }
static llvm::StructType *makeImportTableType(llvm::LLVMContext &context) {
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_import_table_v0_t")) {
    return existingType;
  }
  auto *i8PtrType = llvm::IntegerType::getInt8PtrTy(context);
  auto *i64Type = llvm::IntegerType::getInt64Ty(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i64Type,
                                            i8PtrType,
                                        },
                                        "iree_hal_executable_import_table_v0_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.anon = type { i32, i32, i32 }
// %union.iree_hal_vec3_t = type { %struct.anon }
static llvm::StructType *makeVec3Type(llvm::LLVMContext &context) {
  if (auto *existingType =
          llvm::StructType::getTypeByName(context, "iree_hal_vec3_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *type = llvm::StructType::create(context,
                                        {
                                            i32Type,
                                            i32Type,
                                            i32Type,
                                        },
                                        "iree_hal_vec3_t",
                                        /*isPacked=*/false);
  return type;
}

// %struct.iree_hal_executable_dispatch_state_v0_t = type {
//   %union.iree_hal_vec3_t,
//   %union.iree_hal_vec3_t,
//   i64,
//   i32*,
//   i64,
//   i8**,
//   i64*,
//   %struct.iree_hal_executable_import_table_v0_t*
// }
static llvm::StructType *makeDispatchStateType(llvm::LLVMContext &context) {
  auto *type = llvm::StructType::getTypeByName(
      context, "iree_hal_executable_dispatch_state_v0_t");
  assert(type && "state type must be defined by ConvertToLLVM");
  return type;
}

// i32 (%struct.iree_hal_executable_dispatch_state_v0_t*,
//      %union.iree_hal_vec3_t*)
static llvm::FunctionType *makeDispatchFunctionType(
    llvm::LLVMContext &context) {
  auto *dispatchStateType = makeDispatchStateType(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *vec3Type = llvm::ArrayType::get(i32Type, 3);
  return llvm::FunctionType::get(i32Type,
                                 {
                                     dispatchStateType->getPointerTo(),
                                     vec3Type->getPointerTo(),
                                 },
                                 /*isVarArg=*/false);
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
//   i32,
//   i32 (%struct.iree_hal_executable_dispatch_state_v0_t*,
//        %union.iree_hal_vec3_t*)**,
//   i8**,
//   i8**
// }
static llvm::StructType *makeLibraryType(llvm::StructType *libraryHeaderType) {
  auto &context = libraryHeaderType->getContext();
  if (auto *existingType = llvm::StructType::getTypeByName(
          context, "iree_hal_executable_library_v0_t")) {
    return existingType;
  }
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  auto *dispatchFunctionType = makeDispatchFunctionType(context);
  auto *i8PtrType = llvm::IntegerType::getInt8PtrTy(context);
  auto *type = llvm::StructType::create(
      context,
      {
          libraryHeaderType->getPointerTo(),
          i32Type,
          dispatchFunctionType->getPointerTo()->getPointerTo(),
          i8PtrType->getPointerTo(),
          i8PtrType->getPointerTo(),
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
  auto *ptrType = llvm::Type::getInt8PtrTy(context);
  auto *libraryHeaderType = makeLibraryHeaderType(context);

  // %struct.iree_hal_executable_library_header_t**
  // @iree_hal_library_query(i32, void*)
  auto *queryFuncType =
      llvm::FunctionType::get(libraryHeaderType->getPointerTo(),
                              {
                                  i32Type,
                                  ptrType,
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
      builder.CreateICmpEQ(func->getArg(0), llvm::ConstantInt::get(i32Type, 0)),
      builder.CreatePointerCast(v0, libraryHeaderType->getPointerTo()),
      llvm::ConstantPointerNull::get(libraryHeaderType->getPointerTo())));

  return func;
}

llvm::Constant *LibraryBuilder::buildLibraryV0(std::string libraryName) {
  auto &context = module->getContext();
  auto *libraryHeaderType = makeLibraryHeaderType(context);
  auto *libraryType = makeLibraryType(libraryHeaderType);
  auto *dispatchFunctionType = makeDispatchFunctionType(context);
  auto *i8Type = llvm::IntegerType::getInt8Ty(context);
  auto *i32Type = llvm::IntegerType::getInt32Ty(context);
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);

  // ----- Header -----

  auto *libraryHeader = new llvm::GlobalVariable(
      *module, libraryHeaderType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantStruct::get(
          libraryHeaderType,
          {
              // version=
              llvm::ConstantInt::get(i32Type,
                                     static_cast<int64_t>(Version::V_0)),
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

  // ----- Entry points -----

  SmallVector<llvm::Constant *, 4> entryPointFuncValues;
  for (auto entryPoint : entryPoints) {
    entryPointFuncValues.push_back(entryPoint.func);
  }
  auto *entryPointFuncsType = llvm::ArrayType::get(
      dispatchFunctionType->getPointerTo(), entryPointFuncValues.size());
  llvm::Constant *entryPointFuncs = new llvm::GlobalVariable(
      *module, entryPointFuncsType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantArray::get(entryPointFuncsType, entryPointFuncValues),
      /*Name=*/libraryName + "_funcs");
  // TODO(benvanik): force alignment (16? natural pointer width *2?)
  entryPointFuncs = llvm::ConstantExpr::getInBoundsGetElementPtr(
      entryPointFuncsType, entryPointFuncs,
      ArrayRef<llvm::Constant *>{zero, zero});

  llvm::Constant *entryPointNames =
      llvm::Constant::getNullValue(i8Type->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *, 4> entryPointNameValues;
    for (auto entryPoint : entryPoints) {
      entryPointNameValues.push_back(
          getStringConstant(entryPoint.name, module));
    }
    auto *entryPointNamesType = llvm::ArrayType::get(
        i8Type->getPointerTo(), entryPointNameValues.size());
    auto *global = new llvm::GlobalVariable(
        *module, entryPointNamesType, /*isConstant=*/true,
        llvm::GlobalVariable::PrivateLinkage,
        llvm::ConstantArray::get(entryPointNamesType, entryPointNameValues),
        /*Name=*/libraryName + "_names");
    // TODO(benvanik): force alignment (16? natural pointer width *2?)

    entryPointNames = llvm::ConstantExpr::getInBoundsGetElementPtr(
        entryPointNamesType, global, ArrayRef<llvm::Constant *>{zero, zero});
  }

  llvm::Constant *entryPointTags =
      llvm::Constant::getNullValue(i8Type->getPointerTo());
  if (mode == Mode::INCLUDE_REFLECTION_ATTRS) {
    SmallVector<llvm::Constant *, 4> entryPointTagValues;
    for (auto entryPoint : entryPoints) {
      entryPointTagValues.push_back(getStringConstant(entryPoint.tag, module));
    }
    auto *entryPointTagsType = llvm::ArrayType::get(i8Type->getPointerTo(),
                                                    entryPointTagValues.size());
    auto *global = new llvm::GlobalVariable(
        *module, entryPointTagsType, /*isConstant=*/true,
        llvm::GlobalVariable::PrivateLinkage,
        llvm::ConstantArray::get(entryPointTagsType, entryPointTagValues),
        /*Name=*/libraryName + "_tags");
    // TODO(benvanik): force alignment (16? natural pointer width *2?)

    entryPointTags = llvm::ConstantExpr::getInBoundsGetElementPtr(
        entryPointTagsType, global, ArrayRef<llvm::Constant *>{zero, zero});
  }

  // ----- Library -----

  auto *library = new llvm::GlobalVariable(
      *module, libraryType, /*isConstant=*/true,
      llvm::GlobalVariable::PrivateLinkage,
      llvm::ConstantStruct::get(
          libraryType,
          {
              // header=
              libraryHeader,
              // entry_point_count=
              llvm::ConstantInt::get(i32Type, entryPoints.size()),
              // entry_points=
              entryPointFuncs,
              // entry_point_names=
              entryPointNames,
              // entry_point_tags=
              entryPointTags,
          }),
      /*Name=*/libraryName);
  // TODO(benvanik): force alignment (8? natural pointer width?)

  return library;
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
