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

// This file defines a helper to trigger the registration of all translations
// in and out of MLIR to the system.
//
// Based on MLIR's InitAllTranslations but without translations we don't care
// about.

#ifndef IREE_TOOLS_INIT_TRANSLATIONS_H_
#define IREE_TOOLS_INIT_TRANSLATIONS_H_

#include "iree/compiler/Translation/IREEVM.h"

namespace mlir {

void registerToSPIRVTranslation();

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerMlirTranslations() {
  static bool init_once = []() {
    registerToSPIRVTranslation();
    return true;
  }();
  (void)init_once;
}

namespace iree_compiler {

// This function should be called before creating any MLIRContext if one
// expects all the possible translations to be made available to the context
// automatically.
inline void registerIreeTranslations() {
  static bool init_once = []() {
    registerIREEVMTranslation();
    return true;
  }();
  (void)init_once;
}

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_TOOLS_INIT_TRANSLATIONS_H_
