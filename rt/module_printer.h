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

#ifndef IREE_RT_MODULE_PRINTER_H_
#define IREE_RT_MODULE_PRINTER_H_

#include <ostream>

#include "base/bitfield.h"
#include "base/status.h"
#include "rt/module.h"

namespace iree {
namespace rt {

enum class PrintModuleFlag {
  kNone = 0,
  kDisassemble = 1 << 0,
};
IREE_BITFIELD(PrintModuleFlag);
using PrintModuleFlagBitfield = PrintModuleFlag;

// Prints all functions within the module to the given |stream|.
Status PrintModuleToStream(const Module& module, std::ostream* stream);
Status PrintModuleToStream(const Module& module, PrintModuleFlagBitfield flags,
                           std::ostream* stream);

}  // namespace rt
}  // namespace iree

#endif  // IREE_RT_MODULE_PRINTER_H_
