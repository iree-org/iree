//===- RegisterGemmini.h - Registration for Gemmini dialect & passes ------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_THIRDPARTY_BUDDY_GEMMINI_REGISTER_GEMMINI_H
#define IREE_COMPILER_THIRDPARTY_BUDDY_GEMMINI_REGISTER_GEMMINI_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"

namespace buddy {
namespace gemmini {
class GemminiDialect;
} // namespace gemmini
} // namespace buddy

namespace mlir {
namespace buddy {
// Forward declarations for pass registration functions
void registerLowerLinalgToGemminiPass();
void registerLowerGemminiPass();
void registerGemminiIRDumpsPass();
} // namespace buddy

namespace iree_compiler {

// Register Gemmini dialect
void registerGemminiDialect(DialectRegistry &registry);

// Register all Gemmini passes
void registerGemminiPasses();

} // namespace iree_compiler
} // namespace mlir

#endif // IREE_COMPILER_THIRDPARTY_BUDDY_GEMMINI_REGISTER_GEMMINI_H
