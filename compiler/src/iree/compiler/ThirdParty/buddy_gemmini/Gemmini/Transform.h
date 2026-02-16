//===- Transform.h - MLIR Dialect for RISC-V Gemmmini extension ---------===//
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

#ifndef GEMMINI_TRANSLATE_H
#define GEMMINI_TRANSLATE_H
#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_EX 0
#define CONFIG_BERT 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define MVIN_SCALE_IDENTITY 1.0
#define ACC_SCALE_IDENTITY 1.0
#define BANK_NUM 4
#define MAX_BYTES 64
#define HAS_FIRST_LAYER_OPTIMIZATIONS

typedef uint32_t acc_scale_t_bits;
typedef float acc_scale_t;
typedef uint32_t scale_t_bits;
typedef float scale_t;
typedef int32_t scale_acc_t;

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

void populateGemminiLegalizeForLLVMExportPatterns(
    LLVMTypeConverter &converter, RewritePatternSet &patterns, int64_t dim,
    int64_t addrLen, int64_t accRows, int64_t bankRows, size_t sizeOfElemT,
    size_t sizeOfAccT);
void configureGemminiLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // GEMMINI_TRANSLATE_H
