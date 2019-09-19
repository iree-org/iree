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

// Opcode table for the V0 binary format.
// Additions are fine but changing the behavior or order of any opcodes will
// break pasring of existing files.
//
// Opcodes have been selected on frequency of use, general applicability, and
// relative stability. Experimental ops should be implemented via the FFI fisrt
// before graduating into the core set. Ops that may only be present on certain
// targets should also be kept as imports via the FFI.
//
// Opcodes may be specified for particular types (int32_t), categories of types
// (all floating-point types), or implicit types (output matches input). Saving
// opcode space by sharing a single opcode for multiple types is preferred
// except where hot operations are performed (for example, comparison used in
// loop iteratosr).

#ifndef IREE_SCHEMAS_BYTECODE_BYTECODE_V0_H_
#define IREE_SCHEMAS_BYTECODE_BYTECODE_V0_H_

#include <cstdint>

#include "iree/base/bitfield.h"

namespace iree {

#define IREE_CONSTANT_ENCODING_LIST(ENC) \
  ENC(0x00, kDense, "dense")             \
  ENC(0x01, kSplat, "splat")

#define IREE_TYPE_LIST(TYP)                      \
  TYP(0x00, kI8, "i8", 1)                        \
  TYP(0x01, kI16, "i16", 2)                      \
  TYP(0x02, kI32, "i32", 4)                      \
  TYP(0x03, kI64, "i64", 8)                      \
  TYP(0x04, kF16, "f16", 2)                      \
  TYP(0x05, kF32, "f32", 4)                      \
  TYP(0x06, kF64, "f64", 8)                      \
  TYP(0x80, kDevice, "device", 0)                \
  TYP(0x81, kCommandBuffer, "command_buffer", 0) \
  TYP(0x82, kEvent, "event", 0)                  \
  TYP(0x83, kSemaphore, "semaphore", 0)          \
  TYP(0x84, kFence, "fence", 0)                  \
  TYP(0xFF, kOpaque, "opaque", 0)

#define IREE_CMPI_PREDICATE_LIST(PRED) \
  PRED(0, kEq, "eq")                   \
  PRED(1, kNe, "ne")                   \
  PRED(2, kSlt, "slt")                 \
  PRED(3, kSle, "sle")                 \
  PRED(4, kSgt, "sgt")                 \
  PRED(5, kSge, "sge")                 \
  PRED(6, kUlt, "ult")                 \
  PRED(7, kUle, "ule")                 \
  PRED(8, kUgt, "ugt")                 \
  PRED(9, kUge, "uge")

#define IREE_CMPF_PREDICATE_LIST(PRED) \
  PRED(0, kFalse, "false")             \
  PRED(1, kOeq, "oeq")                 \
  PRED(2, kOgt, "ogt")                 \
  PRED(3, kOge, "oge")                 \
  PRED(4, kOlt, "olt")                 \
  PRED(5, kOle, "ole")                 \
  PRED(6, kOne, "one")                 \
  PRED(7, kOrd, "ord")                 \
  PRED(8, kUeq, "ueq")                 \
  PRED(9, kUgt, "ugt")                 \
  PRED(10, kUge, "uge")                \
  PRED(11, kUlt, "ult")                \
  PRED(12, kUle, "ule")                \
  PRED(13, kUne, "une")                \
  PRED(14, kUno, "uno")                \
  PRED(15, kTrue, "true")

// NOTE: FF is a to-be-defined flag value for encoding/decoding.
#define FLAG(V) ::iree::OpcodeFlag::V

#define RSV(opcode, RESERVED_OPC)                                             \
  RESERVED_OPC(opcode, kReserved##opcode, "rsv." #opcode, FLAG(kDefault), "", \
               FF)

#define DECLARE_ENUM(ordinal, enum_name, ...) enum_name = ordinal,

enum class ConstantEncoding : uint8_t {
  IREE_CONSTANT_ENCODING_LIST(DECLARE_ENUM)
};

enum class BuiltinType : uint8_t { IREE_TYPE_LIST(DECLARE_ENUM) };

enum class CmpIPredicate : uint8_t { IREE_CMPI_PREDICATE_LIST(DECLARE_ENUM) };

enum class CmpFPredicate : uint8_t { IREE_CMPF_PREDICATE_LIST(DECLARE_ENUM) };

#undef DECLARE_ENUM

static constexpr uint8_t kBuiltinTypeCount =
    static_cast<uint8_t>(BuiltinType::kF64) + 1;

enum class OpcodeFlag : uint8_t {
  kDefault = 0,
};
IREE_BITFIELD(OpcodeFlag);
using OpcodeFlagBitfield = OpcodeFlag;

enum class OperandEncoding : char {
  kNone = '\0',
  kInputSlot = 's',
  kVariadicInputSlots = 'S',
  kOutputSlot = 'o',
  kVariadicOutputSlots = 'O',
  kResultSlot = 'r',
  kVariadicResultSlots = 'R',
  kVariadicTransferSlots = 'T',
  kConstant = 'c',
  kFunctionOrdinal = 'f',
  kImportOrdinal = 'F',
  kDispatchOrdinal = 'd',
  kBlockOffset = 'b',
  kTypeIndex = 't',
  kIndex = 'i',
  kIndexList = 'I',
  kCmpIPredicate = 'p',
  kCmpFPredicate = 'P',
};
IREE_BITFIELD(OperandEncoding);
using OperandEncodingBitfield = OperandEncoding;

}  // namespace iree

#endif  // IREE_SCHEMAS_BYTECODE_BYTECODE_V0_H_
