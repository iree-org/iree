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
// break parsing of existing files.
//
// Opcodes have been selected on frequency of use, general applicability, and
// relative stability. Experimental ops should be implemented via the Foreign
// Function Interface (FFI) first before graduating into the core set. Ops that
// may only be present on certain targets should also be kept as imports via the
// FFI.
//
// Opcodes may be specified for particular types (int32_t), categories of types
// (all floating-point types), or implicit types (output matches input). Saving
// opcode space by sharing a single opcode for multiple types is preferred
// except where hot operations are performed (for example, comparison used in
// loop iterators).

#ifndef IREE_SCHEMAS_BYTECODE_INTERPRETER_BYTECODE_V0_H_
#define IREE_SCHEMAS_BYTECODE_INTERPRETER_BYTECODE_V0_H_

#include "schemas/bytecode/bytecode_v0.h"

namespace iree {

#define IREE_INTERPRETER_OPCODE_LIST(OPC, RESERVED_OPC)                       \
  OPC(0x00, kConstant, "constant", FLAG(kDefault), "cr", FF)                  \
                                                                              \
  OPC(0x01, kCall, "call", FLAG(kDefault), "fSR", FF)                         \
  OPC(0x02, kCallImport, "call_import", FLAG(kDefault), "FSR", FF)            \
  OPC(0x03, kCallIndirect, "call_indirect", FLAG(kDefault), "tsSR", FF)       \
  OPC(0x04, kReturn, "return", FLAG(kDefault), "S", FF)                       \
  OPC(0x05, kBranch, "br", FLAG(kDefault), "bT", FF)                          \
  OPC(0x06, kCondBranch, "cond_br", FLAG(kDefault), "sbTbT", FF)              \
  OPC(0x07, kCmpI, "cmp_i", FLAG(kDefault), "psso", FF)                       \
  OPC(0x08, kCmpF, "cmp_f", FLAG(kDefault), "Psso", FF)                       \
                                                                              \
  RSV(0x09, RESERVED_OPC)                                                     \
  RSV(0x0A, RESERVED_OPC)                                                     \
  RSV(0x0B, RESERVED_OPC)                                                     \
  RSV(0x0C, RESERVED_OPC)                                                     \
  RSV(0x0D, RESERVED_OPC)                                                     \
  RSV(0x0E, RESERVED_OPC)                                                     \
  RSV(0x0F, RESERVED_OPC)                                                     \
  RSV(0x10, RESERVED_OPC)                                                     \
  RSV(0x11, RESERVED_OPC)                                                     \
  RSV(0x12, RESERVED_OPC)                                                     \
  RSV(0x13, RESERVED_OPC)                                                     \
  RSV(0x14, RESERVED_OPC)                                                     \
  RSV(0x15, RESERVED_OPC)                                                     \
  RSV(0x16, RESERVED_OPC)                                                     \
  RSV(0x17, RESERVED_OPC)                                                     \
  RSV(0x18, RESERVED_OPC)                                                     \
  RSV(0x19, RESERVED_OPC)                                                     \
  RSV(0x1A, RESERVED_OPC)                                                     \
  RSV(0x1B, RESERVED_OPC)                                                     \
  RSV(0x1C, RESERVED_OPC)                                                     \
  RSV(0x1D, RESERVED_OPC)                                                     \
  RSV(0x1E, RESERVED_OPC)                                                     \
  RSV(0x1F, RESERVED_OPC)                                                     \
                                                                              \
  OPC(0x20, kAllocStatic, "alloc_static", FLAG(kDefault), "Icr", FF)          \
  OPC(0x21, kAllocStack, "alloc_stack", FLAG(kDefault), "itISr", FF)          \
  OPC(0x22, kAllocStackInit, "alloc_stack_init", FLAG(kDefault), "tIScr", FF) \
  OPC(0x23, kAllocHeap, "alloc_heap", FLAG(kDefault), "itISr", FF)            \
  OPC(0x24, kDiscard, "discard", FLAG(kDefault), "s", FF)                     \
                                                                              \
  RSV(0x25, RESERVED_OPC)                                                     \
  RSV(0x26, RESERVED_OPC)                                                     \
  RSV(0x27, RESERVED_OPC)                                                     \
  RSV(0x28, RESERVED_OPC)                                                     \
  RSV(0x29, RESERVED_OPC)                                                     \
  RSV(0x2A, RESERVED_OPC)                                                     \
  RSV(0x2B, RESERVED_OPC)                                                     \
  RSV(0x2C, RESERVED_OPC)                                                     \
  RSV(0x2D, RESERVED_OPC)                                                     \
  RSV(0x2E, RESERVED_OPC)                                                     \
  RSV(0x2F, RESERVED_OPC)                                                     \
                                                                              \
  OPC(0x30, kRank, "rank", FLAG(kDefault), "so", FF)                          \
  OPC(0x31, kDim, "dim", FLAG(kDefault), "iso", FF)                           \
  OPC(0x32, kShape, "shape", FLAG(kDefault), "so", FF)                        \
  OPC(0x33, kLength, "length", FLAG(kDefault), "so", FF)                      \
  OPC(0x34, kDynamicSlice, "dynamic_slice", FLAG(kDefault), "sssr", FF)       \
  OPC(0x35, kStaticSlice, "static_slice", FLAG(kDefault), "sIIr", FF)         \
  OPC(0x36, kDynamicCopy, "dynamic_copy", FLAG(kDefault), "ssoss", FF)        \
  OPC(0x37, kStaticCopy, "static_copy", FLAG(kDefault), "sIoII", FF)          \
  OPC(0x38, kClone, "clone", FLAG(kDefault), "sr", FF)                        \
  RSV(0x39, RESERVED_OPC)                                                     \
  OPC(0x3A, kSplit, "split", FLAG(kDefault), "isR", FF)                       \
  OPC(0x3B, kAssign, "assign", FLAG(kDefault), "sr", FF)                      \
  OPC(0x3C, kCondAssign, "cond_assign", FLAG(kDefault), "sssr", FF)           \
  OPC(0x3D, kReshape, "reshape", FLAG(kDefault), "ssr", FF)                   \
  OPC(0x3E, kSelect, "select", FLAG(kDefault), "ssso", FF)                    \
  OPC(0x3F, kTranspose, "transpose", FLAG(kDefault), "sso", FF)               \
  OPC(0x40, kBroadcast, "broadcast", FLAG(kDefault), "sso", FF)               \
  OPC(0x41, kTile, "tile", FLAG(kDefault), "sso", FF)                         \
  OPC(0x42, kReverse, "reverse", FLAG(kDefault), "sso", FF)                   \
  OPC(0x43, kPad, "pad", FLAG(kDefault), "ssssso", FF)                        \
                                                                              \
  RSV(0x44, RESERVED_OPC)                                                     \
  RSV(0x45, RESERVED_OPC)                                                     \
  RSV(0x46, RESERVED_OPC)                                                     \
  RSV(0x47, RESERVED_OPC)                                                     \
  RSV(0x48, RESERVED_OPC)                                                     \
  RSV(0x49, RESERVED_OPC)                                                     \
  RSV(0x4A, RESERVED_OPC)                                                     \
  RSV(0x4B, RESERVED_OPC)                                                     \
  RSV(0x4C, RESERVED_OPC)                                                     \
  RSV(0x4D, RESERVED_OPC)                                                     \
  RSV(0x4E, RESERVED_OPC)                                                     \
  RSV(0x4F, RESERVED_OPC)                                                     \
                                                                              \
  OPC(0x50, kNot, "not", FLAG(kDefault), "so", FF)                            \
  OPC(0x51, kAnd, "and", FLAG(kDefault), "sso", FF)                           \
  OPC(0x52, kOr, "or", FLAG(kDefault), "sso", FF)                             \
  OPC(0x53, kXor, "xor", FLAG(kDefault), "sso", FF)                           \
  OPC(0x54, kShiftLeft, "sll", FLAG(kDefault), "sso", FF)                     \
  OPC(0x55, kShiftRightLogical, "srl", FLAG(kDefault), "sso", FF)             \
  OPC(0x56, kShiftRightArithmetic, "sra", FLAG(kDefault), "sso", FF)          \
                                                                              \
  RSV(0x57, RESERVED_OPC)                                                     \
  RSV(0x58, RESERVED_OPC)                                                     \
  RSV(0x59, RESERVED_OPC)                                                     \
  RSV(0x5A, RESERVED_OPC)                                                     \
  RSV(0x5B, RESERVED_OPC)                                                     \
  RSV(0x5C, RESERVED_OPC)                                                     \
  RSV(0x5D, RESERVED_OPC)                                                     \
  RSV(0x5E, RESERVED_OPC)                                                     \
  RSV(0x5F, RESERVED_OPC)                                                     \
  RSV(0x60, RESERVED_OPC)                                                     \
  RSV(0x61, RESERVED_OPC)                                                     \
  RSV(0x62, RESERVED_OPC)                                                     \
  RSV(0x63, RESERVED_OPC)                                                     \
  RSV(0x64, RESERVED_OPC)                                                     \
  RSV(0x65, RESERVED_OPC)                                                     \
  RSV(0x66, RESERVED_OPC)                                                     \
  RSV(0x67, RESERVED_OPC)                                                     \
  RSV(0x68, RESERVED_OPC)                                                     \
  RSV(0x69, RESERVED_OPC)                                                     \
  RSV(0x6A, RESERVED_OPC)                                                     \
  RSV(0x6B, RESERVED_OPC)                                                     \
  RSV(0x6C, RESERVED_OPC)                                                     \
  RSV(0x6D, RESERVED_OPC)                                                     \
  RSV(0x6E, RESERVED_OPC)                                                     \
  RSV(0x6F, RESERVED_OPC)                                                     \
                                                                              \
  /* TODO(benvanik): remove ones we don't need/can emulate */                 \
  OPC(0x70, kAddI, "add_i", FLAG(kDefault), "sso", FF)                        \
  OPC(0x71, kAddF, "add_f", FLAG(kDefault), "sso", FF)                        \
  OPC(0x72, kSubI, "sub_i", FLAG(kDefault), "sso", FF)                        \
  OPC(0x73, kSubF, "sub_f", FLAG(kDefault), "sso", FF)                        \
  OPC(0x74, kAbsI, "abs_i", FLAG(kDefault), "so", FF)                         \
  OPC(0x75, kAbsF, "abs_f", FLAG(kDefault), "so", FF)                         \
  OPC(0x76, kMulI, "mul_i", FLAG(kDefault), "sso", FF)                        \
  OPC(0x77, kMulF, "mul_f", FLAG(kDefault), "sso", FF)                        \
  OPC(0x78, kDivIS, "div_i_s", FLAG(kDefault), "sso", FF)                     \
  OPC(0x79, kDivIU, "div_i_u", FLAG(kDefault), "sso", FF)                     \
  OPC(0x7A, kDivF, "div_f", FLAG(kDefault), "sso", FF)                        \
  OPC(0x7B, kMulAddI, "madd_i", FLAG(kDefault), "ssso", FF)                   \
  OPC(0x7C, kMulAddF, "madd_f", FLAG(kDefault), "ssso", FF)                   \
  OPC(0x7D, kCosF, "cos_f", FLAG(kDefault), "so", FF)                         \
  OPC(0x7E, kSinF, "sin_f", FLAG(kDefault), "so", FF)                         \
  OPC(0x7F, kTanhF, "tanh_f", FLAG(kDefault), "so", FF)                       \
  OPC(0x80, kAtan2F, "atan2_f", FLAG(kDefault), "sso", FF)                    \
  OPC(0x81, kExpF, "exp_f", FLAG(kDefault), "so", FF)                         \
  OPC(0x82, kLogF, "log_f", FLAG(kDefault), "so", FF)                         \
  OPC(0x83, kRsqrtF, "rsqrt_f", FLAG(kDefault), "so", FF)                     \
                                                                              \
  RSV(0x84, RESERVED_OPC)                                                     \
  RSV(0x85, RESERVED_OPC)                                                     \
  RSV(0x86, RESERVED_OPC)                                                     \
  RSV(0x87, RESERVED_OPC)                                                     \
  RSV(0x88, RESERVED_OPC)                                                     \
  RSV(0x89, RESERVED_OPC)                                                     \
  RSV(0x8A, RESERVED_OPC)                                                     \
  RSV(0x8B, RESERVED_OPC)                                                     \
  RSV(0x8C, RESERVED_OPC)                                                     \
  RSV(0x8D, RESERVED_OPC)                                                     \
  RSV(0x8E, RESERVED_OPC)                                                     \
  RSV(0x8F, RESERVED_OPC)                                                     \
                                                                              \
  OPC(0x90, kMinIS, "min_i_s", FLAG(kDefault), "sso", FF)                     \
  OPC(0x91, kMinIU, "min_i_u", FLAG(kDefault), "sso", FF)                     \
  OPC(0x92, kMinF, "min_f", FLAG(kDefault), "sso", FF)                        \
  OPC(0x93, kMaxIS, "max_i_s", FLAG(kDefault), "sso", FF)                     \
  OPC(0x94, kMaxIU, "max_i_u", FLAG(kDefault), "sso", FF)                     \
  OPC(0x95, kMaxF, "max_f", FLAG(kDefault), "sso", FF)                        \
  OPC(0x96, kClampIS, "clamp_i_s", FLAG(kDefault), "ssso", FF)                \
  OPC(0x97, kClampIU, "clamp_i_u", FLAG(kDefault), "ssso", FF)                \
  OPC(0x98, kClampF, "clamp_f", FLAG(kDefault), "ssso", FF)                   \
  OPC(0x99, kFloorF, "floor_f", FLAG(kDefault), "so", FF)                     \
  OPC(0x9A, kCeilF, "ceil_f", FLAG(kDefault), "so", FF)                       \
                                                                              \
  OPC(0x9B, kConvertSS, "convert_s_s", FLAG(kDefault), "tsto", FF)            \
  OPC(0x9C, kConvertUU, "convert_u_u", FLAG(kDefault), "tsto", FF)            \
  OPC(0x9D, kConvertSU, "convert_s_u", FLAG(kDefault), "tsto", FF)            \
  OPC(0x9E, kConvertUS, "convert_u_s", FLAG(kDefault), "tsto", FF)            \
                                                                              \
  RSV(0x9F, RESERVED_OPC)                                                     \
                                                                              \
  /* TODO(benvanik): reduction/sum/etc */                                     \
  /* TODO(benvanik): sort */                                                  \
                                                                              \
  OPC(0xA0, kMatMulI, "matmul_i", FLAG(kDefault), "sssso", FF)                \
  OPC(0xA1, kMatMulF, "matmul_f", FLAG(kDefault), "sso", FF)                  \
  /* TODO(benvanik): convolution */                                           \
                                                                              \
  OPC(0xA2, kReduceSumI, "reduce_sum_i", FLAG(kDefault), "ssio", FF)          \
  OPC(0xA3, kReduceSumF, "reduce_sum_f", FLAG(kDefault), "ssio", FF)          \
  OPC(0xA4, kReduceMinI, "reduce_min_i", FLAG(kDefault), "ssio", FF)          \
  OPC(0xA5, kReduceMinF, "reduce_min_f", FLAG(kDefault), "ssio", FF)          \
  OPC(0xA6, kReduceMaxI, "reduce_max_i", FLAG(kDefault), "ssio", FF)          \
  OPC(0xA7, kReduceMaxF, "reduce_max_f", FLAG(kDefault), "ssio", FF)          \
  RSV(0xA8, RESERVED_OPC)                                                     \
  RSV(0xA9, RESERVED_OPC)                                                     \
  RSV(0xAA, RESERVED_OPC)                                                     \
  RSV(0xAB, RESERVED_OPC)                                                     \
  RSV(0xAC, RESERVED_OPC)                                                     \
  RSV(0xAD, RESERVED_OPC)                                                     \
  RSV(0xAE, RESERVED_OPC)                                                     \
  RSV(0xAF, RESERVED_OPC)                                                     \
  RSV(0xB0, RESERVED_OPC)                                                     \
  RSV(0xB1, RESERVED_OPC)                                                     \
  RSV(0xB2, RESERVED_OPC)                                                     \
  RSV(0xB3, RESERVED_OPC)                                                     \
  RSV(0xB4, RESERVED_OPC)                                                     \
  RSV(0xB5, RESERVED_OPC)                                                     \
  RSV(0xB6, RESERVED_OPC)                                                     \
  RSV(0xB7, RESERVED_OPC)                                                     \
  RSV(0xB8, RESERVED_OPC)                                                     \
  RSV(0xB9, RESERVED_OPC)                                                     \
  RSV(0xBA, RESERVED_OPC)                                                     \
  RSV(0xBB, RESERVED_OPC)                                                     \
  RSV(0xBC, RESERVED_OPC)                                                     \
  RSV(0xBD, RESERVED_OPC)                                                     \
  RSV(0xBE, RESERVED_OPC)                                                     \
  RSV(0xBF, RESERVED_OPC)                                                     \
  RSV(0xC0, RESERVED_OPC)                                                     \
  RSV(0xC1, RESERVED_OPC)                                                     \
  RSV(0xC2, RESERVED_OPC)                                                     \
  RSV(0xC3, RESERVED_OPC)                                                     \
  RSV(0xC4, RESERVED_OPC)                                                     \
  RSV(0xC5, RESERVED_OPC)                                                     \
  RSV(0xC6, RESERVED_OPC)                                                     \
  RSV(0xC7, RESERVED_OPC)                                                     \
  RSV(0xC8, RESERVED_OPC)                                                     \
  RSV(0xC9, RESERVED_OPC)                                                     \
  RSV(0xCA, RESERVED_OPC)                                                     \
  RSV(0xCB, RESERVED_OPC)                                                     \
  RSV(0xCC, RESERVED_OPC)                                                     \
  RSV(0xCD, RESERVED_OPC)                                                     \
  RSV(0xCE, RESERVED_OPC)                                                     \
  RSV(0xCF, RESERVED_OPC)                                                     \
  RSV(0xD0, RESERVED_OPC)                                                     \
  RSV(0xD1, RESERVED_OPC)                                                     \
  RSV(0xD2, RESERVED_OPC)                                                     \
  RSV(0xD3, RESERVED_OPC)                                                     \
  RSV(0xD4, RESERVED_OPC)                                                     \
  RSV(0xD5, RESERVED_OPC)                                                     \
  RSV(0xD6, RESERVED_OPC)                                                     \
  RSV(0xD7, RESERVED_OPC)                                                     \
  RSV(0xD8, RESERVED_OPC)                                                     \
  RSV(0xD9, RESERVED_OPC)                                                     \
  RSV(0xDA, RESERVED_OPC)                                                     \
  RSV(0xDB, RESERVED_OPC)                                                     \
  RSV(0xDC, RESERVED_OPC)                                                     \
  RSV(0xDD, RESERVED_OPC)                                                     \
  RSV(0xDE, RESERVED_OPC)                                                     \
  RSV(0xDF, RESERVED_OPC)                                                     \
  RSV(0xE0, RESERVED_OPC)                                                     \
  RSV(0xE1, RESERVED_OPC)                                                     \
  RSV(0xE2, RESERVED_OPC)                                                     \
  RSV(0xE3, RESERVED_OPC)                                                     \
  RSV(0xE4, RESERVED_OPC)                                                     \
  RSV(0xE5, RESERVED_OPC)                                                     \
  RSV(0xE6, RESERVED_OPC)                                                     \
  RSV(0xE7, RESERVED_OPC)                                                     \
  RSV(0xE8, RESERVED_OPC)                                                     \
  RSV(0xE9, RESERVED_OPC)                                                     \
  RSV(0xEA, RESERVED_OPC)                                                     \
  RSV(0xEB, RESERVED_OPC)                                                     \
  RSV(0xEC, RESERVED_OPC)                                                     \
  RSV(0xED, RESERVED_OPC)                                                     \
  RSV(0xEE, RESERVED_OPC)                                                     \
  RSV(0xEF, RESERVED_OPC)                                                     \
  RSV(0xF0, RESERVED_OPC)                                                     \
  RSV(0xF1, RESERVED_OPC)                                                     \
  RSV(0xF2, RESERVED_OPC)                                                     \
  RSV(0xF3, RESERVED_OPC)                                                     \
  RSV(0xF4, RESERVED_OPC)                                                     \
  RSV(0xF5, RESERVED_OPC)                                                     \
  RSV(0xF6, RESERVED_OPC)                                                     \
  RSV(0xF7, RESERVED_OPC)                                                     \
  RSV(0xF8, RESERVED_OPC)                                                     \
  RSV(0xF9, RESERVED_OPC)                                                     \
  RSV(0xFA, RESERVED_OPC)                                                     \
  RSV(0xFB, RESERVED_OPC)                                                     \
  RSV(0xFC, RESERVED_OPC)                                                     \
                                                                              \
  OPC(0xFD, kTrace, "trace", FLAG(kDefault), "s", FF)                         \
  OPC(0xFE, kCondBreak, "cond_break", FLAG(kDefault), "s", FF)                \
  OPC(0xFF, kBreak, "break", FLAG(kDefault), "", FF)

#define DECLARE_ENUM(ordinal, enum_name, ...) enum_name = ordinal,
enum class InterpreterOpcode : uint8_t {
  IREE_INTERPRETER_OPCODE_LIST(DECLARE_ENUM, DECLARE_ENUM)
};
#undef DECLARE_ENUM

}  // namespace iree

#endif  // IREE_SCHEMAS_BYTECODE_INTERPRETER_BYTECODE_V0_H_
