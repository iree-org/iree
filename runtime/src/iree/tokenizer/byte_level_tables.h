// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Precomputed lookup tables for GPT-2 ByteLevel encoding/decoding.
//
// This file is included by both the BPE encoder (model/bpe.c) and the
// ByteLevel decoder (decoder/byte_level.c). All tables are designed for
// O(1) lookup during encoding/decoding operations.
//
// ============================================================================
// GPT-2 ByteLevel Encoding Background
// ============================================================================
//
// GPT-2 introduced "ByteLevel" BPE where each input byte (0-255) is mapped to
// a Unicode codepoint before tokenization. This has two benefits:
//
//   1. All 256 byte values become visible printable characters, making the
//      vocabulary human-readable (no invisible control characters).
//
//   2. The tokenizer operates on Unicode text, enabling standard BPE merge
//      operations on UTF-8 encoded strings.
//
// The mapping partitions bytes into "nice" and "non-nice" categories:
//
//   Nice bytes (188 total): Printable ASCII (0x21-0x7E) and most Latin-1
//   supplement (0xA1-0xAC, 0xAE-0xFF). These map to themselves (identity).
//
//   Non-nice bytes (68 total): Control characters (0x00-0x20), DEL (0x7F),
//   C1 controls (0x80-0x9F), NBSP (0xA0), and soft hyphen (0xAD). These map
//   to U+0100 through U+0143 to avoid invisible/problematic characters.
//
// The most commonly seen mapped character is U+0120 (Ġ), which represents
// a space (0x20) and appears as a word-boundary marker in GPT-2 vocabularies.
// Similarly, U+010A (Ċ) represents newline (0x0A).
//
// References:
//   - GPT-2 paper:
//     https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
//   - HuggingFace tokenizers ByteLevel implementation

#ifndef IREE_TOKENIZER_UTIL_BYTE_LEVEL_TABLES_H_
#define IREE_TOKENIZER_UTIL_BYTE_LEVEL_TABLES_H_

#include <stdbool.h>
#include <stdint.h>

//===----------------------------------------------------------------------===//
// Forward Mapping: Byte -> Codepoint (for encoding)
//===----------------------------------------------------------------------===//

// Maps each input byte (0-255) to its ByteLevel Unicode codepoint.
//
// Usage: uint16_t codepoint = iree_tokenizer_byte_level_mapping[byte];
//
// Result range: [0x21, 0x7E] u [0xA1, 0xFF] u [0x100, 0x143]
// All codepoints require at most 2 bytes when UTF-8 encoded.
//
// Table layout (16 values per row):
//   Row 0-1:   0x00-0x1F -> 0x100-0x11F (control chars)
//   Row 2:     0x20 -> 0x120 (space), 0x21-0x2F -> identity
//   Row 3-7:   0x30-0x7E -> identity (printable ASCII)
//   Row 7:     0x7F -> 0x121 (DEL)
//   Row 8-9:   0x80-0x9F -> 0x122-0x141 (C1 controls)
//   Row 10:    0xA0 -> 0x142 (NBSP), 0xA1-0xAC -> identity, 0xAD -> 0x143
//   Row 11-15: 0xAE-0xFF -> identity (Latin-1 supplement)

// clang-format off
static const uint16_t iree_tokenizer_byte_level_mapping[256] = {
    0x100, 0x101, 0x102, 0x103, 0x104, 0x105, 0x106, 0x107, 0x108, 0x109, 0x10A, 0x10B, 0x10C, 0x10D, 0x10E, 0x10F,
    0x110, 0x111, 0x112, 0x113, 0x114, 0x115, 0x116, 0x117, 0x118, 0x119, 0x11A, 0x11B, 0x11C, 0x11D, 0x11E, 0x11F,
    0x120, 0x021, 0x022, 0x023, 0x024, 0x025, 0x026, 0x027, 0x028, 0x029, 0x02A, 0x02B, 0x02C, 0x02D, 0x02E, 0x02F,
    0x030, 0x031, 0x032, 0x033, 0x034, 0x035, 0x036, 0x037, 0x038, 0x039, 0x03A, 0x03B, 0x03C, 0x03D, 0x03E, 0x03F,
    0x040, 0x041, 0x042, 0x043, 0x044, 0x045, 0x046, 0x047, 0x048, 0x049, 0x04A, 0x04B, 0x04C, 0x04D, 0x04E, 0x04F,
    0x050, 0x051, 0x052, 0x053, 0x054, 0x055, 0x056, 0x057, 0x058, 0x059, 0x05A, 0x05B, 0x05C, 0x05D, 0x05E, 0x05F,
    0x060, 0x061, 0x062, 0x063, 0x064, 0x065, 0x066, 0x067, 0x068, 0x069, 0x06A, 0x06B, 0x06C, 0x06D, 0x06E, 0x06F,
    0x070, 0x071, 0x072, 0x073, 0x074, 0x075, 0x076, 0x077, 0x078, 0x079, 0x07A, 0x07B, 0x07C, 0x07D, 0x07E, 0x121,
    0x122, 0x123, 0x124, 0x125, 0x126, 0x127, 0x128, 0x129, 0x12A, 0x12B, 0x12C, 0x12D, 0x12E, 0x12F, 0x130, 0x131,
    0x132, 0x133, 0x134, 0x135, 0x136, 0x137, 0x138, 0x139, 0x13A, 0x13B, 0x13C, 0x13D, 0x13E, 0x13F, 0x140, 0x141,
    0x142, 0x0A1, 0x0A2, 0x0A3, 0x0A4, 0x0A5, 0x0A6, 0x0A7, 0x0A8, 0x0A9, 0x0AA, 0x0AB, 0x0AC, 0x143, 0x0AE, 0x0AF,
    0x0B0, 0x0B1, 0x0B2, 0x0B3, 0x0B4, 0x0B5, 0x0B6, 0x0B7, 0x0B8, 0x0B9, 0x0BA, 0x0BB, 0x0BC, 0x0BD, 0x0BE, 0x0BF,
    0x0C0, 0x0C1, 0x0C2, 0x0C3, 0x0C4, 0x0C5, 0x0C6, 0x0C7, 0x0C8, 0x0C9, 0x0CA, 0x0CB, 0x0CC, 0x0CD, 0x0CE, 0x0CF,
    0x0D0, 0x0D1, 0x0D2, 0x0D3, 0x0D4, 0x0D5, 0x0D6, 0x0D7, 0x0D8, 0x0D9, 0x0DA, 0x0DB, 0x0DC, 0x0DD, 0x0DE, 0x0DF,
    0x0E0, 0x0E1, 0x0E2, 0x0E3, 0x0E4, 0x0E5, 0x0E6, 0x0E7, 0x0E8, 0x0E9, 0x0EA, 0x0EB, 0x0EC, 0x0ED, 0x0EE, 0x0EF,
    0x0F0, 0x0F1, 0x0F2, 0x0F3, 0x0F4, 0x0F5, 0x0F6, 0x0F7, 0x0F8, 0x0F9, 0x0FA, 0x0FB, 0x0FC, 0x0FD, 0x0FE, 0x0FF,
};
// clang-format on

//===----------------------------------------------------------------------===//
// Reverse Mapping: Codepoint -> Byte (for decoding)
//===----------------------------------------------------------------------===//

// Maps the shifted codepoint range (0x100-0x143) back to original bytes.
//
// Usage: For codepoint in [0x100, 0x143]:
//        uint8_t byte = iree_tokenizer_byte_level_reverse_mapping[codepoint -
//        0x100];
//
// Only 68 entries needed (for the non-identity mapped bytes).
// Identity-mapped codepoints (0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF) decode to
// themselves.

// clang-format off
static const uint8_t iree_tokenizer_byte_level_reverse_mapping[68] = {
    // 0x100-0x11F -> 0x00-0x1F (control chars)
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
    0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
    // 0x120 -> 0x20 (space), 0x121 -> 0x7F (DEL)
    0x20, 0x7F,
    // 0x122-0x141 -> 0x80-0x9F (C1 controls)
    0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,
    0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,
    // 0x142 -> 0xA0 (NBSP), 0x143 -> 0xAD (soft hyphen)
    0xA0, 0xAD,
};
// clang-format on

//===----------------------------------------------------------------------===//
// Precomputed UTF-8 Encoding (for fast trie traversal)
//===----------------------------------------------------------------------===//

// Pre-computed UTF-8 bytes for each ByteLevel-mapped codepoint.
//
// This table eliminates runtime UTF-8 encoding during trie traversal. Given an
// input byte, we can directly read the UTF-8 bytes needed to look up the
// corresponding vocabulary token without any computation.
//
// Each entry contains:
//   bytes[0..1]: The UTF-8 encoded bytes (1 or 2 bytes used)
//   length:      Number of valid bytes (1 or 2)
//
// UTF-8 encoding for codepoints in the ByteLevel range:
//   0x00-0x7F:   1 byte  (0xxxxxxx)
//   0x80-0x7FF:  2 bytes (110xxxxx 10xxxxxx)

typedef struct iree_tokenizer_byte_level_utf8_t {
  // UTF-8 encoded bytes (index 1 unused if length==1).
  uint8_t bytes[2];
  // Number of valid UTF-8 bytes (1 or 2).
  uint8_t length;
  uint8_t padding;  // Alignment.
} iree_tokenizer_byte_level_utf8_t;

// Helper macros for table initialization.
#define IREE_BL_U1_(cp) {{(uint8_t)(cp), 0}, 1, 0}
#define IREE_BL_U2_(cp) \
  {{(uint8_t)(0xC0 | ((cp) >> 6)), (uint8_t)(0x80 | ((cp) & 0x3F))}, 2, 0}

// clang-format off
static const iree_tokenizer_byte_level_utf8_t iree_tokenizer_byte_level_utf8[256] = {
    // 0x00-0x1F: Control chars -> U+0100-U+011F (2-byte UTF-8)
    IREE_BL_U2_(0x100), IREE_BL_U2_(0x101), IREE_BL_U2_(0x102), IREE_BL_U2_(0x103), IREE_BL_U2_(0x104), IREE_BL_U2_(0x105), IREE_BL_U2_(0x106), IREE_BL_U2_(0x107),
    IREE_BL_U2_(0x108), IREE_BL_U2_(0x109), IREE_BL_U2_(0x10A), IREE_BL_U2_(0x10B), IREE_BL_U2_(0x10C), IREE_BL_U2_(0x10D), IREE_BL_U2_(0x10E), IREE_BL_U2_(0x10F),
    IREE_BL_U2_(0x110), IREE_BL_U2_(0x111), IREE_BL_U2_(0x112), IREE_BL_U2_(0x113), IREE_BL_U2_(0x114), IREE_BL_U2_(0x115), IREE_BL_U2_(0x116), IREE_BL_U2_(0x117),
    IREE_BL_U2_(0x118), IREE_BL_U2_(0x119), IREE_BL_U2_(0x11A), IREE_BL_U2_(0x11B), IREE_BL_U2_(0x11C), IREE_BL_U2_(0x11D), IREE_BL_U2_(0x11E), IREE_BL_U2_(0x11F),
    // 0x20: Space -> U+0120 (2-byte UTF-8)
    IREE_BL_U2_(0x120),
    // 0x21-0x7E: Printable ASCII (1-byte UTF-8, identity)
    IREE_BL_U1_(0x21), IREE_BL_U1_(0x22), IREE_BL_U1_(0x23), IREE_BL_U1_(0x24), IREE_BL_U1_(0x25), IREE_BL_U1_(0x26), IREE_BL_U1_(0x27),
    IREE_BL_U1_(0x28), IREE_BL_U1_(0x29), IREE_BL_U1_(0x2A), IREE_BL_U1_(0x2B), IREE_BL_U1_(0x2C), IREE_BL_U1_(0x2D), IREE_BL_U1_(0x2E), IREE_BL_U1_(0x2F),
    IREE_BL_U1_(0x30), IREE_BL_U1_(0x31), IREE_BL_U1_(0x32), IREE_BL_U1_(0x33), IREE_BL_U1_(0x34), IREE_BL_U1_(0x35), IREE_BL_U1_(0x36), IREE_BL_U1_(0x37),
    IREE_BL_U1_(0x38), IREE_BL_U1_(0x39), IREE_BL_U1_(0x3A), IREE_BL_U1_(0x3B), IREE_BL_U1_(0x3C), IREE_BL_U1_(0x3D), IREE_BL_U1_(0x3E), IREE_BL_U1_(0x3F),
    IREE_BL_U1_(0x40), IREE_BL_U1_(0x41), IREE_BL_U1_(0x42), IREE_BL_U1_(0x43), IREE_BL_U1_(0x44), IREE_BL_U1_(0x45), IREE_BL_U1_(0x46), IREE_BL_U1_(0x47),
    IREE_BL_U1_(0x48), IREE_BL_U1_(0x49), IREE_BL_U1_(0x4A), IREE_BL_U1_(0x4B), IREE_BL_U1_(0x4C), IREE_BL_U1_(0x4D), IREE_BL_U1_(0x4E), IREE_BL_U1_(0x4F),
    IREE_BL_U1_(0x50), IREE_BL_U1_(0x51), IREE_BL_U1_(0x52), IREE_BL_U1_(0x53), IREE_BL_U1_(0x54), IREE_BL_U1_(0x55), IREE_BL_U1_(0x56), IREE_BL_U1_(0x57),
    IREE_BL_U1_(0x58), IREE_BL_U1_(0x59), IREE_BL_U1_(0x5A), IREE_BL_U1_(0x5B), IREE_BL_U1_(0x5C), IREE_BL_U1_(0x5D), IREE_BL_U1_(0x5E), IREE_BL_U1_(0x5F),
    IREE_BL_U1_(0x60), IREE_BL_U1_(0x61), IREE_BL_U1_(0x62), IREE_BL_U1_(0x63), IREE_BL_U1_(0x64), IREE_BL_U1_(0x65), IREE_BL_U1_(0x66), IREE_BL_U1_(0x67),
    IREE_BL_U1_(0x68), IREE_BL_U1_(0x69), IREE_BL_U1_(0x6A), IREE_BL_U1_(0x6B), IREE_BL_U1_(0x6C), IREE_BL_U1_(0x6D), IREE_BL_U1_(0x6E), IREE_BL_U1_(0x6F),
    IREE_BL_U1_(0x70), IREE_BL_U1_(0x71), IREE_BL_U1_(0x72), IREE_BL_U1_(0x73), IREE_BL_U1_(0x74), IREE_BL_U1_(0x75), IREE_BL_U1_(0x76), IREE_BL_U1_(0x77),
    IREE_BL_U1_(0x78), IREE_BL_U1_(0x79), IREE_BL_U1_(0x7A), IREE_BL_U1_(0x7B), IREE_BL_U1_(0x7C), IREE_BL_U1_(0x7D), IREE_BL_U1_(0x7E),
    // 0x7F: DEL -> U+0121 (2-byte UTF-8)
    IREE_BL_U2_(0x121),
    // 0x80-0x9F: C1 controls -> U+0122-U+0141 (2-byte UTF-8)
    IREE_BL_U2_(0x122), IREE_BL_U2_(0x123), IREE_BL_U2_(0x124), IREE_BL_U2_(0x125), IREE_BL_U2_(0x126), IREE_BL_U2_(0x127), IREE_BL_U2_(0x128), IREE_BL_U2_(0x129),
    IREE_BL_U2_(0x12A), IREE_BL_U2_(0x12B), IREE_BL_U2_(0x12C), IREE_BL_U2_(0x12D), IREE_BL_U2_(0x12E), IREE_BL_U2_(0x12F), IREE_BL_U2_(0x130), IREE_BL_U2_(0x131),
    IREE_BL_U2_(0x132), IREE_BL_U2_(0x133), IREE_BL_U2_(0x134), IREE_BL_U2_(0x135), IREE_BL_U2_(0x136), IREE_BL_U2_(0x137), IREE_BL_U2_(0x138), IREE_BL_U2_(0x139),
    IREE_BL_U2_(0x13A), IREE_BL_U2_(0x13B), IREE_BL_U2_(0x13C), IREE_BL_U2_(0x13D), IREE_BL_U2_(0x13E), IREE_BL_U2_(0x13F), IREE_BL_U2_(0x140), IREE_BL_U2_(0x141),
    // 0xA0: NBSP -> U+0142 (2-byte UTF-8)
    IREE_BL_U2_(0x142),
    // 0xA1-0xAC: Latin-1 identity (2-byte UTF-8 since >= 0x80)
    IREE_BL_U2_(0xA1), IREE_BL_U2_(0xA2), IREE_BL_U2_(0xA3), IREE_BL_U2_(0xA4), IREE_BL_U2_(0xA5), IREE_BL_U2_(0xA6), IREE_BL_U2_(0xA7),
    IREE_BL_U2_(0xA8), IREE_BL_U2_(0xA9), IREE_BL_U2_(0xAA), IREE_BL_U2_(0xAB), IREE_BL_U2_(0xAC),
    // 0xAD: Soft hyphen -> U+0143 (2-byte UTF-8)
    IREE_BL_U2_(0x143),
    // 0xAE-0xFF: Latin-1 identity (2-byte UTF-8)
    IREE_BL_U2_(0xAE), IREE_BL_U2_(0xAF),
    IREE_BL_U2_(0xB0), IREE_BL_U2_(0xB1), IREE_BL_U2_(0xB2), IREE_BL_U2_(0xB3), IREE_BL_U2_(0xB4), IREE_BL_U2_(0xB5), IREE_BL_U2_(0xB6), IREE_BL_U2_(0xB7),
    IREE_BL_U2_(0xB8), IREE_BL_U2_(0xB9), IREE_BL_U2_(0xBA), IREE_BL_U2_(0xBB), IREE_BL_U2_(0xBC), IREE_BL_U2_(0xBD), IREE_BL_U2_(0xBE), IREE_BL_U2_(0xBF),
    IREE_BL_U2_(0xC0), IREE_BL_U2_(0xC1), IREE_BL_U2_(0xC2), IREE_BL_U2_(0xC3), IREE_BL_U2_(0xC4), IREE_BL_U2_(0xC5), IREE_BL_U2_(0xC6), IREE_BL_U2_(0xC7),
    IREE_BL_U2_(0xC8), IREE_BL_U2_(0xC9), IREE_BL_U2_(0xCA), IREE_BL_U2_(0xCB), IREE_BL_U2_(0xCC), IREE_BL_U2_(0xCD), IREE_BL_U2_(0xCE), IREE_BL_U2_(0xCF),
    IREE_BL_U2_(0xD0), IREE_BL_U2_(0xD1), IREE_BL_U2_(0xD2), IREE_BL_U2_(0xD3), IREE_BL_U2_(0xD4), IREE_BL_U2_(0xD5), IREE_BL_U2_(0xD6), IREE_BL_U2_(0xD7),
    IREE_BL_U2_(0xD8), IREE_BL_U2_(0xD9), IREE_BL_U2_(0xDA), IREE_BL_U2_(0xDB), IREE_BL_U2_(0xDC), IREE_BL_U2_(0xDD), IREE_BL_U2_(0xDE), IREE_BL_U2_(0xDF),
    IREE_BL_U2_(0xE0), IREE_BL_U2_(0xE1), IREE_BL_U2_(0xE2), IREE_BL_U2_(0xE3), IREE_BL_U2_(0xE4), IREE_BL_U2_(0xE5), IREE_BL_U2_(0xE6), IREE_BL_U2_(0xE7),
    IREE_BL_U2_(0xE8), IREE_BL_U2_(0xE9), IREE_BL_U2_(0xEA), IREE_BL_U2_(0xEB), IREE_BL_U2_(0xEC), IREE_BL_U2_(0xED), IREE_BL_U2_(0xEE), IREE_BL_U2_(0xEF),
    IREE_BL_U2_(0xF0), IREE_BL_U2_(0xF1), IREE_BL_U2_(0xF2), IREE_BL_U2_(0xF3), IREE_BL_U2_(0xF4), IREE_BL_U2_(0xF5), IREE_BL_U2_(0xF6), IREE_BL_U2_(0xF7),
    IREE_BL_U2_(0xF8), IREE_BL_U2_(0xF9), IREE_BL_U2_(0xFA), IREE_BL_U2_(0xFB), IREE_BL_U2_(0xFC), IREE_BL_U2_(0xFD), IREE_BL_U2_(0xFE), IREE_BL_U2_(0xFF),
};
// clang-format on

#undef IREE_BL_U1_
#undef IREE_BL_U2_

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

// Returns true if the codepoint is in the identity-mapped range.
// These codepoints decode to themselves (no table lookup needed).
static inline bool iree_tokenizer_byte_level_is_identity(uint32_t codepoint) {
  return (codepoint >= 0x21 && codepoint <= 0x7E) ||  // Printable ASCII
         (codepoint >= 0xA1 && codepoint <= 0xAC) ||  // Latin-1 subset
         (codepoint >= 0xAE && codepoint <= 0xFF);    // Latin-1 subset
}

#endif  // IREE_TOKENIZER_UTIL_BYTE_LEVEL_TABLES_H_
