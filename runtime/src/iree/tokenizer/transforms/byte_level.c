// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/transforms/byte_level.h"

#include "iree/base/internal/unicode.h"
#include "iree/tokenizer/normalizer.h"

//===----------------------------------------------------------------------===//
// ByteLevel Lookup Tables
//===----------------------------------------------------------------------===//

// GPT-2 byte-to-Unicode mapping.
//
// The mapping ensures every byte (0-255) has a unique Unicode character:
// - Printable ASCII (0x21-0x7E): map directly (! through ~)
// - Extended printable (0xA1-0xAC, 0xAE-0xFF): map directly
// - Control/special bytes: map to codepoints 256+ (offset to avoid conflicts)
//
// This allows any byte sequence to be represented as valid Unicode text.

// Compile-time lookup table: byte -> Unicode codepoint.
static const uint16_t IREE_TOKENIZER_BYTE_TO_CHAR[256] = {
    // clang-format off
    // 0x00-0x1F: Control characters -> codepoints 256-287 (offset mapping)
    256,  257,  258,  259,  260,  261,  262,  263,   // 0x00-0x07
    264,  265,  266,  267,  268,  269,  270,  271,   // 0x08-0x0F
    272,  273,  274,  275,  276,  277,  278,  279,   // 0x10-0x17
    280,  281,  282,  283,  284,  285,  286,  287,   // 0x18-0x1F
    // 0x20: Space -> codepoint 288 (offset mapping)
    288,
    // 0x21-0x7E: Printable ASCII -> direct mapping (codepoint == byte)
    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,        // 0x21-0x27: !"#$%&'
    0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E, 0x2F,  // 0x28-0x2F: ()*+,-./
    0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,  // 0x30-0x37: 01234567
    0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F,  // 0x38-0x3F: 89:;<=>?
    0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47,  // 0x40-0x47: @ABCDEFG
    0x48, 0x49, 0x4A, 0x4B, 0x4C, 0x4D, 0x4E, 0x4F,  // 0x48-0x4F: HIJKLMNO
    0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57,  // 0x50-0x57: PQRSTUVW
    0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D, 0x5E, 0x5F,  // 0x58-0x5F: XYZ[\]^_
    0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,  // 0x60-0x67: `abcdefg
    0x68, 0x69, 0x6A, 0x6B, 0x6C, 0x6D, 0x6E, 0x6F,  // 0x68-0x6F: hijklmno
    0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,  // 0x70-0x77: pqrstuvw
    0x78, 0x79, 0x7A, 0x7B, 0x7C, 0x7D, 0x7E,        // 0x78-0x7E: xyz{|}~
    // 0x7F: DEL -> codepoint 289 (offset mapping)
    289,
    // 0x80-0x9F: C1 control characters -> codepoints 290-321 (offset mapping)
    290,  291,  292,  293,  294,  295,  296,  297,   // 0x80-0x87
    298,  299,  300,  301,  302,  303,  304,  305,   // 0x88-0x8F
    306,  307,  308,  309,  310,  311,  312,  313,   // 0x90-0x97
    314,  315,  316,  317,  318,  319,  320,  321,   // 0x98-0x9F
    // 0xA0: Non-breaking space -> codepoint 322 (offset mapping)
    322,
    // 0xA1-0xAC: Latin-1 supplement -> direct mapping
    0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8,  // 0xA1-0xA8
    0xA9, 0xAA, 0xAB, 0xAC,                          // 0xA9-0xAC
    // 0xAD: Soft hyphen -> codepoint 323 (offset mapping)
    323,
    // 0xAE-0xFF: Latin-1 supplement -> direct mapping
    0xAE, 0xAF,                                      // 0xAE-0xAF
    0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,  // 0xB0-0xB7
    0xB8, 0xB9, 0xBA, 0xBB, 0xBC, 0xBD, 0xBE, 0xBF,  // 0xB8-0xBF
    0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,  // 0xC0-0xC7
    0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCD, 0xCE, 0xCF,  // 0xC8-0xCF
    0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7,  // 0xD0-0xD7
    0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE, 0xDF,  // 0xD8-0xDF
    0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,  // 0xE0-0xE7
    0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED, 0xEE, 0xEF,  // 0xE8-0xEF
    0xF0, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7,  // 0xF0-0xF7
    0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF,  // 0xF8-0xFF
    // clang-format on
};

// Reverse lookup: Unicode codepoint -> byte value.
// Returns 0xFF and sets valid=false for invalid codepoints.
static uint8_t iree_tokenizer_char_to_byte(uint32_t codepoint, bool* valid) {
  *valid = true;

  // Direct mappings: codepoint == byte value.
  if ((codepoint >= 0x21 && codepoint <= 0x7E) ||
      (codepoint >= 0xA1 && codepoint <= 0xAC) ||
      (codepoint >= 0xAE && codepoint <= 0xFF)) {
    return (uint8_t)codepoint;
  }

  // Offset mappings: codepoint = 256 + offset.
  if (codepoint >= 256 && codepoint <= 323) {
    static const uint8_t offset_to_byte[] = {
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,  // 256-263 -> 0x00-0x07
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,  // 264-271 -> 0x08-0x0F
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,  // 272-279 -> 0x10-0x17
        0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,  // 280-287 -> 0x18-0x1F
        0x20,                                            // 288 -> 0x20 (space)
        0x7F,                                            // 289 -> 0x7F (DEL)
        0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,  // 290-297 -> 0x80-0x87
        0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0x8E, 0x8F,  // 298-305 -> 0x88-0x8F
        0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,  // 306-313 -> 0x90-0x97
        0x98, 0x99, 0x9A, 0x9B, 0x9C, 0x9D, 0x9E, 0x9F,  // 314-321 -> 0x98-0x9F
        0xA0,                                            // 322 -> 0xA0 (NBSP)
        0xAD,  // 323 -> 0xAD (soft hyphen)
    };
    return offset_to_byte[codepoint - 256];
  }

  // Invalid codepoint for ByteLevel encoding.
  *valid = false;
  return 0xFF;
}

//===----------------------------------------------------------------------===//
// ByteLevel Encode
//===----------------------------------------------------------------------===//

// Encode context for streaming ByteLevel output.
typedef struct {
  char data[IREE_TOKENIZER_DATA_BATCH_CAPACITY];
  iree_host_size_t data_used;
  iree_tokenizer_string_callback_fn_t callback;
  void* user_data;
} iree_tokenizer_byte_level_encode_context_t;

// Append a ByteLevel-mapped codepoint to the buffer, flushing if needed.
static iree_status_t iree_tokenizer_byte_level_emit_codepoint(
    iree_tokenizer_byte_level_encode_context_t* ctx, uint16_t cp) {
  iree_host_size_t char_length = iree_unicode_utf8_encoded_length(cp);
  if (ctx->data_used + char_length > IREE_TOKENIZER_DATA_BATCH_CAPACITY) {
    if (ctx->data_used == 0) return iree_ok_status();
    iree_string_view_t segment =
        iree_make_string_view(ctx->data, ctx->data_used);
    iree_string_view_list_t list = {1, &segment};
    IREE_RETURN_IF_ERROR(ctx->callback(ctx->user_data, list));
    ctx->data_used = 0;
  }
  ctx->data_used += iree_unicode_utf8_encode(cp, ctx->data + ctx->data_used);
  return iree_ok_status();
}

// Flush any remaining buffered data.
static iree_status_t iree_tokenizer_byte_level_flush(
    iree_tokenizer_byte_level_encode_context_t* ctx) {
  if (ctx->data_used == 0) return iree_ok_status();
  iree_string_view_t segment = iree_make_string_view(ctx->data, ctx->data_used);
  iree_string_view_list_t list = {1, &segment};
  return ctx->callback(ctx->user_data, list);
}

// Fast path: no normalization, iterate raw bytes directly.
static iree_status_t iree_tokenizer_byte_level_encode_raw(
    iree_tokenizer_byte_level_encode_context_t* ctx, iree_string_view_t text) {
  for (iree_host_size_t i = 0; i < text.size; ++i) {
    uint16_t cp = IREE_TOKENIZER_BYTE_TO_CHAR[(uint8_t)text.data[i]];
    IREE_RETURN_IF_ERROR(iree_tokenizer_byte_level_emit_codepoint(ctx, cp));
  }
  return iree_ok_status();
}

// Slow path: decode UTF-8, normalize, then map resulting bytes.
static iree_status_t iree_tokenizer_byte_level_encode_normalized(
    iree_tokenizer_byte_level_encode_context_t* ctx,
    const iree_tokenizer_normalizer_t* normalizer, iree_string_view_t text) {
  iree_host_size_t position = 0;
  while (position < text.size) {
    uint32_t cp = iree_unicode_utf8_decode(text, &position);

    // Normalize this codepoint (may produce 0-4 codepoints).
    uint32_t norm_cps[4];
    iree_host_size_t norm_count;
    iree_tokenizer_normalizer_normalize_codepoint(normalizer, cp, norm_cps,
                                                  &norm_count);

    // Process each normalized codepoint.
    for (iree_host_size_t n = 0; n < norm_count; ++n) {
      // Encode normalized codepoint to UTF-8 bytes.
      char utf8_buffer[4];
      iree_host_size_t utf8_length =
          iree_unicode_utf8_encode(norm_cps[n], utf8_buffer);

      // Map each byte to ByteLevel Unicode character.
      for (iree_host_size_t b = 0; b < utf8_length; ++b) {
        uint16_t byte_cp = IREE_TOKENIZER_BYTE_TO_CHAR[(uint8_t)utf8_buffer[b]];
        IREE_RETURN_IF_ERROR(
            iree_tokenizer_byte_level_emit_codepoint(ctx, byte_cp));
      }
    }
  }
  return iree_ok_status();
}

// ByteLevel encoding with optional inline normalization.
// Maps each input byte to a Unicode character.
iree_status_t iree_tokenizer_byte_level_encode(
    const iree_tokenizer_normalizer_t* normalizer,
    const iree_tokenizer_byte_level_config_t* config, iree_string_view_t text,
    iree_tokenizer_string_callback_fn_t callback, void* user_data) {
  IREE_ASSERT_ARGUMENT(config);
  if (text.size == 0) return iree_ok_status();

  iree_tokenizer_byte_level_encode_context_t ctx = {
      .data_used = 0,
      .callback = callback,
      .user_data = user_data,
  };

  // Write prefix space if needed.
  bool add_space =
      iree_any_bit_set(config->flags,
                       IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE) &&
      text.data[0] != ' ';
  if (add_space) {
    uint16_t space_cp = IREE_TOKENIZER_BYTE_TO_CHAR[0x20];
    ctx.data_used += iree_unicode_utf8_encode(space_cp, ctx.data);
  }

  // Dispatch to appropriate encoding path.
  if (!normalizer || normalizer->type == IREE_TOKENIZER_NORMALIZER_NONE) {
    IREE_RETURN_IF_ERROR(iree_tokenizer_byte_level_encode_raw(&ctx, text));
  } else if (normalizer->type == IREE_TOKENIZER_NORMALIZER_NFC) {
    // NFC requires string-level processing (needs to see codepoint sequences).
    // Pre-normalize the entire text, then use the raw path.
    char normalized_buffer[8192];
    iree_host_size_t normalized_length = 0;
    IREE_RETURN_IF_ERROR(iree_tokenizer_normalizer_apply(
        normalizer, text, normalized_buffer, sizeof(normalized_buffer),
        &normalized_length));
    IREE_RETURN_IF_ERROR(iree_tokenizer_byte_level_encode_raw(
        &ctx, iree_make_string_view(normalized_buffer, normalized_length)));
  } else {
    IREE_RETURN_IF_ERROR(
        iree_tokenizer_byte_level_encode_normalized(&ctx, normalizer, text));
  }

  return iree_tokenizer_byte_level_flush(&ctx);
}

//===----------------------------------------------------------------------===//
// ByteLevel Decode
//===----------------------------------------------------------------------===//

// ByteLevel decoding: maps Unicode characters back to bytes.
// Since each encoded codepoint becomes exactly 1 byte, output is always <=
// input size. This allows safe in-place decoding.
iree_status_t iree_tokenizer_byte_level_decode(
    const iree_tokenizer_byte_level_config_t* config, iree_string_view_t text,
    char* out_buffer, iree_host_size_t max_size, iree_host_size_t* out_size) {
  IREE_ASSERT_ARGUMENT(config);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_ASSERT_ARGUMENT(out_size);
  *out_size = 0;

  if (text.size == 0) {
    return iree_ok_status();
  }

  // Decode directly into the output buffer.
  char* write_ptr = out_buffer;
  iree_host_size_t position = 0;
  bool is_first = true;

  while (position < text.size) {
    uint32_t cp = iree_unicode_utf8_decode(text, &position);
    bool valid;
    uint8_t byte = iree_tokenizer_char_to_byte(cp, &valid);
    if (!valid) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "invalid codepoint %" PRIu32 " in ByteLevel encoded text", cp);
    }

    // Skip prefix space if configured and this is the first character.
    if (is_first &&
        iree_any_bit_set(config->flags,
                         IREE_TOKENIZER_BYTE_LEVEL_FLAG_ADD_PREFIX_SPACE) &&
        byte == 0x20) {
      is_first = false;
      continue;
    }
    is_first = false;

    // Check buffer capacity.
    iree_host_size_t written = (iree_host_size_t)(write_ptr - out_buffer);
    if (written >= max_size) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "output buffer too small");
    }
    *write_ptr++ = (char)byte;
  }

  *out_size = (iree_host_size_t)(write_ptr - out_buffer);
  return iree_ok_status();
}
