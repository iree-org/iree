// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/format/huggingface/decoder_json.h"

#include "iree/base/internal/json.h"
#include "iree/tokenizer/decoder/byte_fallback.h"
#include "iree/tokenizer/decoder/byte_level.h"
#include "iree/tokenizer/decoder/ctc.h"
#include "iree/tokenizer/decoder/metaspace.h"
#include "iree/tokenizer/decoder/replace.h"
#include "iree/tokenizer/decoder/sequence.h"
#include "iree/tokenizer/decoder/strip.h"
#include "iree/tokenizer/decoder/wordpiece.h"

//===----------------------------------------------------------------------===//
// ByteFallback Decoder
//===----------------------------------------------------------------------===//

// Parses a ByteFallback decoder.
// JSON structure: {"type": "ByteFallback"}
static iree_status_t iree_tokenizer_parse_byte_fallback_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  *out_decoder = NULL;
  (void)decoder_value;
  return iree_tokenizer_decoder_byte_fallback_allocate(allocator, out_decoder);
}

//===----------------------------------------------------------------------===//
// CTC Decoder
//===----------------------------------------------------------------------===//

// Parses a CTC decoder for speech-to-text models (Wav2Vec2, HuBERT, etc.).
// JSON structure:
//   {
//     "type": "CTC",
//     "pad_token": "<pad>",             // Required: blank/padding token
//     "word_delimiter_token": "|",      // Required: word boundary marker
//     "cleanup": true                   // Required: apply wordpiece cleanup
//   }
//
// Unlike most HF decoders, CTC does not have #[serde(default)] attributes,
// so all fields must be present in the JSON.
static iree_status_t iree_tokenizer_parse_ctc_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  *out_decoder = NULL;
  // Parse and unescape pad_token field (required).
  iree_string_view_t pad_token_json = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("pad_token"), &pad_token_json));

  char pad_token_buffer[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t pad_token_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(pad_token_json, sizeof(pad_token_buffer),
                                pad_token_buffer, &pad_token_length));

  // Parse and unescape word_delimiter_token field (required).
  iree_string_view_t word_delimiter_json = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("word_delimiter_token"), &word_delimiter_json));

  char word_delimiter_buffer[IREE_TOKENIZER_DECODER_CTC_MAX_TOKEN_SIZE];
  iree_host_size_t word_delimiter_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_unescape_string(
      word_delimiter_json, sizeof(word_delimiter_buffer), word_delimiter_buffer,
      &word_delimiter_length));

  // Parse cleanup flag (required - no default).
  iree_string_view_t cleanup_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("cleanup"), &cleanup_value));
  bool cleanup = false;
  if (iree_string_view_equal(cleanup_value, IREE_SV("true"))) {
    cleanup = true;
  } else if (!iree_string_view_equal(cleanup_value, IREE_SV("false"))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "CTC cleanup must be 'true' or 'false', got '%.*s'",
                            (int)cleanup_value.size, cleanup_value.data);
  }

  return iree_tokenizer_decoder_ctc_allocate(
      iree_make_string_view(pad_token_buffer, pad_token_length),
      iree_make_string_view(word_delimiter_buffer, word_delimiter_length),
      cleanup, allocator, out_decoder);
}

//===----------------------------------------------------------------------===//
// Metaspace Decoder
//===----------------------------------------------------------------------===//

// Parses a Metaspace decoder.
// JSON structure:
//   {
//     "type": "Metaspace",
//     "replacement": "▁",         // Required (char, no serde default in HF)
//     "prepend_scheme": "always", // Optional, default "always"
//     "split": true               // Ignored for decoders
//   }
static iree_status_t iree_tokenizer_parse_metaspace_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  *out_decoder = NULL;
  // Parse replacement character (required, no serde default in HF).
  uint32_t replacement_codepoint = 0;  // 0 means use default.
  iree_string_view_t replacement_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("replacement"), &replacement_value));
  IREE_RETURN_IF_ERROR(iree_json_parse_codepoint(replacement_value, 0x2581,
                                                 &replacement_codepoint));

  // Parse prepend_scheme (optional, default "always").
  iree_tokenizer_decoder_metaspace_prepend_scheme_t prepend_scheme =
      IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS;
  iree_string_view_t prepend_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      decoder_value, IREE_SV("prepend_scheme"), &prepend_value));
  if (prepend_value.size > 0 &&
      !iree_string_view_equal(prepend_value, IREE_SV("null"))) {
    if (iree_string_view_equal(prepend_value, IREE_SV("always"))) {
      prepend_scheme = IREE_TOKENIZER_DECODER_METASPACE_PREPEND_ALWAYS;
    } else if (iree_string_view_equal(prepend_value, IREE_SV("never"))) {
      prepend_scheme = IREE_TOKENIZER_DECODER_METASPACE_PREPEND_NEVER;
    } else if (iree_string_view_equal(prepend_value, IREE_SV("first"))) {
      prepend_scheme = IREE_TOKENIZER_DECODER_METASPACE_PREPEND_FIRST;
    } else {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "invalid Metaspace prepend_scheme: '%.*s'",
                              (int)prepend_value.size, prepend_value.data);
    }
  }

  return iree_tokenizer_decoder_metaspace_allocate(
      replacement_codepoint, prepend_scheme, allocator, out_decoder);
}

//===----------------------------------------------------------------------===//
// Replace Decoder
//===----------------------------------------------------------------------===//

// Maximum buffer size for Replace decoder pattern/content strings.
// Empirical: ALL observed decoder Replace patterns use ▁ (3 bytes) → " " (1
// byte). 16 bytes provides 5x headroom.
#define IREE_TOKENIZER_REPLACE_DECODER_MAX_STRING_LENGTH 16

// Parses a Replace decoder.
// JSON structure:
//   {
//     "type": "Replace",
//     "pattern": {"String": "▁"},  // Tagged enum: {"String": "..."} only
//     "content": " "               // Replacement string
//   }
//
// Only String patterns are supported. Regex patterns cannot be statically
// verified as shrinking, which violates the zero-allocation in-place
// transformation guarantee.
static iree_status_t iree_tokenizer_parse_replace_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  *out_decoder = NULL;
  // Parse pattern field (tagged enum object).
  iree_string_view_t pattern_object = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("pattern"), &pattern_object));

  // Check for Regex variant (reject it).
  iree_string_view_t regex_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pattern_object, IREE_SV("Regex"), &regex_value));
  if (regex_value.size > 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "Replace decoder with regex pattern not "
                            "supported (cannot verify shrinking statically)");
  }

  // Look up the String variant.
  iree_string_view_t pattern_json_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_try_lookup_object_value(
      pattern_object, IREE_SV("String"), &pattern_json_string));
  if (pattern_json_string.size == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Replace pattern must be tagged enum with 'String' or 'Regex' key");
  }

  // Unescape the pattern string.
  char pattern_buffer[IREE_TOKENIZER_REPLACE_DECODER_MAX_STRING_LENGTH];
  iree_host_size_t pattern_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(pattern_json_string, sizeof(pattern_buffer),
                                pattern_buffer, &pattern_length));

  // Parse and unescape content field.
  iree_string_view_t content_json_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("content"), &content_json_string));

  char content_buffer[IREE_TOKENIZER_REPLACE_DECODER_MAX_STRING_LENGTH];
  iree_host_size_t content_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(content_json_string, sizeof(content_buffer),
                                content_buffer, &content_length));

  return iree_tokenizer_decoder_replace_allocate(
      iree_make_string_view(pattern_buffer, pattern_length),
      iree_make_string_view(content_buffer, content_length), allocator,
      out_decoder);
}

//===----------------------------------------------------------------------===//
// Strip Decoder
//===----------------------------------------------------------------------===//

// Parses a Strip decoder.
// JSON structure:
//   {
//     "type": "Strip",
//     "content": " ",   // Required: character(s) to strip
//     "start": 1,       // Required: strip up to N from beginning
//     "stop": 0         // Required: strip up to N from end
//   }
static iree_status_t iree_tokenizer_parse_strip_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  *out_decoder = NULL;
  // Parse and unescape content field.
  iree_string_view_t content_json_string = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("content"), &content_json_string));

  char content_buffer[IREE_TOKENIZER_DECODER_STRIP_MAX_CONTENT_LENGTH];
  iree_host_size_t content_length = 0;
  IREE_RETURN_IF_ERROR(
      iree_json_unescape_string(content_json_string, sizeof(content_buffer),
                                content_buffer, &content_length));

  // Parse start count (required, no serde default in HF).
  iree_string_view_t start_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("start"), &start_value));
  int64_t start_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(start_value, &start_count));
  if (start_count < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strip start must be non-negative, got %" PRId64,
                            start_count);
  }

  // Parse stop count (required, no serde default in HF).
  iree_string_view_t stop_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("stop"), &stop_value));
  int64_t stop_count = 0;
  IREE_RETURN_IF_ERROR(iree_json_parse_int64(stop_value, &stop_count));
  if (stop_count < 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Strip stop must be non-negative, got %" PRId64,
                            stop_count);
  }

  return iree_tokenizer_decoder_strip_allocate(
      iree_make_string_view(content_buffer, content_length),
      (iree_host_size_t)start_count, (iree_host_size_t)stop_count, allocator,
      out_decoder);
}

//===----------------------------------------------------------------------===//
// WordPiece Decoder
//===----------------------------------------------------------------------===//

// Parses a WordPiece decoder.
// JSON structure:
//   {
//     "type": "WordPiece",
//     "prefix": "##",      // Required (no serde default in HF)
//     "cleanup": true      // Required (no serde default in HF)
//   }
static iree_status_t iree_tokenizer_parse_wordpiece_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  *out_decoder = NULL;
  // Parse prefix (required, no serde default in HF).
  char prefix_buffer[IREE_TOKENIZER_WORDPIECE_MAX_PREFIX_LENGTH];
  iree_host_size_t prefix_length = 0;
  IREE_RETURN_IF_ERROR(iree_json_lookup_string(
      decoder_value, IREE_SV("prefix"),
      iree_make_mutable_string_view(prefix_buffer, sizeof(prefix_buffer)),
      &prefix_length));

  // Parse cleanup flag (required, no serde default in HF).
  bool cleanup = false;
  IREE_RETURN_IF_ERROR(
      iree_json_lookup_bool(decoder_value, IREE_SV("cleanup"), &cleanup));

  iree_tokenizer_decoder_wordpiece_config_t config = {
      .prefix = iree_make_string_view(prefix_buffer, prefix_length),
      .cleanup = cleanup,
  };
  return iree_tokenizer_decoder_wordpiece_allocate(config, allocator,
                                                   out_decoder);
}

//===----------------------------------------------------------------------===//
// Sequence Decoder
//===----------------------------------------------------------------------===//

// Context for sequence decoder array enumeration.
typedef struct iree_tokenizer_decoder_sequence_parse_context_t {
  iree_allocator_t allocator;
  iree_tokenizer_decoder_t** children;
  iree_host_size_t child_count;
  iree_host_size_t parsed_count;
} iree_tokenizer_decoder_sequence_parse_context_t;

// Visitor callback for parsing sequence decoder children.
static iree_status_t iree_tokenizer_parse_decoder_sequence_child_visitor(
    void* user_data, iree_host_size_t index, iree_string_view_t value) {
  iree_tokenizer_decoder_sequence_parse_context_t* context =
      (iree_tokenizer_decoder_sequence_parse_context_t*)user_data;
  if (index >= context->child_count) {
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "array index %zu exceeds expected count %zu", index,
                            context->child_count);
  }
  iree_status_t status = iree_tokenizer_huggingface_parse_decoder(
      value, context->allocator, &context->children[index]);
  if (iree_status_is_ok(status)) {
    ++context->parsed_count;
  } else {
    status =
        iree_status_annotate_f(status, "in Sequence decoder child %zu", index);
  }
  return status;
}

// Parses a Sequence decoder.
// JSON structure:
//   {
//     "type": "Sequence",
//     "decoders": [
//       {"type": "Replace", ...},
//       {"type": "ByteFallback"},
//       ...
//     ]
//   }
//
// NULL children (from no-op decoders like Fuse) are filtered out.
static iree_status_t iree_tokenizer_parse_sequence_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  *out_decoder = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Look up the decoders array.
  iree_string_view_t decoders_value = iree_string_view_empty();
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_lookup_object_value(decoder_value, IREE_SV("decoders"),
                                        &decoders_value));

  // Get array length.
  iree_host_size_t json_child_count = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_json_array_length(decoders_value, &json_child_count));

  // Validate count against maximum depth.
  if (json_child_count > IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                             "Sequence decoder has %zu children, maximum is %d",
                             json_child_count,
                             IREE_TOKENIZER_DECODER_SEQUENCE_MAX_DEPTH));
  }

  // Allocate array for child decoder pointers.
  iree_tokenizer_decoder_t** children = NULL;
  if (json_child_count > 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_allocator_malloc_array(allocator, json_child_count,
                                        sizeof(*children), (void**)&children));
    memset(children, 0, json_child_count * sizeof(*children));
  }

  // Parse each child decoder using enumeration.
  iree_tokenizer_decoder_sequence_parse_context_t context = {
      .allocator = allocator,
      .children = children,
      .child_count = json_child_count,
      .parsed_count = 0,
  };
  iree_status_t status =
      json_child_count > 0
          ? iree_json_enumerate_array(
                decoders_value,
                iree_tokenizer_parse_decoder_sequence_child_visitor, &context)
          : iree_ok_status();

  // Compact array: filter out NULL children (no-op decoders like Fuse).
  iree_host_size_t non_null_count = 0;
  if (iree_status_is_ok(status) && children) {
    for (iree_host_size_t i = 0; i < json_child_count; ++i) {
      if (children[i]) {
        children[non_null_count++] = children[i];
      }
    }
  }

  // Dispatch based on child count after compaction:
  //   0 children → NULL (passthrough)
  //   1 child    → return the child directly (no sequence wrapper)
  //   2+ children → create sequence
  if (iree_status_is_ok(status)) {
    if (non_null_count == 0) {
      *out_decoder = NULL;
    } else if (non_null_count == 1) {
      *out_decoder = children[0];
      children[0] = NULL;  // Prevent cleanup from freeing it.
    } else {
      status = iree_tokenizer_decoder_sequence_allocate(
          children, non_null_count, allocator, out_decoder);
    }
  }

  // Cleanup on failure: free any successfully parsed children.
  if (!iree_status_is_ok(status) && children) {
    for (iree_host_size_t i = 0; i < json_child_count; ++i) {
      if (children[i]) {
        iree_tokenizer_decoder_free(children[i]);
      }
    }
  }

  // Free the temporary array (sequence made its own copy for 2+ case).
  if (children) {
    iree_allocator_free(allocator, children);
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Top-Level Decoder Dispatcher
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_huggingface_parse_decoder(
    iree_string_view_t decoder_value, iree_allocator_t allocator,
    iree_tokenizer_decoder_t** out_decoder) {
  IREE_ASSERT_ARGUMENT(out_decoder);
  *out_decoder = NULL;

  // Handle null decoder.
  if (iree_string_view_equal(decoder_value, IREE_SV("null"))) {
    return iree_ok_status();
  }

  // Get decoder type.
  iree_string_view_t type_value = iree_string_view_empty();
  IREE_RETURN_IF_ERROR(iree_json_lookup_object_value(
      decoder_value, IREE_SV("type"), &type_value));

  // Dispatch based on type.
  if (iree_string_view_equal(type_value, IREE_SV("ByteFallback"))) {
    return iree_tokenizer_parse_byte_fallback_decoder(decoder_value, allocator,
                                                      out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("CTC"))) {
    return iree_tokenizer_parse_ctc_decoder(decoder_value, allocator,
                                            out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Metaspace"))) {
    return iree_tokenizer_parse_metaspace_decoder(decoder_value, allocator,
                                                  out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Sequence"))) {
    return iree_tokenizer_parse_sequence_decoder(decoder_value, allocator,
                                                 out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Replace"))) {
    return iree_tokenizer_parse_replace_decoder(decoder_value, allocator,
                                                out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("Fuse"))) {
    // Fuse concatenates tokens without separators. Our streaming architecture
    // uses seen_first_output state instead, making Fuse a no-op.
    *out_decoder = NULL;
    return iree_ok_status();
  } else if (iree_string_view_equal(type_value, IREE_SV("Strip"))) {
    return iree_tokenizer_parse_strip_decoder(decoder_value, allocator,
                                              out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("ByteLevel"))) {
    // ByteLevel decoder only reverses byte-to-unicode encoding. The
    // add_prefix_space and trim_offsets fields affect encoding/offset-tracking
    // (pre_tokenizer role), not decoding. We validate their presence for HF
    // schema compliance but correctly ignore their values. This is NOT a silent
    // failure - ignoring them in the decoder role IS correct behavior.
    bool add_prefix_space = false;
    IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
        decoder_value, IREE_SV("add_prefix_space"), &add_prefix_space));
    bool trim_offsets = false;
    IREE_RETURN_IF_ERROR(iree_json_lookup_bool(
        decoder_value, IREE_SV("trim_offsets"), &trim_offsets));
    return iree_tokenizer_decoder_byte_level_allocate(allocator, out_decoder);
  } else if (iree_string_view_equal(type_value, IREE_SV("WordPiece"))) {
    return iree_tokenizer_parse_wordpiece_decoder(decoder_value, allocator,
                                                  out_decoder);
  }

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "decoder type not supported: '%.*s'",
                          (int)type_value.size, type_value.data);
}
