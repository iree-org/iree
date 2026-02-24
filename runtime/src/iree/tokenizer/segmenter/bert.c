// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/bert.h"

#include <string.h>

#include "iree/base/internal/unicode.h"

//===----------------------------------------------------------------------===//
// Character Classification
//===----------------------------------------------------------------------===//

// BERT character classes for the pre-tokenizer.
typedef enum {
  IREE_TOKENIZER_BERT_CHAR_WHITESPACE,
  IREE_TOKENIZER_BERT_CHAR_PUNCTUATION,
  IREE_TOKENIZER_BERT_CHAR_WORD,
  IREE_TOKENIZER_BERT_CHAR_INCOMPLETE,
} iree_tokenizer_bert_char_class_t;

// Returns true if the codepoint is BERT punctuation.
// Matches HuggingFace's is_bert_punc(): ASCII punctuation OR Unicode P.
//
// ASCII punctuation (matching Rust's char::is_ascii_punctuation):
//   33-47:   ! " # $ % & ' ( ) * + , - . /
//   58-64:   : ; < = > ? @
//   91-96:   [ \ ] ^ _ `
//   123-126: { | } ~
static inline bool iree_tokenizer_is_bert_punctuation(uint32_t codepoint) {
  if ((codepoint >= 33 && codepoint <= 47) ||
      (codepoint >= 58 && codepoint <= 64) ||
      (codepoint >= 91 && codepoint <= 96) ||
      (codepoint >= 123 && codepoint <= 126)) {
    return true;
  }
  if (codepoint >= 0x80) {
    return iree_unicode_is_punctuation(codepoint);
  }
  return false;
}

// Decodes the next codepoint at |position| in |data[0..size)| and returns its
// BERT character class. On success, |*out_byte_length| is set to the number of
// bytes the codepoint occupies. Returns INCOMPLETE if the UTF-8 sequence
// extends past |size| (caller should stop processing and defer to next chunk).
static inline iree_tokenizer_bert_char_class_t
iree_tokenizer_bert_classify_next(const char* data, iree_host_size_t position,
                                  iree_host_size_t size,
                                  iree_host_size_t* out_byte_length) {
  uint8_t lead_byte = (uint8_t)data[position];
  iree_host_size_t sequence_length =
      iree_unicode_utf8_sequence_length(lead_byte);
  if (position + sequence_length > size) {
    *out_byte_length = 0;
    return IREE_TOKENIZER_BERT_CHAR_INCOMPLETE;
  }

  iree_string_view_t view = {data, size};
  iree_host_size_t decode_position = position;
  uint32_t codepoint = iree_unicode_utf8_decode(view, &decode_position);
  *out_byte_length = decode_position - position;

  if (iree_unicode_is_whitespace(codepoint)) {
    return IREE_TOKENIZER_BERT_CHAR_WHITESPACE;
  }
  if (iree_tokenizer_is_bert_punctuation(codepoint)) {
    return IREE_TOKENIZER_BERT_CHAR_PUNCTUATION;
  }
  return IREE_TOKENIZER_BERT_CHAR_WORD;
}

// Tries to append a segment [start, end) to |output| at index |*segment_count|.
// Returns true if the segment was written (capacity available), advancing
// |*segment_count|. Returns false if output is full (caller should stop).
static inline bool iree_tokenizer_bert_emit_segment(
    iree_tokenizer_segment_output_t output, iree_host_size_t* segment_count,
    iree_host_size_t start, iree_host_size_t end) {
  if (*segment_count >= output.capacity) return false;
  output.values[*segment_count].start = start;
  output.values[*segment_count].end = end;
  ++(*segment_count);
  return true;
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

// State preserved across capacity-limited finalize() calls.
typedef struct iree_tokenizer_bert_finalize_state_t {
  // True if we have a segment computed but not yet emitted.
  bool has_pending;
  // Byte range of the pending segment within remaining_input.
  iree_host_size_t pending_start;
  iree_host_size_t pending_end;
  // Where to resume scanning in remaining_input on next finalize() call.
  iree_host_size_t scan_position;
} iree_tokenizer_bert_finalize_state_t;

typedef struct iree_tokenizer_segmenter_bert_t {
  iree_tokenizer_segmenter_t base;
  iree_allocator_t allocator;
} iree_tokenizer_segmenter_bert_t;

typedef struct iree_tokenizer_segmenter_bert_state_t {
  iree_tokenizer_segmenter_state_t base;
  // True if we've started a word segment but haven't found its end yet.
  bool in_word;
  // Start position of current word segment (cumulative offset across all
  // process() calls). Only valid when in_word is true.
  iree_host_size_t word_start;
  // Total bytes consumed (returned via out_consumed) across all process()
  // calls.
  iree_host_size_t bytes_consumed;
  // State for incremental finalize() with limited output capacity.
  iree_tokenizer_bert_finalize_state_t finalize;
} iree_tokenizer_segmenter_bert_state_t;

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_bert_vtable;

iree_status_t iree_tokenizer_segmenter_bert_allocate(
    iree_allocator_t allocator, iree_tokenizer_segmenter_t** out_segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_segmenter);
  *out_segmenter = NULL;

  iree_tokenizer_segmenter_bert_t* segmenter = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(allocator, sizeof(*segmenter), (void**)&segmenter));

  iree_tokenizer_segmenter_initialize(
      &segmenter->base, &iree_tokenizer_segmenter_bert_vtable,
      sizeof(iree_tokenizer_segmenter_bert_state_t));
  segmenter->allocator = allocator;

  *out_segmenter = &segmenter->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_bert_destroy(
    iree_tokenizer_segmenter_t* segmenter) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_bert_t* self =
      (iree_tokenizer_segmenter_bert_t*)segmenter;
  iree_allocator_t allocator = self->allocator;
  iree_allocator_free(allocator, self);
  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_tokenizer_segmenter_bert_state_initialize(
    const iree_tokenizer_segmenter_t* segmenter, void* storage,
    iree_tokenizer_segmenter_state_t** out_state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_tokenizer_segmenter_bert_state_t* state =
      (iree_tokenizer_segmenter_bert_state_t*)storage;
  memset(state, 0, sizeof(*state));
  state->base.segmenter = segmenter;
  *out_state = &state->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_tokenizer_segmenter_bert_state_deinitialize(
    iree_tokenizer_segmenter_state_t* state) {
  IREE_TRACE_ZONE_BEGIN(z0);
  (void)state;
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Process
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_bert_state_process(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t input,
    iree_tokenizer_segment_output_t output, iree_host_size_t* out_consumed,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_bert_state_t* self =
      (iree_tokenizer_segmenter_bert_state_t*)state;

  *out_consumed = 0;
  *out_segment_count = 0;

  if (input.size == 0) {
    return iree_ok_status();
  }

  iree_host_size_t segment_count = 0;
  iree_host_size_t position = 0;

  while (position < input.size) {
    iree_host_size_t byte_length = 0;
    iree_tokenizer_bert_char_class_t char_class =
        iree_tokenizer_bert_classify_next(input.data, position, input.size,
                                          &byte_length);

    if (char_class == IREE_TOKENIZER_BERT_CHAR_INCOMPLETE) {
      break;
    }

    if (char_class == IREE_TOKENIZER_BERT_CHAR_WHITESPACE) {
      // End current word (if any), skip the whitespace.
      if (self->in_word) {
        iree_host_size_t relative_start =
            self->word_start - self->bytes_consumed;
        if (!iree_tokenizer_bert_emit_segment(output, &segment_count,
                                              relative_start, position)) {
          // No capacity — stop before consuming the word.
          self->in_word = false;
          *out_consumed = relative_start;
          self->bytes_consumed += relative_start;
          *out_segment_count = segment_count;
          return iree_ok_status();
        }
        self->in_word = false;
      }
      position += byte_length;
    } else if (char_class == IREE_TOKENIZER_BERT_CHAR_PUNCTUATION) {
      // End current word (if any), then emit punctuation as isolated segment.
      if (self->in_word) {
        iree_host_size_t relative_start =
            self->word_start - self->bytes_consumed;
        if (!iree_tokenizer_bert_emit_segment(output, &segment_count,
                                              relative_start, position)) {
          self->in_word = false;
          *out_consumed = relative_start;
          self->bytes_consumed += relative_start;
          *out_segment_count = segment_count;
          return iree_ok_status();
        }
        self->in_word = false;
      }
      // Emit punctuation as its own segment.
      if (!iree_tokenizer_bert_emit_segment(output, &segment_count, position,
                                            position + byte_length)) {
        // No room for punctuation — stop without consuming it.
        *out_consumed = position;
        self->bytes_consumed += position;
        *out_segment_count = segment_count;
        return iree_ok_status();
      }
      position += byte_length;
    } else {
      // Word character: start or continue a word.
      if (!self->in_word) {
        self->in_word = true;
        self->word_start = self->bytes_consumed + position;
      }
      position += byte_length;
    }
  }

  // If mid-word at end of input, don't consume those bytes (leave for
  // finalize which receives them as remaining_input).
  if (self->in_word) {
    iree_host_size_t relative_start = self->word_start - self->bytes_consumed;
    self->bytes_consumed += relative_start;
    *out_consumed = relative_start;
  } else {
    self->bytes_consumed += position;
    *out_consumed = position;
  }
  *out_segment_count = segment_count;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Finalize
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_segmenter_bert_state_finalize(
    iree_tokenizer_segmenter_state_t* state, iree_string_view_t remaining_input,
    iree_tokenizer_segment_output_t output,
    iree_host_size_t* out_segment_count) {
  iree_tokenizer_segmenter_bert_state_t* self =
      (iree_tokenizer_segmenter_bert_state_t*)state;

  iree_host_size_t segment_count = 0;

  // Emit any pending segment from a previous capacity-limited call.
  if (self->finalize.has_pending) {
    if (!iree_tokenizer_bert_emit_segment(output, &segment_count,
                                          self->finalize.pending_start,
                                          self->finalize.pending_end)) {
      *out_segment_count = 0;
      return iree_ok_status();
    }
    self->finalize.has_pending = false;
  }

  // Resume scanning from saved position (0 on first call).
  iree_host_size_t position = self->finalize.scan_position;

  while (position < remaining_input.size) {
    iree_host_size_t byte_length = 0;
    iree_tokenizer_bert_char_class_t char_class =
        iree_tokenizer_bert_classify_next(remaining_input.data, position,
                                          remaining_input.size, &byte_length);

    if (char_class == IREE_TOKENIZER_BERT_CHAR_INCOMPLETE) {
      // Incomplete UTF-8 at end — emit remaining bytes as a word segment.
      if (!iree_tokenizer_bert_emit_segment(output, &segment_count, position,
                                            remaining_input.size)) {
        self->finalize.has_pending = true;
        self->finalize.pending_start = position;
        self->finalize.pending_end = remaining_input.size;
        self->finalize.scan_position = remaining_input.size;
        self->in_word = true;
        *out_segment_count = segment_count;
        return iree_ok_status();
      }
      position = remaining_input.size;
      break;
    }

    if (char_class == IREE_TOKENIZER_BERT_CHAR_WHITESPACE) {
      position += byte_length;
    } else if (char_class == IREE_TOKENIZER_BERT_CHAR_PUNCTUATION) {
      if (!iree_tokenizer_bert_emit_segment(output, &segment_count, position,
                                            position + byte_length)) {
        self->finalize.has_pending = true;
        self->finalize.pending_start = position;
        self->finalize.pending_end = position + byte_length;
        self->finalize.scan_position = position + byte_length;
        self->in_word = true;
        *out_segment_count = segment_count;
        return iree_ok_status();
      }
      position += byte_length;
    } else {
      // Word character — scan to end of word.
      iree_host_size_t word_start = position;
      position += byte_length;
      while (position < remaining_input.size) {
        char_class = iree_tokenizer_bert_classify_next(
            remaining_input.data, position, remaining_input.size, &byte_length);
        if (char_class != IREE_TOKENIZER_BERT_CHAR_WORD) {
          // INCOMPLETE also breaks the word (include trailing invalid bytes).
          if (char_class == IREE_TOKENIZER_BERT_CHAR_INCOMPLETE) {
            position = remaining_input.size;
          }
          break;
        }
        position += byte_length;
      }

      if (!iree_tokenizer_bert_emit_segment(output, &segment_count, word_start,
                                            position)) {
        self->finalize.has_pending = true;
        self->finalize.pending_start = word_start;
        self->finalize.pending_end = position;
        self->finalize.scan_position = position;
        self->in_word = true;
        *out_segment_count = segment_count;
        return iree_ok_status();
      }
    }
  }

  // All done.
  self->finalize.scan_position = 0;
  self->in_word = false;
  *out_segment_count = segment_count;
  return iree_ok_status();
}

static bool iree_tokenizer_segmenter_bert_state_has_pending(
    const iree_tokenizer_segmenter_state_t* state) {
  const iree_tokenizer_segmenter_bert_state_t* self =
      (const iree_tokenizer_segmenter_bert_state_t*)state;
  return self->in_word;
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

static const iree_tokenizer_segmenter_vtable_t
    iree_tokenizer_segmenter_bert_vtable = {
        .destroy = iree_tokenizer_segmenter_bert_destroy,
        .state_initialize = iree_tokenizer_segmenter_bert_state_initialize,
        .state_deinitialize = iree_tokenizer_segmenter_bert_state_deinitialize,
        .state_process = iree_tokenizer_segmenter_bert_state_process,
        .state_finalize = iree_tokenizer_segmenter_bert_state_finalize,
        .state_has_pending = iree_tokenizer_segmenter_bert_state_has_pending,
};
