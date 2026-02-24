// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/postprocessor.h"

#include "iree/base/api.h"
#include "iree/tokenizer/vocab/vocab.h"

//===----------------------------------------------------------------------===//
// Validation
//===----------------------------------------------------------------------===//

static iree_status_t iree_tokenizer_postprocessor_validate_template(
    const iree_tokenizer_postprocessor_template_t* t,
    const char* template_name) {
  uint8_t total = iree_tokenizer_postprocessor_template_total_count(t);
  if (total > IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "postprocessor %s template has %u total pieces (max %u)", template_name,
        (unsigned)total, (unsigned)IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES);
  }
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Lifecycle
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_postprocessor_initialize(
    const iree_tokenizer_postprocessor_template_t* single_template,
    const iree_tokenizer_postprocessor_template_t* pair_template,
    iree_tokenizer_postprocessor_flags_t flags,
    iree_tokenizer_postprocessor_t* out_postprocessor) {
  IREE_ASSERT_ARGUMENT(single_template);
  IREE_ASSERT_ARGUMENT(out_postprocessor);

  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_postprocessor, 0, sizeof(*out_postprocessor));

  // Validate template sizes.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_tokenizer_postprocessor_validate_template(single_template,
                                                         "single"));
  if (pair_template) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_tokenizer_postprocessor_validate_template(pair_template, "pair"));
  }

  // Copy templates (fixed-size inline structs).
  memcpy(&out_postprocessor->single, single_template,
         sizeof(iree_tokenizer_postprocessor_template_t));
  if (pair_template) {
    memcpy(&out_postprocessor->pair, pair_template,
           sizeof(iree_tokenizer_postprocessor_template_t));
  }

  out_postprocessor->flags = flags;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

void iree_tokenizer_postprocessor_deinitialize(
    iree_tokenizer_postprocessor_t* postprocessor) {
  if (!postprocessor) return;
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(postprocessor, 0, sizeof(*postprocessor));
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Encode Operations
//===----------------------------------------------------------------------===//

// Emits special tokens from a template phase into the token output.
// Writes tokens starting at |output_offset|, bounded by output.capacity.
// |base| selects the starting index in the template's arrays.
// |position| tracks progress (in/out) and reaches |count| when complete.
// Special tokens get zero-length offsets (no corresponding source text).
static iree_host_size_t iree_tokenizer_postprocessor_emit_phase(
    const iree_tokenizer_postprocessor_template_t* template_data, uint8_t base,
    uint8_t count, uint8_t* IREE_RESTRICT position,
    iree_tokenizer_token_output_t output, iree_host_size_t output_offset) {
  iree_host_size_t emitted = 0;
  while (*position < count) {
    // Check for available capacity with overflow protection.
    iree_host_size_t out_index = 0;
    if (!iree_host_size_checked_add(output_offset, emitted, &out_index) ||
        out_index >= output.capacity) {
      break;  // Overflow or no room.
    }
    uint8_t template_index = base + *position;
    output.token_ids[out_index] = template_data->token_ids[template_index];
    if (output.type_ids) {
      output.type_ids[out_index] = template_data->type_ids[template_index];
    }
    if (output.token_offsets) {
      output.token_offsets[out_index].start = 0;
      output.token_offsets[out_index].end = 0;
    }
    ++(*position);
    ++emitted;
  }
  return emitted;
}

iree_host_size_t iree_tokenizer_postprocessor_emit_prefix(
    iree_tokenizer_postprocessor_encode_state_t* state,
    iree_tokenizer_token_output_t output, iree_host_size_t output_offset) {
  if (state->phase != IREE_TOKENIZER_POSTPROCESSOR_PHASE_PREFIX) return 0;
  iree_host_size_t emitted = iree_tokenizer_postprocessor_emit_phase(
      state->active_template, /*base=*/0, state->active_template->prefix_count,
      &state->position, output, output_offset);
  if (state->position >= state->active_template->prefix_count) {
    state->phase = IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A;
    state->position = 0;
  }
  return emitted;
}

void iree_tokenizer_postprocessor_assign_type_ids(
    const iree_tokenizer_postprocessor_encode_state_t* state,
    iree_tokenizer_token_output_t output, iree_host_size_t offset,
    iree_host_size_t count) {
  if (!output.type_ids || count == 0) return;
  if (state->phase != IREE_TOKENIZER_POSTPROCESSOR_PHASE_SEQUENCE_A) return;
  // Bounds check with overflow protection: offset + count must not exceed
  // capacity and must not overflow.
  iree_host_size_t end = 0;
  if (!iree_host_size_checked_add(offset, count, &end) ||
      end > output.capacity) {
    return;  // Overflow or out of bounds.
  }
  memset(&output.type_ids[offset], state->active_template->sequence_a_type_id,
         count);
}

iree_host_size_t iree_tokenizer_postprocessor_emit_suffix(
    iree_tokenizer_postprocessor_encode_state_t* state,
    iree_tokenizer_token_output_t output, iree_host_size_t output_offset) {
  if (state->phase != IREE_TOKENIZER_POSTPROCESSOR_PHASE_SUFFIX) return 0;
  uint8_t base = state->active_template->prefix_count +
                 state->active_template->infix_count;
  iree_host_size_t emitted = iree_tokenizer_postprocessor_emit_phase(
      state->active_template, base, state->active_template->suffix_count,
      &state->position, output, output_offset);
  if (state->position >= state->active_template->suffix_count) {
    state->phase = IREE_TOKENIZER_POSTPROCESSOR_PHASE_DONE;
  }
  return emitted;
}

//===----------------------------------------------------------------------===//
// Offset Trimming
//===----------------------------------------------------------------------===//

// Counts leading whitespace in a token string, returning the number of bytes
// in the original input that should be trimmed from offset.start.
//
// ByteLevel encoding uses special Unicode characters to represent whitespace:
//   U+0120 (Ġ, UTF-8: 0xC4 0xA0) represents space (0x20)
//   U+010A (Ċ, UTF-8: 0xC4 0x8A) represents newline (0x0A)
//   U+010D (ċ, UTF-8: 0xC4 0x8D) represents carriage return (0x0D)
// Each sentinel in the token text corresponds to exactly 1 byte in the
// original input.
//
// Returns the count in terms of ORIGINAL INPUT bytes, not token text bytes.
static iree_host_size_t iree_tokenizer_postprocessor_count_leading_whitespace(
    iree_string_view_t text) {
  iree_host_size_t count = 0;
  iree_host_size_t position = 0;
  while (position < text.size) {
    uint8_t byte = (uint8_t)text.data[position];
    if (byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D) {
      // ASCII whitespace: space, tab, newline, carriage return.
      // 1 byte in token text = 1 byte in original input.
      ++count;
      ++position;
    } else if (position + 1 < text.size && byte == 0xC4) {
      // Check for ByteLevel whitespace sentinels (all start with 0xC4).
      uint8_t next_byte = (uint8_t)text.data[position + 1];
      if (next_byte == 0xA0 || next_byte == 0x8A || next_byte == 0x8D) {
        // Ġ (0xA0) = space, Ċ (0x8A) = newline, ċ (0x8D) = carriage return.
        // 2 bytes in token text = 1 byte in original input.
        ++count;
        position += 2;
      } else {
        break;
      }
    } else {
      // Non-whitespace character.
      break;
    }
  }
  return count;
}

// Counts trailing whitespace in a token string, returning the number of bytes
// in the original input that should be trimmed from offset.end.
static iree_host_size_t iree_tokenizer_postprocessor_count_trailing_whitespace(
    iree_string_view_t text) {
  iree_host_size_t count = 0;
  iree_host_size_t position = text.size;
  while (position > 0) {
    uint8_t byte = (uint8_t)text.data[position - 1];
    if (byte == 0x20 || byte == 0x09 || byte == 0x0A || byte == 0x0D) {
      // ASCII whitespace.
      ++count;
      --position;
    } else if (position >= 2 && (uint8_t)text.data[position - 2] == 0xC4) {
      // Check for ByteLevel whitespace sentinels (second byte of UTF-8).
      if (byte == 0xA0 || byte == 0x8A || byte == 0x8D) {
        // Ġ (0xA0) = space, Ċ (0x8A) = newline, ċ (0x8D) = carriage return.
        ++count;
        position -= 2;
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return count;
}

void iree_tokenizer_postprocessor_trim_token_offsets(
    iree_tokenizer_postprocessor_encode_state_t* state,
    const iree_tokenizer_vocab_t* vocab, iree_tokenizer_token_output_t output,
    iree_host_size_t model_token_start, iree_host_size_t model_token_count) {
  // Early exit if trimming is disabled or no offsets to trim.
  if (!state ||
      !iree_any_bit_set(state->flags,
                        IREE_TOKENIZER_POSTPROCESSOR_FLAG_TRIM_OFFSETS)) {
    return;
  }
  if (!output.token_offsets || model_token_count == 0) return;
  if (!vocab) return;

  for (iree_host_size_t i = 0; i < model_token_count; ++i) {
    // Compute token_index with overflow protection.
    iree_host_size_t token_index = 0;
    if (!iree_host_size_checked_add(model_token_start, i, &token_index) ||
        token_index >= output.capacity) {
      break;
    }

    iree_tokenizer_token_id_t token_id = output.token_ids[token_index];
    iree_string_view_t token_text =
        iree_tokenizer_vocab_token_text(vocab, token_id);
    if (token_text.size == 0) continue;

    iree_host_size_t leading =
        iree_tokenizer_postprocessor_count_leading_whitespace(token_text);
    iree_host_size_t trailing =
        iree_tokenizer_postprocessor_count_trailing_whitespace(token_text);

    // Special case: if ADD_PREFIX_SPACE is set and this is the first model
    // token across the entire encode, preserve exactly 1 leading space (it was
    // artificially added and is part of the token's content).
    if (!state->first_model_token_trimmed &&
        iree_any_bit_set(state->flags,
                         IREE_TOKENIZER_POSTPROCESSOR_FLAG_ADD_PREFIX_SPACE) &&
        leading > 0) {
      --leading;
    }
    state->first_model_token_trimmed = true;

    iree_tokenizer_offset_t* offset = &output.token_offsets[token_index];

    // Trim leading: advance start but don't exceed end. Use checked arithmetic
    // to avoid overflow when offset->start + leading exceeds SIZE_MAX.
    iree_host_size_t new_start = 0;
    if (!iree_host_size_checked_add(offset->start, leading, &new_start) ||
        new_start > offset->end) {
      new_start = offset->end;
    }

    // Trim trailing: move end back but don't go below new_start.
    // The span available for trimming is [new_start, offset->end].
    iree_host_size_t new_end = offset->end;
    if (trailing > 0) {
      iree_host_size_t span = new_end - new_start;
      if (trailing >= span) {
        new_end = new_start;
      } else {
        new_end -= trailing;
      }
    }

    offset->start = new_start;
    offset->end = new_end;
  }
}
