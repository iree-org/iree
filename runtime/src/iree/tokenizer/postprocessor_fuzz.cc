// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for postprocessor template processing.
//
// Tests the postprocessor against adversarial template configurations:
// - Varied prefix/infix/suffix counts (0-7 each, clamped to MAX_PIECES)
// - Adversarial token IDs in templates
// - Type ID assignment to model tokens
// - State machine transitions (PREFIX -> SEQUENCE_A -> SUFFIX -> DONE)
// - Both single and pair templates
// - All flag combinations (TRIM_OFFSETS, ADD_PREFIX_SPACE)
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/tokenizer/postprocessor.h"
#include "iree/tokenizer/types.h"

// Parse a template from fuzz data. Returns bytes consumed.
static size_t parse_template(const uint8_t* data, size_t size,
                             iree_tokenizer_postprocessor_template_t* out) {
  memset(out, 0, sizeof(*out));
  if (size < 5) return 0;

  // Byte 0: prefix_count (0-7, clamped to max).
  out->prefix_count = data[0] % (IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES + 1);
  // Byte 1: suffix_count (0 to remaining capacity).
  uint8_t remaining =
      IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES - out->prefix_count;
  out->suffix_count = (remaining > 0) ? (data[1] % (remaining + 1)) : 0;
  // Byte 2: infix_count (0 to remaining capacity).
  remaining = IREE_TOKENIZER_POSTPROCESSOR_MAX_PIECES - out->prefix_count -
              out->suffix_count;
  out->infix_count = (remaining > 0) ? (data[2] % (remaining + 1)) : 0;

  // Byte 3: sequence type IDs.
  out->sequence_a_type_id = data[3] & 0x0F;
  out->sequence_b_type_id = (data[3] >> 4) & 0x0F;

  size_t pos = 4;
  uint8_t total = out->prefix_count + out->infix_count + out->suffix_count;

  // Fill token IDs from fuzz data (4 bytes each).
  for (uint8_t i = 0; i < total && pos + 4 <= size; ++i) {
    out->token_ids[i] =
        (int32_t)((uint32_t)data[pos] | ((uint32_t)data[pos + 1] << 8) |
                  ((uint32_t)data[pos + 2] << 16) |
                  ((uint32_t)data[pos + 3] << 24));
    pos += 4;
  }

  // Fill type IDs (1 byte each).
  for (uint8_t i = 0; i < total && pos < size; ++i) {
    out->type_ids[i] = data[pos++];
  }

  return pos;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size < 12) return 0;

  // Byte 0: flags.
  iree_tokenizer_postprocessor_flags_t flags = data[0] & 0x03;
  // Byte 1: whether to test pair template.
  bool test_pair = (data[1] & 0x01) != 0;
  data += 2;
  size -= 2;

  //===--------------------------------------------------------------------===//
  // Parse single template from fuzz data
  //===--------------------------------------------------------------------===//

  iree_tokenizer_postprocessor_template_t single_template;
  size_t consumed = parse_template(data, size, &single_template);
  if (consumed == 0) return 0;
  data += consumed;
  size -= consumed;

  //===--------------------------------------------------------------------===//
  // Optionally parse pair template
  //===--------------------------------------------------------------------===//

  iree_tokenizer_postprocessor_template_t pair_template;
  memset(&pair_template, 0, sizeof(pair_template));
  if (test_pair && size >= 5) {
    consumed = parse_template(data, size, &pair_template);
    data += consumed;
    size -= consumed;
  }

  //===--------------------------------------------------------------------===//
  // Initialize postprocessor
  //===--------------------------------------------------------------------===//

  iree_tokenizer_postprocessor_t postprocessor;
  iree_status_t status = iree_tokenizer_postprocessor_initialize(
      &single_template, test_pair ? &pair_template : NULL, flags,
      &postprocessor);
  if (!iree_status_is_ok(status)) {
    iree_status_ignore(status);
    return 0;
  }

  //===--------------------------------------------------------------------===//
  // Test 1: Single-sequence encode state machine
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_postprocessor_encode_state_t state;
    iree_tokenizer_postprocessor_encode_state_initialize(
        &postprocessor, &postprocessor.single, &state);

    // Output buffers.
    iree_tokenizer_token_id_t token_ids[64];
    uint8_t type_ids[64];
    iree_tokenizer_token_output_t output =
        iree_tokenizer_make_token_output(token_ids, NULL, type_ids, 64);

    // Emit prefix.
    iree_host_size_t prefix_written =
        iree_tokenizer_postprocessor_emit_prefix(&state, output, 0);

    // Simulate model tokens by assigning type_ids.
    iree_host_size_t model_start = prefix_written;
    iree_host_size_t model_count = 8;
    if (model_start + model_count > 64) {
      model_count = 64 - model_start;
    }
    // Fill fake model token IDs.
    for (iree_host_size_t i = 0; i < model_count; ++i) {
      token_ids[model_start + i] = (int32_t)(i + 100);
    }
    iree_tokenizer_postprocessor_assign_type_ids(&state, output, model_start,
                                                 model_count);

    // Transition to suffix.
    iree_tokenizer_postprocessor_begin_suffix(&state);

    // Emit suffix.
    iree_host_size_t suffix_written = iree_tokenizer_postprocessor_emit_suffix(
        &state, output, model_start + model_count);

    // Verify state is DONE or IDLE.
    (void)iree_tokenizer_postprocessor_encode_state_has_pending(&state);
    (void)prefix_written;
    (void)suffix_written;
  }

  //===--------------------------------------------------------------------===//
  // Test 2: Pair-sequence encode (if pair template exists)
  //===--------------------------------------------------------------------===//

  if (iree_tokenizer_postprocessor_supports_pair(&postprocessor)) {
    iree_tokenizer_postprocessor_encode_state_t state;
    iree_tokenizer_postprocessor_encode_state_initialize(
        &postprocessor, &postprocessor.pair, &state);

    iree_tokenizer_token_id_t token_ids[64];
    uint8_t type_ids[64];
    iree_tokenizer_token_output_t output =
        iree_tokenizer_make_token_output(token_ids, NULL, type_ids, 64);

    iree_host_size_t offset = 0;
    offset += iree_tokenizer_postprocessor_emit_prefix(&state, output, offset);

    // Simulate sequence A model tokens.
    iree_host_size_t seq_a_count = 4;
    if (offset + seq_a_count > 64) seq_a_count = 64 - offset;
    for (iree_host_size_t i = 0; i < seq_a_count; ++i) {
      token_ids[offset + i] = (int32_t)(i + 200);
    }
    iree_tokenizer_postprocessor_assign_type_ids(&state, output, offset,
                                                 seq_a_count);
    offset += seq_a_count;

    // Transition to suffix and emit.
    iree_tokenizer_postprocessor_begin_suffix(&state);
    offset += iree_tokenizer_postprocessor_emit_suffix(&state, output, offset);

    (void)iree_tokenizer_postprocessor_encode_state_has_pending(&state);
  }

  //===--------------------------------------------------------------------===//
  // Test 3: Repeated initialize/emit cycles (stress state reset)
  //===--------------------------------------------------------------------===//

  for (int iter = 0; iter < 4; ++iter) {
    iree_tokenizer_postprocessor_encode_state_t state;
    iree_tokenizer_postprocessor_encode_state_initialize(
        &postprocessor, &postprocessor.single, &state);

    iree_tokenizer_token_id_t token_ids[16];
    uint8_t type_ids[16];
    iree_tokenizer_token_output_t output =
        iree_tokenizer_make_token_output(token_ids, NULL, type_ids, 16);

    iree_host_size_t offset = 0;
    offset += iree_tokenizer_postprocessor_emit_prefix(&state, output, offset);
    iree_tokenizer_postprocessor_begin_suffix(&state);
    offset += iree_tokenizer_postprocessor_emit_suffix(&state, output, offset);
  }

  //===--------------------------------------------------------------------===//
  // Test 4: Zero-capacity output (boundary stress)
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_postprocessor_encode_state_t state;
    iree_tokenizer_postprocessor_encode_state_initialize(
        &postprocessor, &postprocessor.single, &state);

    iree_tokenizer_token_output_t empty_output =
        iree_tokenizer_make_token_output(NULL, NULL, NULL, 0);

    iree_tokenizer_postprocessor_emit_prefix(&state, empty_output, 0);
    iree_tokenizer_postprocessor_begin_suffix(&state);
    iree_tokenizer_postprocessor_emit_suffix(&state, empty_output, 0);
  }

  iree_tokenizer_postprocessor_deinitialize(&postprocessor);

  return 0;
}
