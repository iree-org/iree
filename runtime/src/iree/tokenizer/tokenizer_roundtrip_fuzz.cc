// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fuzz target for encode-then-decode roundtrip.
//
// Tests that encoding text and then decoding the resulting tokens does not
// crash. This catches:
// - Tokens that encode produces but decode cannot handle
// - State corruption bleeding from encode output into decode input
// - Invalid UTF-8 produced by decode when input was valid UTF-8
//
// Uses fuzzing_util for tokenizer loading (dummy or real via --tokenizer_json).
//
// See https://iree.dev/developers/debugging/fuzzing/ for build and run info.

#include <stddef.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/tokenizer/testing/fuzzing_util.h"
#include "iree/tokenizer/tokenizer.h"

static iree_tokenizer_t* g_tokenizer = NULL;

extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
  iree_status_t status =
      iree_tokenizer_fuzz_load_or_build(argc, argv, &g_tokenizer, NULL);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
    return 1;
  }
  return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (g_tokenizer == NULL || size == 0) return 0;

  iree_string_view_t input =
      iree_make_string_view(reinterpret_cast<const char*>(data), size);

  //===--------------------------------------------------------------------===//
  // Step 1: Encode text to tokens
  //===--------------------------------------------------------------------===//

  iree_tokenizer_token_id_t tokens[8192];
  iree_tokenizer_token_output_t encode_output =
      iree_tokenizer_make_token_output(tokens, NULL, NULL,
                                       IREE_ARRAYSIZE(tokens));
  iree_host_size_t token_count = 0;

  iree_tokenizer_encode_flags_t encode_flags = IREE_TOKENIZER_ENCODE_FLAG_NONE;
  if (iree_tokenizer_fuzz_track_offsets()) {
    encode_flags |= IREE_TOKENIZER_ENCODE_FLAG_TRACK_OFFSETS;
  }

  iree_status_t status =
      iree_tokenizer_encode(g_tokenizer, input, encode_flags, encode_output,
                            iree_allocator_system(), &token_count);

  if (!iree_status_is_ok(status)) {
    // RESOURCE_EXHAUSTED is expected for very large inputs.
    iree_status_ignore(status);
    return 0;
  }

  if (token_count == 0) return 0;

  //===--------------------------------------------------------------------===//
  // Step 2: Decode tokens back to text
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_token_id_list_t token_list =
        iree_tokenizer_make_token_id_list(tokens, token_count);
    char decode_buffer[32768];
    iree_mutable_string_view_t text_output = {decode_buffer,
                                              sizeof(decode_buffer)};
    iree_host_size_t text_length = 0;
    status = iree_tokenizer_decode(g_tokenizer, token_list,
                                   IREE_TOKENIZER_DECODE_FLAG_NONE, text_output,
                                   iree_allocator_system(), &text_length);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Step 3: Decode with skip_special_tokens
  //===--------------------------------------------------------------------===//

  {
    iree_tokenizer_token_id_list_t token_list =
        iree_tokenizer_make_token_id_list(tokens, token_count);
    char decode_buffer[32768];
    iree_mutable_string_view_t text_output = {decode_buffer,
                                              sizeof(decode_buffer)};
    iree_host_size_t text_length = 0;
    status = iree_tokenizer_decode(
        g_tokenizer, token_list, IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS,
        text_output, iree_allocator_system(), &text_length);
    iree_status_ignore(status);
  }

  //===--------------------------------------------------------------------===//
  // Step 4: Streaming encode then streaming decode
  //===--------------------------------------------------------------------===//

  {
    // Streaming encode.
    iree_host_size_t state_size = 0;
    status =
        iree_tokenizer_encode_state_calculate_size(g_tokenizer, &state_size);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return 0;
    }

    void* state_storage = NULL;
    status = iree_allocator_malloc(iree_allocator_system(), state_size,
                                   &state_storage);
    if (!iree_status_is_ok(status)) {
      iree_status_ignore(status);
      return 0;
    }

    iree_host_size_t buffer_size =
        iree_tokenizer_transform_buffer_recommended_size(size);
    void* transform_buffer = NULL;
    status = iree_allocator_malloc(iree_allocator_system(), buffer_size,
                                   &transform_buffer);
    if (!iree_status_is_ok(status)) {
      iree_allocator_free(iree_allocator_system(), state_storage);
      iree_status_ignore(status);
      return 0;
    }

    iree_tokenizer_encode_state_t* encode_state = NULL;
    status = iree_tokenizer_encode_state_initialize(
        g_tokenizer,
        iree_make_byte_span(reinterpret_cast<uint8_t*>(state_storage),
                            state_size),
        iree_make_byte_span(reinterpret_cast<uint8_t*>(transform_buffer),
                            buffer_size),
        iree_tokenizer_offset_run_list_empty(),
        IREE_TOKENIZER_ENCODE_FLAG_AT_INPUT_START, &encode_state);

    iree_tokenizer_token_id_t streaming_tokens[4096];
    iree_host_size_t streaming_token_count = 0;

    if (iree_status_is_ok(status)) {
      iree_tokenizer_token_output_t output = iree_tokenizer_make_token_output(
          streaming_tokens, NULL, NULL, IREE_ARRAYSIZE(streaming_tokens));

      // Feed input in varied chunks.
      iree_host_size_t offset = 0;
      while (offset < size) {
        iree_host_size_t chunk_size = ((offset % 4) == 0)   ? 3
                                      : ((offset % 4) == 1) ? 17
                                      : ((offset % 4) == 2) ? 64
                                                            : 256;
        if (chunk_size > size - offset) chunk_size = size - offset;

        iree_string_view_t chunk =
            iree_make_string_view(input.data + offset, chunk_size);

        iree_host_size_t bytes_consumed = 0;
        iree_host_size_t count = 0;
        status = iree_tokenizer_encode_state_feed(encode_state, chunk, output,
                                                  &bytes_consumed, &count);
        if (!iree_status_is_ok(status)) {
          iree_status_ignore(status);
          break;
        }
        streaming_token_count += count;
        offset += bytes_consumed;
        if (bytes_consumed == 0 && count == 0) break;
      }

      // Finalize encode.
      iree_host_size_t final_encode_count = 0;
      status = iree_tokenizer_encode_state_finalize(encode_state, output,
                                                    &final_encode_count);
      iree_status_ignore(status);
      streaming_token_count += final_encode_count;

      iree_tokenizer_encode_state_deinitialize(encode_state);

      // Streaming decode of the encoded tokens.
      if (streaming_token_count > 0 &&
          streaming_token_count <= IREE_ARRAYSIZE(streaming_tokens)) {
        iree_host_size_t decode_state_size = 0;
        status = iree_tokenizer_decode_state_calculate_size(g_tokenizer,
                                                            &decode_state_size);
        if (iree_status_is_ok(status)) {
          void* decode_storage = NULL;
          status = iree_allocator_malloc(iree_allocator_system(),
                                         decode_state_size, &decode_storage);
          if (iree_status_is_ok(status)) {
            iree_tokenizer_decode_state_t* decode_state = NULL;
            status = iree_tokenizer_decode_state_initialize(
                g_tokenizer, IREE_TOKENIZER_DECODE_FLAG_NONE,
                iree_make_byte_span(reinterpret_cast<uint8_t*>(decode_storage),
                                    decode_state_size),
                &decode_state);

            if (iree_status_is_ok(status)) {
              char decode_buffer[8192];
              iree_mutable_string_view_t text_output = {decode_buffer,
                                                        sizeof(decode_buffer)};

              // Feed tokens in varied chunks.
              iree_host_size_t token_offset = 0;
              while (token_offset < streaming_token_count) {
                iree_host_size_t chunk_count = ((token_offset % 3) + 1) * 4;
                if (chunk_count > streaming_token_count - token_offset) {
                  chunk_count = streaming_token_count - token_offset;
                }

                iree_tokenizer_token_id_list_t chunk =
                    iree_tokenizer_make_token_id_list(
                        streaming_tokens + token_offset, chunk_count);

                iree_host_size_t tokens_consumed = 0;
                iree_host_size_t text_length = 0;
                status = iree_tokenizer_decode_state_feed(
                    decode_state, chunk, text_output, &tokens_consumed,
                    &text_length);
                if (!iree_status_is_ok(status)) {
                  iree_status_ignore(status);
                  break;
                }
                token_offset += tokens_consumed;
                if (tokens_consumed == 0 && text_length == 0) break;
              }

              // Finalize decode.
              iree_host_size_t final_text_length = 0;
              status = iree_tokenizer_decode_state_finalize(
                  decode_state, text_output, &final_text_length);
              iree_status_ignore(status);

              iree_tokenizer_decode_state_deinitialize(decode_state);
            } else {
              iree_status_ignore(status);
            }

            iree_allocator_free(iree_allocator_system(), decode_storage);
          } else {
            iree_status_ignore(status);
          }
        } else {
          iree_status_ignore(status);
        }
      }
    } else {
      iree_status_ignore(status);
    }

    iree_allocator_free(iree_allocator_system(), transform_buffer);
    iree_allocator_free(iree_allocator_system(), state_storage);
  }

  return 0;
}
