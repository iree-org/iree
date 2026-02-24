// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/testing/fuzzing_util.h"

#include <stdio.h>
#include <string.h>

#include "iree/tokenizer/decoder/byte_level.h"
#include "iree/tokenizer/format/huggingface/tokenizer_json.h"
#include "iree/tokenizer/model/bpe.h"
#include "iree/tokenizer/segmenter/whitespace.h"
#include "iree/tokenizer/vocab/vocab.h"
#include "iree/tokenizer/vocab/vocab_builder.h"

// Global flag state, set during iree_tokenizer_fuzz_load_or_build.
static bool g_track_offsets = false;

bool iree_tokenizer_fuzz_track_offsets(void) { return g_track_offsets; }

//===----------------------------------------------------------------------===//
// argv Flag Parsing
//===----------------------------------------------------------------------===//

// Removes argv[index] by shifting subsequent entries down and decrementing
// *argc. This ensures libfuzzer never sees our custom flags.
static void iree_fuzz_remove_argv(int* argc, char*** argv, int index) {
  for (int i = index; i < *argc - 1; ++i) {
    (*argv)[i] = (*argv)[i + 1];
  }
  --(*argc);
}

// Scans argv for custom flags, extracts their values, and removes them.
// Returns the --tokenizer_json path (or NULL if absent).
static const char* iree_fuzz_parse_and_remove_flags(int* argc, char*** argv) {
  const char* tokenizer_json_path = NULL;

  for (int i = 1; i < *argc; /* incremented conditionally */) {
    if (strncmp((*argv)[i], "--tokenizer_json=", 17) == 0) {
      tokenizer_json_path = (*argv)[i] + 17;
      iree_fuzz_remove_argv(argc, argv, i);
    } else if (strcmp((*argv)[i], "--track_offsets") == 0) {
      g_track_offsets = true;
      iree_fuzz_remove_argv(argc, argv, i);
    } else {
      ++i;
    }
  }

  return tokenizer_json_path;
}

//===----------------------------------------------------------------------===//
// Real Tokenizer Loading
//===----------------------------------------------------------------------===//

static iree_status_t iree_fuzz_load_tokenizer_from_file(
    const char* path, iree_tokenizer_t** out_tokenizer,
    int32_t* out_vocab_size) {
  // Read the file contents.
  FILE* file = fopen(path, "rb");
  if (!file) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "failed to open tokenizer JSON: %s", path);
  }

  fseek(file, 0, SEEK_END);
  long file_size = ftell(file);
  fseek(file, 0, SEEK_SET);

  if (file_size <= 0 || file_size > 100 * 1024 * 1024) {
    fclose(file);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "tokenizer JSON file size out of range: %ld bytes",
                            file_size);
  }

  char* buffer = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), (iree_host_size_t)file_size, (void**)&buffer);
  if (!iree_status_is_ok(status)) {
    fclose(file);
    return status;
  }

  size_t bytes_read = fread(buffer, 1, (size_t)file_size, file);
  fclose(file);

  if (bytes_read != (size_t)file_size) {
    iree_allocator_free(iree_allocator_system(), buffer);
    return iree_make_status(IREE_STATUS_DATA_LOSS,
                            "short read on tokenizer JSON: %zu of %ld bytes",
                            bytes_read, file_size);
  }

  // Parse the JSON.
  iree_string_view_t json =
      iree_make_string_view(buffer, (iree_host_size_t)file_size);
  status = iree_tokenizer_from_huggingface_json(json, iree_allocator_system(),
                                                out_tokenizer);
  iree_allocator_free(iree_allocator_system(), buffer);

  if (iree_status_is_ok(status) && out_vocab_size) {
    *out_vocab_size = (int32_t)iree_tokenizer_vocab_capacity(
        iree_tokenizer_vocab(*out_tokenizer));
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Dummy Tokenizer Construction
//===----------------------------------------------------------------------===//

static iree_status_t iree_fuzz_build_dummy_tokenizer(
    iree_tokenizer_t** out_tokenizer, int32_t* out_vocab_size) {
  iree_tokenizer_vocab_builder_t* vocab_builder = NULL;
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_allocate(
      /*capacity_hint=*/512, iree_allocator_system(), &vocab_builder));

  // Add 256 byte-level tokens (IDs 0-255).
  for (int i = 0; i < 256; ++i) {
    char c = (char)i;
    iree_status_t status = iree_tokenizer_vocab_builder_add_token(
        vocab_builder, iree_make_string_view(&c, 1), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_vocab_builder_free(vocab_builder);
      return status;
    }
  }

  // Add multi-character tokens to exercise merge/decode logic.
  static const char* const kMultiTokens[] = {
      "hello", "world", "the ", " and ", "\n\n", "  ", "...", "!!!",
  };
  for (size_t i = 0; i < sizeof(kMultiTokens) / sizeof(kMultiTokens[0]); ++i) {
    iree_status_t status = iree_tokenizer_vocab_builder_add_token(
        vocab_builder, iree_make_cstring_view(kMultiTokens[i]), 0.0f,
        IREE_TOKENIZER_TOKEN_ATTR_NONE);
    if (!iree_status_is_ok(status)) {
      iree_tokenizer_vocab_builder_free(vocab_builder);
      return status;
    }
  }

  // Mark special tokens.
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_set_special_token(
      vocab_builder, IREE_TOKENIZER_SPECIAL_TOKEN_UNK, 0));
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_set_special_token(
      vocab_builder, IREE_TOKENIZER_SPECIAL_TOKEN_BOS, 1));
  IREE_RETURN_IF_ERROR(iree_tokenizer_vocab_builder_set_special_token(
      vocab_builder, IREE_TOKENIZER_SPECIAL_TOKEN_EOS, 2));

  iree_tokenizer_vocab_t* vocab = NULL;
  iree_status_t status =
      iree_tokenizer_vocab_builder_build(vocab_builder, &vocab);
  if (!iree_status_is_ok(status)) return status;

  // Build the tokenizer.
  iree_tokenizer_builder_t builder;
  iree_tokenizer_builder_initialize(iree_allocator_system(), &builder);

  // Whitespace segmenter.
  iree_tokenizer_segmenter_t* segmenter = NULL;
  status = iree_tokenizer_segmenter_whitespace_allocate(iree_allocator_system(),
                                                        &segmenter);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_builder_deinitialize(&builder);
    iree_tokenizer_vocab_free(vocab);
    return status;
  }
  iree_tokenizer_builder_set_segmenter(&builder, segmenter);

  // BPE model (no merges = character-level tokenization).
  iree_tokenizer_model_t* model = NULL;
  status = iree_tokenizer_bpe_model_allocate(
      vocab, IREE_TOKENIZER_BPE_FLAG_FUSE_UNK, iree_allocator_system(), &model);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_builder_deinitialize(&builder);
    iree_tokenizer_vocab_free(vocab);
    return status;
  }
  iree_tokenizer_builder_set_model(&builder, model);
  iree_tokenizer_builder_set_vocab(&builder, vocab);

  // Byte-level decoder.
  iree_tokenizer_decoder_t* decoder = NULL;
  status = iree_tokenizer_decoder_byte_level_allocate(iree_allocator_system(),
                                                      &decoder);
  if (!iree_status_is_ok(status)) {
    iree_tokenizer_builder_deinitialize(&builder);
    return status;
  }
  iree_tokenizer_builder_set_decoder(&builder, decoder);

  // Build.
  status = iree_tokenizer_builder_build(&builder, out_tokenizer);
  iree_tokenizer_builder_deinitialize(&builder);

  if (iree_status_is_ok(status) && out_vocab_size) {
    *out_vocab_size = (int32_t)iree_tokenizer_vocab_capacity(vocab);
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

iree_status_t iree_tokenizer_fuzz_load_or_build(
    int* argc, char*** argv, iree_tokenizer_t** out_tokenizer,
    int32_t* out_vocab_size) {
  const char* tokenizer_json_path =
      iree_fuzz_parse_and_remove_flags(argc, argv);

  if (tokenizer_json_path) {
    fprintf(stderr, "fuzzing_util: loading tokenizer from %s\n",
            tokenizer_json_path);
    return iree_fuzz_load_tokenizer_from_file(tokenizer_json_path,
                                              out_tokenizer, out_vocab_size);
  }

  fprintf(stderr, "fuzzing_util: using dummy tokenizer (264 tokens)\n");
  return iree_fuzz_build_dummy_tokenizer(out_tokenizer, out_vocab_size);
}
