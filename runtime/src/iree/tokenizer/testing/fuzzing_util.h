// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared utility for tokenizer fuzz targets.
//
// Provides common infrastructure for all tokenizer fuzz targets:
// - Loading a real tokenizer from HuggingFace JSON (--tokenizer_json=<path>)
// - Building a complete dummy tokenizer when no JSON is specified
// - Parsing custom flags (--track_offsets) and removing them from argv
//
// Usage from LLVMFuzzerInitialize:
//
//   static iree_tokenizer_t* g_tokenizer = NULL;
//   static int32_t g_vocab_size = 0;
//
//   extern "C" int LLVMFuzzerInitialize(int* argc, char*** argv) {
//     iree_status_t status = iree_tokenizer_fuzz_load_or_build(
//         argc, argv, &g_tokenizer, &g_vocab_size);
//     if (!iree_status_is_ok(status)) {
//       iree_status_fprint(stderr, status);
//       iree_status_ignore(status);
//       return 1;
//     }
//     return 0;
//   }
//
// Invoked via iree-bazel-fuzz with optional flags:
//   iree-bazel-fuzz //path:target -- --tokenizer_json=llama3.json
//   iree-bazel-fuzz //path:target -- --track_offsets -max_total_time=120

#ifndef IREE_TOKENIZER_TESTING_FUZZING_UTIL_H_
#define IREE_TOKENIZER_TESTING_FUZZING_UTIL_H_

#include "iree/base/api.h"
#include "iree/tokenizer/tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loads a tokenizer for fuzzing, either from a JSON file or by building a
// dummy. Scans |*argv| for custom flags, parses them, and removes them so
// libfuzzer never sees them.
//
// Custom flags:
//   --tokenizer_json=<path>  Load HuggingFace tokenizer.json from file.
//   --track_offsets           Enable offset tracking during encode.
//
// When --tokenizer_json is absent, builds a complete dummy tokenizer with:
//   - 256 byte-level tokens (IDs 0-255)
//   - 8 multi-character tokens (IDs 256-263)
//   - Special tokens: UNK=0, BOS=1, EOS=2
//   - Whitespace segmenter
//   - BPE model with FUSE_UNK (no merges, character-level)
//   - Byte-level decoder
//
// |out_vocab_size| receives the vocab capacity (max_token_id + 1). May be NULL
// if the caller doesn't need it.
iree_status_t iree_tokenizer_fuzz_load_or_build(
    int* argc, char*** argv, iree_tokenizer_t** out_tokenizer,
    int32_t* out_vocab_size);

// Returns true if --track_offsets was passed. Only valid after
// iree_tokenizer_fuzz_load_or_build has been called.
bool iree_tokenizer_fuzz_track_offsets(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_TESTING_FUZZING_UTIL_H_
