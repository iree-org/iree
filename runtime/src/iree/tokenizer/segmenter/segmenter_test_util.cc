// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/tokenizer/segmenter/segmenter_test_util.h"

#include <algorithm>

#include "iree/base/internal/unicode.h"

namespace iree::tokenizer::testing {

namespace {

// Maximum segments we expect from a single process/finalize call.
constexpr size_t kMaxSegments = 256;

// Converts segments to strings given the input text.
std::vector<std::string> SegmentsToStrings(
    const char* input, const iree_tokenizer_segment_t* segments,
    size_t segment_count) {
  std::vector<std::string> result;
  result.reserve(segment_count);
  for (size_t i = 0; i < segment_count; ++i) {
    result.push_back(std::string(input + segments[i].start,
                                 segments[i].end - segments[i].start));
  }
  return result;
}

}  // namespace

std::vector<std::string> ProcessAndFinalize(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    bool expect_pending_after_process) {
  ScopedSegmenterState state(segmenter);

  // Process the entire input.
  iree_tokenizer_segment_t segments[kMaxSegments];
  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  IREE_EXPECT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input,
      iree_tokenizer_make_segment_output(segments, kMaxSegments), &consumed,
      &segment_count));

  // Pull-based: process() may leave bytes unconsumed (for pending segments).
  // No assertion on consumed == input.size - unconsumed bytes go to finalize.

  // Verify has_pending() after process.
  bool has_pending = iree_tokenizer_segmenter_state_has_pending(state.get());
  EXPECT_EQ(has_pending, expect_pending_after_process)
      << "has_pending() after process() should be "
      << (expect_pending_after_process ? "true" : "false") << " for input: \""
      << std::string(input.data, input.size) << "\"";

  // Collect segments from process().
  std::vector<std::string> result =
      SegmentsToStrings(input.data, segments, segment_count);

  // Always call finalize with remaining unconsumed input.
  // Pull-based: process() leaves unconsumed bytes, finalize receives them.
  iree_string_view_t remaining_input =
      iree_make_string_view(input.data + consumed, input.size - consumed);
  iree_host_size_t finalize_count = 0;
  IREE_EXPECT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining_input,
      iree_tokenizer_make_segment_output(segments, kMaxSegments),
      &finalize_count));

  // Add any segments from finalize (offsets relative to remaining_input).
  for (size_t i = 0; i < finalize_count; ++i) {
    size_t abs_start = consumed + segments[i].start;
    size_t abs_end = consumed + segments[i].end;
    result.push_back(std::string(input.data + abs_start, abs_end - abs_start));
  }

  // Verify has_pending() is false after finalize.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

std::vector<std::string> ProcessChunkedAndFinalize(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    size_t chunk_size, bool expect_pending_after_process) {
  ScopedSegmenterState state(segmenter);

  std::vector<std::string> result;
  iree_tokenizer_segment_t segments[kMaxSegments];
  size_t position = 0;

  // Process input in chunks.
  // Pull-based: segmenter may not consume all bytes, unconsumed go to finalize.
  while (position < input.size) {
    size_t remaining = input.size - position;
    size_t this_chunk = std::min(remaining, chunk_size);
    iree_string_view_t chunk =
        iree_make_string_view(input.data + position, this_chunk);

    iree_host_size_t consumed = 0;
    iree_host_size_t segment_count = 0;

    IREE_EXPECT_OK(iree_tokenizer_segmenter_state_process(
        state.get(), chunk,
        iree_tokenizer_make_segment_output(segments, kMaxSegments), &consumed,
        &segment_count));

    // Convert segments to strings with position adjustment.
    // Segments are relative to chunk start (at position).
    for (size_t i = 0; i < segment_count; ++i) {
      size_t abs_start = position + segments[i].start;
      size_t abs_end = position + segments[i].end;
      result.push_back(
          std::string(input.data + abs_start, abs_end - abs_start));
    }

    position += consumed;

    // Pull-based: if consumed=0, segmenter can't make progress with current
    // chunk. In real streaming, we'd feed more data. For tests, break and
    // send remaining to finalize.
    if (consumed == 0) {
      break;
    }
  }

  // Verify has_pending() after all process() calls.
  // Only check if we've processed all input (didn't break early).
  if (position >= input.size) {
    bool has_pending = iree_tokenizer_segmenter_state_has_pending(state.get());
    EXPECT_EQ(has_pending, expect_pending_after_process)
        << "has_pending() after all process() calls should be "
        << (expect_pending_after_process ? "true" : "false") << " for input: \""
        << std::string(input.data, input.size) << "\""
        << " with chunk_size=" << chunk_size;
  }
  // If we broke early (position < input.size), we expect has_pending=true
  // because not all input was processed. Skip the assertion.

  // Always call finalize with remaining unconsumed input.
  // Pull-based: process() leaves unconsumed bytes, finalize receives them.
  // Use position since we may have broken early (consumed==0).
  iree_string_view_t remaining_input =
      iree_make_string_view(input.data + position, input.size - position);
  iree_host_size_t finalize_count = 0;
  IREE_EXPECT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining_input,
      iree_tokenizer_make_segment_output(segments, kMaxSegments),
      &finalize_count));

  // Add segments from finalize (offsets relative to remaining_input).
  for (size_t i = 0; i < finalize_count; ++i) {
    size_t abs_start = position + segments[i].start;
    size_t abs_end = position + segments[i].end;
    result.push_back(std::string(input.data + abs_start, abs_end - abs_start));
  }

  // Verify has_pending() is false after finalize.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

std::vector<std::string> ProcessChunkedAndFinalizeUtf8(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    size_t chunk_size, bool expect_pending_after_process) {
  ScopedSegmenterState state(segmenter);

  std::vector<std::string> result;
  iree_tokenizer_segment_t segments[kMaxSegments];
  size_t position = 0;

  // Process input in chunks, respecting UTF-8 boundaries.
  // Pull-based: segmenter may not consume all bytes, unconsumed go to finalize.
  while (position < input.size) {
    size_t remaining = input.size - position;
    size_t this_chunk = std::min(remaining, chunk_size);

    // Adjust chunk size to not split UTF-8 codepoints (simulates tokenizer).
    size_t incomplete_tail = iree_unicode_utf8_incomplete_tail_length(
        input.data + position, this_chunk);
    this_chunk -= incomplete_tail;

    // If we can't make progress, take at least one full codepoint.
    if (this_chunk == 0 && remaining > 0) {
      // Start of a multi-byte sequence; include the whole codepoint.
      uint8_t first_byte = (uint8_t)input.data[position];
      if ((first_byte & 0xE0) == 0xC0)
        this_chunk = std::min(remaining, size_t{2});
      else if ((first_byte & 0xF0) == 0xE0)
        this_chunk = std::min(remaining, size_t{3});
      else if ((first_byte & 0xF8) == 0xF0)
        this_chunk = std::min(remaining, size_t{4});
      else
        this_chunk = 1;  // ASCII or invalid, take one byte.
    }

    iree_string_view_t chunk =
        iree_make_string_view(input.data + position, this_chunk);

    iree_host_size_t consumed = 0;
    iree_host_size_t segment_count = 0;

    IREE_EXPECT_OK(iree_tokenizer_segmenter_state_process(
        state.get(), chunk,
        iree_tokenizer_make_segment_output(segments, kMaxSegments), &consumed,
        &segment_count));

    // Convert segments to strings with position adjustment.
    // Segments are relative to chunk start (at position).
    for (size_t i = 0; i < segment_count; ++i) {
      size_t abs_start = position + segments[i].start;
      size_t abs_end = position + segments[i].end;
      result.push_back(
          std::string(input.data + abs_start, abs_end - abs_start));
    }

    position += consumed;

    // Pull-based: if consumed=0, segmenter can't make progress with current
    // chunk. In real streaming, we'd feed more data. For tests, break and
    // send remaining to finalize.
    if (consumed == 0) {
      break;
    }
  }

  // Verify has_pending() after all process() calls.
  // Only check if we've processed all input (didn't break early).
  if (position >= input.size) {
    bool has_pending = iree_tokenizer_segmenter_state_has_pending(state.get());
    EXPECT_EQ(has_pending, expect_pending_after_process)
        << "has_pending() after all process() calls should be "
        << (expect_pending_after_process ? "true" : "false") << " for input: \""
        << std::string(input.data, input.size) << "\""
        << " with chunk_size=" << chunk_size << " (UTF-8 aware)";
  }
  // If we broke early (position < input.size), we expect has_pending=true
  // because not all input was processed. Skip the assertion.

  // Always call finalize with remaining unconsumed input.
  // Pull-based: process() leaves unconsumed bytes, finalize receives them.
  // Use position since we may have broken early (consumed==0).
  iree_string_view_t remaining_input =
      iree_make_string_view(input.data + position, input.size - position);
  iree_host_size_t finalize_count = 0;
  IREE_EXPECT_OK(iree_tokenizer_segmenter_state_finalize(
      state.get(), remaining_input,
      iree_tokenizer_make_segment_output(segments, kMaxSegments),
      &finalize_count));

  // Add segments from finalize (offsets relative to remaining_input).
  for (size_t i = 0; i < finalize_count; ++i) {
    size_t abs_start = position + segments[i].start;
    size_t abs_end = position + segments[i].end;
    result.push_back(std::string(input.data + abs_start, abs_end - abs_start));
  }

  // Verify has_pending() is false after finalize.
  EXPECT_FALSE(iree_tokenizer_segmenter_state_has_pending(state.get()))
      << "has_pending() should be false after finalize()";

  return result;
}

void TestWithAllChunkSizes(iree_tokenizer_segmenter_t* segmenter,
                           iree_string_view_t input,
                           const std::vector<std::string>& expected_segments,
                           bool expect_pending_after_process) {
  // First test with no chunking (baseline).
  std::vector<std::string> baseline =
      ProcessAndFinalize(segmenter, input, expect_pending_after_process);
  EXPECT_EQ(baseline, expected_segments)
      << "Baseline (no chunking) produced unexpected segments";

  // Test with each standard chunk size.
  for (size_t chunk_size : kStandardChunkSizes) {
    SCOPED_TRACE(::testing::Message() << "chunk_size=" << chunk_size);
    std::vector<std::string> chunked = ProcessChunkedAndFinalize(
        segmenter, input, chunk_size, expect_pending_after_process);
    EXPECT_EQ(chunked, expected_segments)
        << "Chunked processing (chunk_size=" << chunk_size
        << ") produced different segments than expected";
  }
}

void TestWithAllChunkSizesUtf8(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    const std::vector<std::string>& expected_segments,
    bool expect_pending_after_process) {
  // First test with no chunking (baseline).
  std::vector<std::string> baseline =
      ProcessAndFinalize(segmenter, input, expect_pending_after_process);
  EXPECT_EQ(baseline, expected_segments)
      << "Baseline (no chunking) produced unexpected segments";

  // Test with each standard chunk size using UTF-8-aware chunking.
  for (size_t chunk_size : kStandardChunkSizes) {
    SCOPED_TRACE(::testing::Message()
                 << "chunk_size=" << chunk_size << " (UTF-8 aware)");
    std::vector<std::string> chunked = ProcessChunkedAndFinalizeUtf8(
        segmenter, input, chunk_size, expect_pending_after_process);
    EXPECT_EQ(chunked, expected_segments)
        << "UTF-8-aware chunked processing (chunk_size=" << chunk_size
        << ") produced different segments than expected";
  }
}

void TestLimitedOutputCapacity(
    iree_tokenizer_segmenter_t* segmenter, iree_string_view_t input,
    const std::vector<std::string>& expected_segments) {
  ScopedSegmenterState state(segmenter);

  std::vector<std::string> result;
  iree_tokenizer_segment_t segment;

  // Process with capacity=1, one segment at a time.
  // Pull-based: when consumed=0, remaining bytes go to finalize.
  size_t position = 0;

  while (position < input.size) {
    iree_string_view_t remaining =
        iree_make_string_view(input.data + position, input.size - position);

    iree_host_size_t consumed = 0;
    iree_host_size_t segment_count = 0;

    IREE_EXPECT_OK(iree_tokenizer_segmenter_state_process(
        state.get(), remaining, iree_tokenizer_make_segment_output(&segment, 1),
        &consumed, &segment_count));

    if (segment_count > 0) {
      // Segment is relative to 'remaining', adjust to input.
      size_t abs_start = position + segment.start;
      size_t abs_end = position + segment.end;
      result.push_back(
          std::string(input.data + abs_start, abs_end - abs_start));
    }

    // Pull-based: if consumed=0, the segmenter can't emit more segments now.
    // Remaining bytes will go to finalize.
    if (consumed == 0) {
      break;
    }

    position += consumed;
  }

  // Finalize with remaining unconsumed input. The first call handles any
  // remaining data (even when has_pending is false after the process fix).
  // Subsequent calls are driven by has_pending() — each segmenter tracks its
  // position internally and uses remaining_input for coordinate calculations
  // across calls. Finalize may need multiple calls with capacity=1 because
  // RESOURCE_EXHAUSTED or the sequence segmenter's resumption protocol
  // (returns OK but sets have_current_segment for the next call).
  iree_string_view_t remaining_input =
      iree_make_string_view(input.data + position, input.size - position);
  bool need_finalize = remaining_input.size > 0;
  // Safety limit: more iterations than possible segments indicates a bug.
  // Each segment is at least 1 byte, so input.size + some slack is an upper
  // bound on reasonable iterations.
  size_t max_iterations = input.size + 100;
  size_t iteration = 0;
  while (need_finalize ||
         iree_tokenizer_segmenter_state_has_pending(state.get())) {
    need_finalize = false;
    ASSERT_LT(++iteration, max_iterations)
        << "Infinite loop detected in finalize";

    iree_host_size_t finalize_count = 0;
    iree_status_t finalize_status = iree_tokenizer_segmenter_state_finalize(
        state.get(), remaining_input,
        iree_tokenizer_make_segment_output(&segment, 1), &finalize_count);

    if (!iree_status_is_ok(finalize_status) &&
        !iree_status_is_resource_exhausted(finalize_status)) {
      IREE_EXPECT_OK(finalize_status);
      break;
    }
    iree_status_ignore(finalize_status);

    if (finalize_count > 0) {
      size_t abs_start = position + segment.start;
      size_t abs_end = position + segment.end;
      result.push_back(
          std::string(input.data + abs_start, abs_end - abs_start));
    } else {
      break;
    }
  }

  EXPECT_EQ(result, expected_segments)
      << "Limited capacity (1) produced different segments";
}

void TestZeroCapacityOutput(iree_tokenizer_segmenter_t* segmenter,
                            iree_string_view_t input) {
  ScopedSegmenterState state(segmenter);

  iree_host_size_t consumed = 0;
  iree_host_size_t segment_count = 0;

  // Process with capacity=0 should consume nothing and produce nothing.
  IREE_EXPECT_OK(iree_tokenizer_segmenter_state_process(
      state.get(), input, iree_tokenizer_segment_output_empty(), &consumed,
      &segment_count));

  EXPECT_EQ(consumed, 0u) << "Zero capacity should consume nothing";
  EXPECT_EQ(segment_count, 0u) << "Zero capacity should produce no segments";
}

}  // namespace iree::tokenizer::testing
