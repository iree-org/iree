// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"
#include "iree/testing/benchmark.h"

// WARNING: these benchmarks likely test the CPU cache more than they test the
// implementation as the bitmaps are effectively guaranteed hot. This gives us
// a machine-dependent upper bound on throughput factoring out cache effects
// but does not directly indicate the time taken when the bitmap is not hot
// (which will be dominated by memory access).

//===----------------------------------------------------------------------===//
// Single bit operations
//===----------------------------------------------------------------------===//

IREE_BENCHMARK_FN(BM_BitmapTest_64) {
  uint64_t words[1] = {0xAAAAAAAAAAAAAAAAull};
  iree_bitmap_t bitmap = {64, words};
  iree_host_size_t index = 0;
  while (iree_benchmark_keep_running(benchmark_state, 64)) {
    for (int i = 0; i < 64; ++i) {
      bool result = iree_bitmap_test(bitmap, index);
      iree_optimization_barrier(result);
      index = (index + 7) & 63;  // Prime number to avoid patterns.
    }
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapTest_64);

IREE_BENCHMARK_FN(BM_BitmapTest_256) {
  uint64_t words[4] = {0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
                       0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull};
  iree_bitmap_t bitmap = {256, words};
  iree_host_size_t index = 0;
  while (iree_benchmark_keep_running(benchmark_state, 256)) {
    for (int i = 0; i < 256; ++i) {
      bool result = iree_bitmap_test(bitmap, index);
      iree_optimization_barrier(result);
      index = (index + 31) & 255;  // Prime number to avoid patterns.
    }
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapTest_256);

IREE_BENCHMARK_FN(BM_BitmapSet_64) {
  uint64_t words[1] = {0};
  iree_bitmap_t bitmap = {64, words};
  iree_host_size_t index = 0;
  while (iree_benchmark_keep_running(benchmark_state, 64)) {
    words[0] = 0;  // Reset for each iteration.
    for (int i = 0; i < 64; ++i) {
      iree_bitmap_set(bitmap, index);
      index = (index + 7) & 63;
    }
    iree_optimization_barrier(words[0]);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapSet_64);

IREE_BENCHMARK_FN(BM_BitmapSet_256) {
  uint64_t words[4] = {0};
  iree_bitmap_t bitmap = {256, words};
  iree_host_size_t index = 0;
  while (iree_benchmark_keep_running(benchmark_state, 256)) {
    memset(words, 0, sizeof(words));  // Reset for each iteration.
    for (int i = 0; i < 256; ++i) {
      iree_bitmap_set(bitmap, index);
      index = (index + 31) & 255;
    }
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapSet_256);

IREE_BENCHMARK_FN(BM_BitmapReset_64) {
  uint64_t words[1] = {UINT64_MAX};
  iree_bitmap_t bitmap = {64, words};
  iree_host_size_t index = 0;
  while (iree_benchmark_keep_running(benchmark_state, 64)) {
    words[0] = UINT64_MAX;  // Reset for each iteration.
    for (int i = 0; i < 64; ++i) {
      iree_bitmap_reset(bitmap, index);
      index = (index + 7) & 63;
    }
    iree_optimization_barrier(words[0]);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapReset_64);

IREE_BENCHMARK_FN(BM_BitmapReset_256) {
  uint64_t words[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
  iree_bitmap_t bitmap = {256, words};
  iree_host_size_t index = 0;
  while (iree_benchmark_keep_running(benchmark_state, 256)) {
    memset(words, 0xFF, sizeof(words));  // Reset for each iteration.
    for (int i = 0; i < 256; ++i) {
      iree_bitmap_reset(bitmap, index);
      index = (index + 31) & 255;
    }
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapReset_256);

//===----------------------------------------------------------------------===//
// Span operations
//===----------------------------------------------------------------------===//

IREE_BENCHMARK_FN(BM_BitmapSetSpan_Small) {
  uint64_t words[4] = {0};
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 32)) {
    memset(words, 0, sizeof(words));
    // Set 32 spans of 8 bits each.
    for (int i = 0; i < 32; ++i) {
      iree_bitmap_set_span(bitmap, i * 8, 8);
    }
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapSetSpan_Small);

IREE_BENCHMARK_FN(BM_BitmapSetSpan_Large) {
  uint64_t words[16] = {0};
  iree_bitmap_t bitmap = {1024, words};
  while (iree_benchmark_keep_running(benchmark_state, 16)) {
    memset(words, 0, sizeof(words));
    // Set 16 spans of 64 bits each.
    for (int i = 0; i < 16; ++i) {
      iree_bitmap_set_span(bitmap, i * 64, 64);
    }
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapSetSpan_Large);

IREE_BENCHMARK_FN(BM_BitmapSetSpan_Unaligned) {
  uint64_t words[4] = {0};
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 16)) {
    memset(words, 0, sizeof(words));
    // Set spans that cross word boundaries.
    for (int i = 0; i < 16; ++i) {
      iree_bitmap_set_span(bitmap, i * 15 + 7, 13);
    }
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapSetSpan_Unaligned);

IREE_BENCHMARK_FN(BM_BitmapResetSpan_Small) {
  uint64_t words[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 32)) {
    memset(words, 0xFF, sizeof(words));
    // Reset 32 spans of 8 bits each.
    for (int i = 0; i < 32; ++i) {
      iree_bitmap_reset_span(bitmap, i * 8, 8);
    }
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapResetSpan_Small);

//===----------------------------------------------------------------------===//
// Full bitmap operations
//===----------------------------------------------------------------------===//

IREE_BENCHMARK_FN(BM_BitmapSetAll_64) {
  uint64_t words[1] = {0};
  iree_bitmap_t bitmap = {64, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    words[0] = 0;
    iree_bitmap_set_all(bitmap);
    iree_optimization_barrier(words[0]);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapSetAll_64);

IREE_BENCHMARK_FN(BM_BitmapSetAll_1024) {
  uint64_t words[16] = {0};
  iree_bitmap_t bitmap = {1024, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    memset(words, 0, sizeof(words));
    iree_bitmap_set_all(bitmap);
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapSetAll_1024);

IREE_BENCHMARK_FN(BM_BitmapResetAll_64) {
  uint64_t words[1] = {UINT64_MAX};
  iree_bitmap_t bitmap = {64, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    words[0] = UINT64_MAX;
    iree_bitmap_reset_all(bitmap);
    iree_optimization_barrier(words[0]);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapResetAll_64);

IREE_BENCHMARK_FN(BM_BitmapResetAll_1024) {
  uint64_t words[16] = {0};
  iree_bitmap_t bitmap = {1024, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    memset(words, 0xFF, sizeof(words));
    iree_bitmap_reset_all(bitmap);
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapResetAll_1024);

//===----------------------------------------------------------------------===//
// Query operations
//===----------------------------------------------------------------------===//

IREE_BENCHMARK_FN(BM_BitmapAnySet_64) {
  uint64_t words[1] = {0x0000000080000000ull};  // One bit set in middle.
  iree_bitmap_t bitmap = {64, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    bool result = iree_bitmap_any_set(bitmap);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapAnySet_64);

IREE_BENCHMARK_FN(BM_BitmapAnySet_1024_Sparse) {
  uint64_t words[16] = {0};
  words[8] = 0x0000000080000000ull;  // One bit set in middle word.
  iree_bitmap_t bitmap = {1024, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    bool result = iree_bitmap_any_set(bitmap);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapAnySet_1024_Sparse);

IREE_BENCHMARK_FN(BM_BitmapNoneSet_64) {
  uint64_t words[1] = {0};
  iree_bitmap_t bitmap = {64, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    bool result = iree_bitmap_none_set(bitmap);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapNoneSet_64);

IREE_BENCHMARK_FN(BM_BitmapNoneSet_1024) {
  uint64_t words[16] = {0};
  iree_bitmap_t bitmap = {1024, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    bool result = iree_bitmap_none_set(bitmap);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapNoneSet_1024);

IREE_BENCHMARK_FN(BM_BitmapCount_64) {
  uint64_t words[1] = {0xAAAAAAAAAAAAAAAAull};  // 32 bits set.
  iree_bitmap_t bitmap = {64, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_host_size_t result = iree_bitmap_count(bitmap);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapCount_64);

IREE_BENCHMARK_FN(BM_BitmapCount_1024) {
  uint64_t words[16];
  memset(words, 0xAA, sizeof(words));  // 50% bits set.
  iree_bitmap_t bitmap = {1024, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_host_size_t result = iree_bitmap_count(bitmap);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapCount_1024);

//===----------------------------------------------------------------------===//
// Find operations
//===----------------------------------------------------------------------===//

IREE_BENCHMARK_FN(BM_BitmapFindFirstSet_Early) {
  uint64_t words[4] = {0x0000000000000001ull, 0, 0, 0};  // First bit set.
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_host_size_t result = iree_bitmap_find_first_set(bitmap, 0);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapFindFirstSet_Early);

IREE_BENCHMARK_FN(BM_BitmapFindFirstSet_Late) {
  uint64_t words[4] = {0, 0, 0, 0x8000000000000000ull};  // Last bit set.
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_host_size_t result = iree_bitmap_find_first_set(bitmap, 0);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapFindFirstSet_Late);

IREE_BENCHMARK_FN(BM_BitmapFindFirstUnset_Early) {
  uint64_t words[4] = {UINT64_MAX - 1, UINT64_MAX, UINT64_MAX, UINT64_MAX};
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_host_size_t result = iree_bitmap_find_first_unset(bitmap, 0);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapFindFirstUnset_Early);

IREE_BENCHMARK_FN(BM_BitmapFindFirstUnset_Late) {
  uint64_t words[4] = {UINT64_MAX, UINT64_MAX, UINT64_MAX,
                       0x7FFFFFFFFFFFFFFFull};
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_host_size_t result = iree_bitmap_find_first_unset(bitmap, 0);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapFindFirstUnset_Late);

// Pattern with small gaps.
IREE_BENCHMARK_FN(BM_BitmapFindFirstUnsetSpan_Small) {
  uint64_t words[4] = {0xF0F0F0F0F0F0F0F0ull, 0xF0F0F0F0F0F0F0F0ull,
                       0xF0F0F0F0F0F0F0F0ull, 0xF0F0F0F0F0F0F0F0ull};
  iree_bitmap_t bitmap = {256, words};
  while (iree_benchmark_keep_running(benchmark_state, 32)) {
    iree_host_size_t offset = 0;
    for (int i = 0; i < 32; ++i) {
      iree_host_size_t result =
          iree_bitmap_find_first_unset_span(bitmap, offset, 4);
      iree_optimization_barrier(result);
      offset = (result + 4) % 256;
    }
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapFindFirstUnsetSpan_Small);

// Mostly full bitmap with one large gap.
IREE_BENCHMARK_FN(BM_BitmapFindFirstUnsetSpan_Large) {
  uint64_t words[16];
  memset(words, 0xFF, sizeof(words));
  words[8] = 0;  // 64-bit gap in the middle.
  words[9] = 0;
  iree_bitmap_t bitmap = {1024, words};
  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    iree_host_size_t result = iree_bitmap_find_first_unset_span(bitmap, 0, 64);
    iree_optimization_barrier(result);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapFindFirstUnsetSpan_Large);

//===----------------------------------------------------------------------===//
// Allocation pattern simulation
//===----------------------------------------------------------------------===//

// Simulates a typical allocation pattern: find free space, set bits,
// occasionally reset bits (deallocation).
IREE_BENCHMARK_FN(BM_BitmapAllocationPattern) {
  uint64_t words[16] = {0};
  iree_bitmap_t bitmap = {1024, words};
  iree_host_size_t alloc_sizes[8] = {8, 16, 32, 8, 64, 16, 8, 128};
  iree_host_size_t alloc_positions[8] = {0};
  int alloc_index = 0;

  while (iree_benchmark_keep_running(benchmark_state, 24)) {
    // Reset bitmap.
    memset(words, 0, sizeof(words));

    // Allocate 16 times.
    for (int i = 0; i < 16; ++i) {
      iree_host_size_t size = alloc_sizes[i & 7];
      iree_host_size_t pos = iree_bitmap_find_first_unset_span(bitmap, 0, size);
      if (pos < bitmap.bit_count) {
        iree_bitmap_set_span(bitmap, pos, size);
        alloc_positions[alloc_index] = pos;
        alloc_index = (alloc_index + 1) & 7;
      }
    }

    // Deallocate 8 times.
    for (int i = 0; i < 8; ++i) {
      iree_host_size_t pos = alloc_positions[i];
      iree_host_size_t size = alloc_sizes[i];
      if (pos < bitmap.bit_count) {
        iree_bitmap_reset_span(bitmap, pos, size);
      }
    }

    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapAllocationPattern);

//===----------------------------------------------------------------------===//
// Large bitmap stress tests (10,000+ bits)
//===----------------------------------------------------------------------===//

// Tests with a large bitmap simulating a large memory allocator with various
// allocation sizes (8-1024 bits) and 50% fragmentation.
IREE_BENCHMARK_FN(BM_BitmapLargeAllocationPattern) {
  uint64_t words[256] = {0};  // 256 * 64 = 16384 bits.
  iree_bitmap_t bitmap = {16384, words};
  iree_host_size_t alloc_sizes[16] = {8,   16, 32, 64, 128, 256, 512, 1024,
                                      128, 64, 32, 16, 8,   256, 512, 128};
  iree_host_size_t alloc_positions[16] = {0};
  int alloc_index = 0;

  while (iree_benchmark_keep_running(benchmark_state, 48)) {
    // Reset bitmap.
    memset(words, 0, sizeof(words));

    // Allocate 32 times.
    for (int i = 0; i < 32; ++i) {
      iree_host_size_t size = alloc_sizes[i & 15];
      iree_host_size_t pos = iree_bitmap_find_first_unset_span(bitmap, 0, size);
      if (pos < bitmap.bit_count) {
        iree_bitmap_set_span(bitmap, pos, size);
        alloc_positions[alloc_index] = pos;
        alloc_index = (alloc_index + 1) & 15;
      }
    }

    // Deallocate 16 times (50% fragmentation).
    for (int i = 0; i < 16; ++i) {
      iree_host_size_t pos = alloc_positions[i];
      iree_host_size_t size = alloc_sizes[i];
      if (pos < bitmap.bit_count) {
        iree_bitmap_reset_span(bitmap, pos, size);
      }
    }

    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapLargeAllocationPattern);

// Creates extreme fragmentation (checkerboard pattern) in a 16K bitmap, then
// searches for contiguous spans of various sizes.
IREE_BENCHMARK_FN(BM_BitmapFragmentationStress) {
  uint64_t words[256] = {0};  // 16384 bits.
  iree_bitmap_t bitmap = {16384, words};

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    // Create checkerboard pattern.
    for (int i = 0; i < 16384; i += 2) {
      iree_bitmap_set(bitmap, i);
    }

    // Now try to find contiguous spans in this fragmented bitmap.
    iree_host_size_t found_count = 0;
    for (int size = 2; size <= 64; size *= 2) {
      iree_host_size_t pos = 0;
      while (pos < bitmap.bit_count) {
        pos = iree_bitmap_find_first_unset_span(bitmap, pos, size);
        if (pos < bitmap.bit_count) {
          found_count++;
          pos += size;
        } else {
          break;
        }
      }
    }

    iree_optimization_barrier(found_count);
    memset(words, 0, sizeof(words));
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapFragmentationStress);

// Scans through a large bit bitmap with sparse clusters, testing both
// find_first_set and find_first_unset_span operations.
IREE_BENCHMARK_FN(BM_BitmapScanLarge_10K) {
  uint64_t words[157] = {0};  // 157 * 64 = 10048 bits.
  iree_bitmap_t bitmap = {10000, words};

  // Set up a sparse pattern: clusters of set bits.
  for (int cluster = 0; cluster < 100; ++cluster) {
    iree_host_size_t start = cluster * 100;
    iree_bitmap_set_span(bitmap, start, 10);
  }

  while (iree_benchmark_keep_running(benchmark_state, 200)) {
    iree_host_size_t total_set = 0;
    iree_host_size_t pos = 0;

    // Scan for all set bits.
    for (int i = 0; i < 100; ++i) {
      pos = iree_bitmap_find_first_set(bitmap, pos);
      if (pos < bitmap.bit_count) {
        total_set++;
        pos++;
      } else {
        break;
      }
    }

    // Scan for unset spans.
    pos = 0;
    for (int i = 0; i < 100; ++i) {
      pos = iree_bitmap_find_first_unset_span(bitmap, pos, 50);
      if (pos < bitmap.bit_count) {
        pos += 50;
      } else {
        break;
      }
    }

    iree_optimization_barrier(total_set);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapScanLarge_10K);

// Tests random access patterns on a large bitmap using prime numbers for good
// distribution, mixing set/test/reset operations.
IREE_BENCHMARK_FN(BM_BitmapRandomAccess_10K) {
  uint64_t words[157] = {0};  // 10048 bits.
  iree_bitmap_t bitmap = {10000, words};

  // Pre-initialize with some data.
  for (int i = 0; i < 10000; i += 7) {
    iree_bitmap_set(bitmap, i);
  }

  while (iree_benchmark_keep_running(benchmark_state, 1000)) {
    // Pseudo-random access pattern using prime numbers.
    iree_host_size_t index = 0;
    for (int i = 0; i < 1000; ++i) {
      index = (index + 997) % 10000;  // Large prime for good distribution.

      // Mix of operations.
      if (i & 1) {
        iree_bitmap_set(bitmap, index);
      } else {
        bool is_set = iree_bitmap_test(bitmap, index);
        if (is_set) {
          iree_bitmap_reset(bitmap, index);
        }
      }
    }
    iree_optimization_barrier(words);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapRandomAccess_10K);

// Tests bulk operations (set_all, reset_all, large spans) on large bitmaps.
IREE_BENCHMARK_FN(BM_BitmapBulkOps_16K) {
  uint64_t words[256] = {0};  // 16384 bits.
  iree_bitmap_t bitmap = {16384, words};

  while (iree_benchmark_keep_running(benchmark_state, 6)) {
    // Set all bits.
    iree_bitmap_set_all(bitmap);

    // Reset large spans.
    iree_bitmap_reset_span(bitmap, 1000, 2000);
    iree_bitmap_reset_span(bitmap, 5000, 3000);

    // Check if any bits are set (should be true).
    bool any = iree_bitmap_any_set(bitmap);
    iree_optimization_barrier(any);

    // Reset all.
    iree_bitmap_reset_all(bitmap);

    // Set large spans.
    iree_bitmap_set_span(bitmap, 2000, 4000);
    iree_bitmap_set_span(bitmap, 8000, 4000);

    // Check if none set (should be false).
    bool none = iree_bitmap_none_set(bitmap);
    iree_optimization_barrier(none);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapBulkOps_16K);

// Worst-case scenario: almost full 10K bitmap with only small 2-bit gaps,
// testing the performance of finding these rare gaps.
IREE_BENCHMARK_FN(BM_BitmapWorstCaseFindSpan_10K) {
  uint64_t words[157] = {0};  // 10048 bits.
  iree_bitmap_t bitmap = {10000, words};

  // Fill most of the bitmap, leaving small gaps.
  iree_bitmap_set_all(bitmap);
  for (int i = 0; i < 10000; i += 100) {
    iree_bitmap_reset_span(bitmap, i, 2);  // Small 2-bit gaps.
  }

  while (iree_benchmark_keep_running(benchmark_state, 100)) {
    iree_host_size_t found = 0;
    // Try to find 100 small gaps.
    for (int i = 0; i < 100; ++i) {
      iree_host_size_t pos = iree_bitmap_find_first_unset_span(bitmap, 0, 2);
      if (pos < bitmap.bit_count) {
        found++;
        // Fill the gap to make next search harder.
        iree_bitmap_set_span(bitmap, pos, 2);
      }
    }
    // Restore gaps for next iteration.
    for (int i = 0; i < 10000; i += 100) {
      iree_bitmap_reset_span(bitmap, i, 2);
    }
    iree_optimization_barrier(found);
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapWorstCaseFindSpan_10K);

// Simulates memory coalescing patterns: allocate small blocks, free some, then
// try to allocate larger blocks (testing fragmentation and defragmentation
// scenarios).
IREE_BENCHMARK_FN(BM_BitmapCoalescingPattern_10K) {
  uint64_t words[157] = {0};  // 10048 bits.
  iree_bitmap_t bitmap = {10000, words};

  while (iree_benchmark_keep_running(benchmark_state, 1)) {
    // Phase 1: Allocate many small blocks.
    for (int i = 0; i < 10000; i += 10) {
      iree_bitmap_set_span(bitmap, i, 8);  // Leave 2-bit gaps.
    }

    // Phase 2: Free every other block.
    for (int i = 0; i < 10000; i += 20) {
      iree_bitmap_reset_span(bitmap, i, 8);
    }

    // Phase 3: Try to allocate larger blocks (coalescing).
    iree_host_size_t large_allocs = 0;
    iree_host_size_t pos = 0;
    while (pos < 9900) {
      pos = iree_bitmap_find_first_unset_span(bitmap, pos, 16);
      if (pos < bitmap.bit_count) {
        iree_bitmap_set_span(bitmap, pos, 16);
        large_allocs++;
        pos += 16;
      } else {
        break;
      }
    }

    iree_optimization_barrier(large_allocs);
    memset(words, 0, sizeof(words));
  }
  return iree_ok_status();
}
IREE_BENCHMARK_REGISTER(BM_BitmapCoalescingPattern_10K);
