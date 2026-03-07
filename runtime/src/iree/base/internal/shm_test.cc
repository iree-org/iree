// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/base/internal/shm.h"

#include <cstring>

#if !defined(IREE_PLATFORM_WINDOWS) && !defined(IREE_PLATFORM_ANDROID)
#include <sys/mman.h>  // shm_unlink for test cleanup
#endif

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class ShmTest : public ::testing::Test {
 protected:
  iree_shm_options_t options_ = iree_shm_options_default();
};

TEST_F(ShmTest, RequiredSizeZeroReturnsOnePage) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_GT(page, 0);
  // Must be a power of two (all known page sizes are).
  EXPECT_EQ(page & (page - 1), 0);
}

TEST_F(ShmTest, RequiredSizeOneReturnsOnePage) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_EQ(iree_shm_required_size(1), page);
}

TEST_F(ShmTest, RequiredSizeExactPageUnchanged) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_EQ(iree_shm_required_size(page), page);
}

TEST_F(ShmTest, RequiredSizeRoundsUpToNextPage) {
  iree_host_size_t page = iree_shm_required_size(0);
  EXPECT_EQ(iree_shm_required_size(page + 1), page * 2);
}

TEST_F(ShmTest, InvalidHandleSentinel) {
  iree_shm_handle_t handle = IREE_SHM_HANDLE_INVALID;
  EXPECT_FALSE(iree_shm_handle_is_valid(handle));
}

TEST_F(ShmTest, HandleCloseOnInvalid) {
  iree_shm_handle_t handle = IREE_SHM_HANDLE_INVALID;
  // Must not crash.
  iree_shm_handle_close(&handle);
  EXPECT_FALSE(iree_shm_handle_is_valid(handle));
}

TEST_F(ShmTest, HandleCloseOnNull) {
  // Must not crash.
  iree_shm_handle_close(NULL);
}

TEST_F(ShmTest, CloseZeroedMapping) {
  iree_shm_mapping_t mapping;
  memset(&mapping, 0, sizeof(mapping));
  mapping.handle = IREE_SHM_HANDLE_INVALID;
  // Must not crash.
  iree_shm_close(&mapping);
}

TEST_F(ShmTest, CloseNull) {
  // Must not crash.
  iree_shm_close(NULL);
}

TEST_F(ShmTest, CreateZeroSizeFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_create(options_, 0, &mapping));
}

TEST_F(ShmTest, CreateAnonymous) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &mapping));

  EXPECT_NE(mapping.base, nullptr);
  EXPECT_GE(mapping.size, (iree_host_size_t)4096);
  EXPECT_TRUE(iree_shm_handle_is_valid(mapping.handle));

  // Verify read/write works.
  memset(mapping.base, 0xAB, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAB);
  EXPECT_EQ(((uint8_t*)mapping.base)[mapping.size - 1], 0xAB);

  iree_shm_close(&mapping);

  // After close, mapping is zeroed.
  EXPECT_EQ(mapping.base, nullptr);
  EXPECT_EQ(mapping.size, 0);
  EXPECT_FALSE(iree_shm_handle_is_valid(mapping.handle));
}

TEST_F(ShmTest, CreateAnonymousSubPageSize) {
  // Requesting less than a page should still get a full page.
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 1, &mapping));

  iree_host_size_t page_size = iree_shm_required_size(0);
  EXPECT_GE(mapping.size, page_size);
  EXPECT_NE(mapping.base, nullptr);

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, HandleDup) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &mapping));

  iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(mapping.handle, &dup_handle));
  EXPECT_TRUE(iree_shm_handle_is_valid(dup_handle));

  // Duplicated handle is independent; closing the original mapping doesn't
  // invalidate the dup.
  iree_shm_close(&mapping);

  // The dup handle is still valid and can be used to open a mapping.
  iree_shm_handle_close(&dup_handle);
  EXPECT_FALSE(iree_shm_handle_is_valid(dup_handle));
}

TEST_F(ShmTest, HandleDupInvalidFails) {
  iree_shm_handle_t dup_handle;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_handle_dup(IREE_SHM_HANDLE_INVALID, &dup_handle));
}

TEST_F(ShmTest, OpenHandle) {
  // Create a region and write a pattern.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &creator));
  memset(creator.base, 0xCD, creator.size);

  // Duplicate the handle (simulates passing to another process).
  iree_shm_handle_t shared_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &shared_handle));

  // Open a second mapping from the duplicated handle.
  iree_shm_mapping_t opener;
  IREE_ASSERT_OK(
      iree_shm_open_handle(shared_handle, options_, creator.size, &opener));
  EXPECT_NE(opener.base, nullptr);
  EXPECT_EQ(opener.size, creator.size);

  // Both mappings must see the same data.
  EXPECT_EQ(memcmp(creator.base, opener.base, creator.size), 0);

  // Write through the opener and verify the creator sees it.
  memset(opener.base, 0xEF, opener.size);
  EXPECT_EQ(((uint8_t*)creator.base)[0], 0xEF);

  iree_shm_handle_close(&shared_handle);
  iree_shm_close(&opener);
  iree_shm_close(&creator);
}

TEST_F(ShmTest, OpenHandleInvalidFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_open_handle(IREE_SHM_HANDLE_INVALID, options_, 4096, &mapping));
}

TEST_F(ShmTest, OpenHandleZeroSizeFails) {
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &creator));

  iree_shm_mapping_t opener;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_open_handle(creator.handle, options_, 0, &opener));

  iree_shm_close(&creator);
}

TEST_F(ShmTest, OpenHandleSizeTooLargeFails) {
  // Create a one-page region, then try to open it with a much larger size.
  // POSIX: our fstat check catches this before mmap (avoids SIGBUS).
  // Windows: MapViewOfFile fails with an error for the same case.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &creator));

  iree_shm_handle_t shared_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &shared_handle));

  iree_shm_mapping_t opener;
  iree_status_t status = iree_shm_open_handle(shared_handle, options_,
                                              creator.size * 1024, &opener);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);

  iree_shm_handle_close(&shared_handle);
  iree_shm_close(&creator);
}

// Named shared memory tests.
//
// Android's bionic libc lacks shm_open/shm_unlink, so the named SHM API
// returns UNAVAILABLE there. These tests are only meaningful on platforms
// that support named shared memory.
#if !defined(IREE_PLATFORM_ANDROID)

// POSIX shm_open names must start with '/'. On Windows the "Local\" prefix is
// added automatically. These tests use POSIX-style names; the implementation
// handles the Windows prefix.
#if defined(IREE_PLATFORM_WINDOWS)
#define TEST_SHM_NAME(suffix) "iree_shm_test_" suffix
#else
#define TEST_SHM_NAME(suffix) "/iree_shm_test_" suffix
#endif  // IREE_PLATFORM_WINDOWS

class ShmNamedTest : public ShmTest {
 protected:
  void TearDown() override {
    // Clean up any named regions left by failed tests.
    for (const auto& name : names_to_cleanup_) {
#if !defined(IREE_PLATFORM_WINDOWS)
      shm_unlink(name.c_str());
#endif
    }
  }
  void TrackName(const char* name) { names_to_cleanup_.push_back(name); }

 private:
  std::vector<std::string> names_to_cleanup_;
};

TEST_F(ShmNamedTest, CreateAndOpenNamed) {
  const char* name = TEST_SHM_NAME("create_open");
  TrackName(name);

  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create_named(iree_make_cstring_view(name), options_,
                                       4096, &creator));
  EXPECT_NE(creator.base, nullptr);
  EXPECT_GE(creator.size, (iree_host_size_t)4096);

  // Write a pattern.
  memset(creator.base, 0x42, creator.size);

  // Open the same region by name.
  iree_shm_mapping_t opener;
  IREE_ASSERT_OK(iree_shm_open_named(iree_make_cstring_view(name), options_,
                                     creator.size, &opener));
  EXPECT_NE(opener.base, nullptr);

  // Both mappings see the same data.
  EXPECT_EQ(memcmp(creator.base, opener.base, creator.size), 0);

  iree_shm_close(&opener);
  iree_shm_close(&creator);

  // Clean up the name on POSIX.
#if !defined(IREE_PLATFORM_WINDOWS)
  shm_unlink(name);
#endif
}

TEST_F(ShmNamedTest, CreateNamedDuplicateFails) {
  const char* name = TEST_SHM_NAME("dup_fail");
  TrackName(name);

  iree_shm_mapping_t first;
  IREE_ASSERT_OK(iree_shm_create_named(iree_make_cstring_view(name), options_,
                                       4096, &first));

  // Creating a second region with the same name must fail.
  iree_shm_mapping_t second;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ALREADY_EXISTS,
                        iree_shm_create_named(iree_make_cstring_view(name),
                                              options_, 4096, &second));

  iree_shm_close(&first);

#if !defined(IREE_PLATFORM_WINDOWS)
  shm_unlink(name);
#endif
}

TEST_F(ShmNamedTest, OpenNamedNonexistentFails) {
  const char* name = TEST_SHM_NAME("nonexistent");
  iree_shm_mapping_t mapping;
  iree_status_t status = iree_shm_open_named(iree_make_cstring_view(name),
                                             options_, 4096, &mapping);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);
}

TEST_F(ShmNamedTest, CreateNamedZeroSizeFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_create_named(iree_make_cstring_view(TEST_SHM_NAME("zero")),
                            options_, 0, &mapping));
}

TEST_F(ShmNamedTest, OpenNamedZeroSizeFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_shm_open_named(iree_make_cstring_view(TEST_SHM_NAME("zero")),
                          options_, 0, &mapping));
}

TEST_F(ShmNamedTest, CreateNamedEmptyNameFails) {
  iree_shm_mapping_t mapping;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_create_named(iree_make_string_view("", 0),
                                              options_, 4096, &mapping));
}

#endif  // !IREE_PLATFORM_ANDROID

TEST_F(ShmTest, QuerySealsInitiallyNone) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &mapping));

  iree_shm_seal_flags_t seals = iree_shm_query_seals(&mapping);
#if defined(IREE_PLATFORM_LINUX)
  // Anonymous (memfd) regions get SHRINK|GROW seals during creation.
  EXPECT_TRUE(seals & IREE_SHM_SEAL_SHRINK);
  EXPECT_TRUE(seals & IREE_SHM_SEAL_GROW);
  EXPECT_FALSE(seals & IREE_SHM_SEAL_WRITE);
  EXPECT_FALSE(seals & IREE_SHM_SEAL_SEAL);
#else
  EXPECT_EQ(seals, IREE_SHM_SEAL_NONE);
#endif

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, QuerySealsNullMapping) {
  EXPECT_EQ(iree_shm_query_seals(NULL), IREE_SHM_SEAL_NONE);
}

TEST_F(ShmTest, QuerySealsUnmappedRegion) {
  iree_shm_mapping_t mapping;
  memset(&mapping, 0, sizeof(mapping));
  mapping.handle = IREE_SHM_HANDLE_INVALID;
  EXPECT_EQ(iree_shm_query_seals(&mapping), IREE_SHM_SEAL_NONE);
}

TEST_F(ShmTest, SealNoneFlagsSucceeds) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &mapping));
  IREE_EXPECT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_NONE));
  iree_shm_close(&mapping);
}

TEST_F(ShmTest, SealNullMappingFails) {
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_shm_seal(NULL, IREE_SHM_SEAL_WRITE));
}

TEST_F(ShmTest, SealWrite) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &mapping));

  // Write data before sealing.
  memset(mapping.base, 0xAA, mapping.size);
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAA);

#if defined(IREE_PLATFORM_LINUX)
  // Linux supports sealing on anonymous (memfd) regions.
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  iree_shm_seal_flags_t seals = iree_shm_query_seals(&mapping);
  EXPECT_TRUE(seals & IREE_SHM_SEAL_WRITE);
  // Data is still readable after sealing.
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAA);
#elif defined(IREE_PLATFORM_APPLE)
  // macOS does not support sealing.
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
#elif defined(IREE_PLATFORM_WINDOWS)
  // Windows uses VirtualProtect for write sealing.
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  iree_shm_seal_flags_t seals = iree_shm_query_seals(&mapping);
  EXPECT_TRUE(seals & IREE_SHM_SEAL_WRITE);
  // Data is still readable after sealing.
  EXPECT_EQ(((uint8_t*)mapping.base)[0], 0xAA);
#endif

  iree_shm_close(&mapping);
}

TEST_F(ShmTest, SealWriteIdempotent) {
  // Sealing the same flag twice must succeed (no-op on second call).
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &mapping));
  memset(mapping.base, 0xBB, mapping.size);

#if defined(IREE_PLATFORM_LINUX) || defined(IREE_PLATFORM_WINDOWS)
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE));
  EXPECT_TRUE(iree_shm_query_seals(&mapping) & IREE_SHM_SEAL_WRITE);
#endif

  iree_shm_close(&mapping);
}

#if defined(IREE_PLATFORM_LINUX)
TEST_F(ShmTest, SealWriteFailsWithSecondWritableMapping) {
  // When a second writable mapping exists, F_SEAL_WRITE fails with EBUSY.
  // Verify the rollback restores the original mapping so it's still usable.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &creator));
  memset(creator.base, 0xDD, creator.size);

  // Open a second writable mapping of the same fd.
  iree_shm_handle_t dup_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &dup_handle));
  iree_shm_mapping_t second;
  IREE_ASSERT_OK(
      iree_shm_open_handle(dup_handle, options_, creator.size, &second));

  // Sealing the creator must fail because the second mapping is writable.
  iree_status_t status = iree_shm_seal(&creator, IREE_SHM_SEAL_WRITE);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);

  // The creator mapping must still be valid and readable after rollback.
  EXPECT_NE(creator.base, nullptr);
  EXPECT_EQ(((uint8_t*)creator.base)[0], 0xDD);
  EXPECT_TRUE(iree_shm_handle_is_valid(creator.handle));

  // The write seal must NOT have been applied.
  EXPECT_FALSE(iree_shm_query_seals(&creator) & IREE_SHM_SEAL_WRITE);

  iree_shm_handle_close(&dup_handle);
  iree_shm_close(&second);
  iree_shm_close(&creator);
}

TEST_F(ShmTest, SealSealPreventsNewSeals) {
  iree_shm_mapping_t mapping;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &mapping));

  // Apply SEAL_SEAL — no more seals can be added after this.
  IREE_ASSERT_OK(iree_shm_seal(&mapping, IREE_SHM_SEAL_SEAL));
  EXPECT_TRUE(iree_shm_query_seals(&mapping) & IREE_SHM_SEAL_SEAL);

  // Attempting to add SEAL_WRITE after SEAL_SEAL must fail.
  iree_status_t status = iree_shm_seal(&mapping, IREE_SHM_SEAL_WRITE);
  EXPECT_FALSE(iree_status_is_ok(status));
  iree_status_ignore(status);

  iree_shm_close(&mapping);
}
#endif  // IREE_PLATFORM_LINUX

TEST_F(ShmTest, SealWriteVisibleToSecondMapping) {
  // Seal via the creator, verify the opener sees the sealed data.
  iree_shm_mapping_t creator;
  IREE_ASSERT_OK(iree_shm_create(options_, 4096, &creator));
  memset(creator.base, 0xCC, creator.size);

  iree_shm_handle_t shared_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(creator.handle, &shared_handle));

  iree_shm_mapping_t opener;
  IREE_ASSERT_OK(
      iree_shm_open_handle(shared_handle, options_, creator.size, &opener));

#if defined(IREE_PLATFORM_LINUX)
  // Both mappings must be made read-only for F_SEAL_WRITE to succeed.
  // Seal via the opener mapping to verify it works from either side.
  // The creator's mapping is still writable, so we need to seal via the
  // creator (which will mprotect the creator's mapping, but the kernel also
  // requires no other writable VMAs — meaning we need to close the opener's
  // writable mapping first, then seal, then reopen as read-only).
  //
  // For simplicity, close the opener, seal via creator, then reopen.
  iree_shm_close(&opener);
  IREE_ASSERT_OK(iree_shm_seal(&creator, IREE_SHM_SEAL_WRITE));

  // Reopen — the new mapping inherits the seal; the kernel won't allow
  // PROT_WRITE since F_SEAL_WRITE is set.
  IREE_ASSERT_OK(
      iree_shm_open_handle(shared_handle, options_, creator.size, &opener));
  // The sealed data is readable through both mappings.
  EXPECT_EQ(((uint8_t*)creator.base)[0], 0xCC);
  EXPECT_EQ(((uint8_t*)opener.base)[0], 0xCC);
  // Both mappings report the seal.
  EXPECT_TRUE(iree_shm_query_seals(&creator) & IREE_SHM_SEAL_WRITE);
  EXPECT_TRUE(iree_shm_query_seals(&opener) & IREE_SHM_SEAL_WRITE);
#elif defined(IREE_PLATFORM_WINDOWS)
  IREE_ASSERT_OK(iree_shm_seal(&creator, IREE_SHM_SEAL_WRITE));
  EXPECT_TRUE(iree_shm_query_seals(&creator) & IREE_SHM_SEAL_WRITE);
  // On Windows, VirtualProtect is per-view — the opener's view is unaffected.
  EXPECT_EQ(((uint8_t*)opener.base)[0], 0xCC);
#endif

  iree_shm_handle_close(&shared_handle);
  iree_shm_close(&opener);
  iree_shm_close(&creator);
}

TEST_F(ShmTest, WriteReadCoherence) {
  // Write via one mapping, read via another opened from a dup'd handle.
  iree_shm_mapping_t writer;
  IREE_ASSERT_OK(iree_shm_create(options_, 8192, &writer));

  iree_shm_handle_t reader_handle = IREE_SHM_HANDLE_INVALID;
  IREE_ASSERT_OK(iree_shm_handle_dup(writer.handle, &reader_handle));

  iree_shm_mapping_t reader;
  IREE_ASSERT_OK(
      iree_shm_open_handle(reader_handle, options_, writer.size, &reader));

  // Write a structured pattern.
  for (iree_host_size_t i = 0; i < writer.size; ++i) {
    ((uint8_t*)writer.base)[i] = (uint8_t)(i & 0xFF);
  }

  // Verify the reader sees it.
  for (iree_host_size_t i = 0; i < reader.size; ++i) {
    ASSERT_EQ(((uint8_t*)reader.base)[i], (uint8_t)(i & 0xFF))
        << "at byte " << i;
  }

  iree_shm_handle_close(&reader_handle);
  iree_shm_close(&reader);
  iree_shm_close(&writer);
}

}  // namespace
