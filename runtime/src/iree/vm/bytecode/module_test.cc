// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for bytecode_module.cc implementations.
// This means mostly just FlatBuffer verification, module interface functions,
// etc. bytecode_dispatch_test.cc covers actual dispatch.

#include "iree/vm/bytecode/module.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "iree/base/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module_test_module_c.h"

static bool operator==(const iree_vm_value_t& lhs,
                       const iree_vm_value_t& rhs) noexcept {
  if (lhs.type != rhs.type) return false;
  switch (lhs.type) {
    default:
    case IREE_VM_VALUE_TYPE_NONE:
      return true;  // none == none
    case IREE_VM_VALUE_TYPE_I8:
      return lhs.i8 == rhs.i8;
    case IREE_VM_VALUE_TYPE_I16:
      return lhs.i16 == rhs.i16;
    case IREE_VM_VALUE_TYPE_I32:
      return lhs.i32 == rhs.i32;
    case IREE_VM_VALUE_TYPE_I64:
      return lhs.i64 == rhs.i64;
    case IREE_VM_VALUE_TYPE_F32:
      return lhs.f32 == rhs.f32;
    case IREE_VM_VALUE_TYPE_F64:
      return lhs.f64 == rhs.f64;
  }
}

static std::ostream& operator<<(std::ostream& os,
                                const iree_vm_value_t& value) {
  switch (value.type) {
    default:
    case IREE_VM_VALUE_TYPE_NONE:
      return os << "??";
    case IREE_VM_VALUE_TYPE_I8:
      return os << value.i8;
    case IREE_VM_VALUE_TYPE_I16:
      return os << value.i16;
    case IREE_VM_VALUE_TYPE_I32:
      return os << value.i32;
    case IREE_VM_VALUE_TYPE_I64:
      return os << value.i64;
    case IREE_VM_VALUE_TYPE_F32:
      return os << value.f32;
    case IREE_VM_VALUE_TYPE_F64:
      return os << value.f64;
  }
}

template <size_t N>
static std::vector<iree_vm_value_t> MakeValuesList(const int32_t (&values)[N]) {
  std::vector<iree_vm_value_t> result;
  result.resize(N);
  for (size_t i = 0; i < N; ++i) result[i] = iree_vm_value_make_i32(values[i]);
  return result;
}

static std::vector<iree_vm_value_t> MakeValueRangeList(int32_t start,
                                                       int32_t end) {
  std::vector<iree_vm_value_t> result;
  result.resize(abs(start - end) + 1);
  int32_t value = start;
  int32_t delta = start < end ? 1 : -1;
  for (size_t i = 0; i < result.size(); ++i, value += delta) {
    result[i] = iree_vm_value_make_i32(value);
  }
  return result;
}

static bool operator==(const iree_vm_ref_t& lhs,
                       const iree_vm_ref_t& rhs) noexcept {
  return lhs.type == rhs.type && lhs.ptr == rhs.ptr;
}

static std::ostream& operator<<(std::ostream& os, const iree_vm_ref_t& value) {
  // Just nulls today.
  return os << (iree_vm_ref_is_null(&value) ? "(null)" : "??");
}

static std::vector<iree_vm_ref_t> MakeNullRefList(size_t count) {
  std::vector<iree_vm_ref_t> result;
  result.resize(count);
  return result;
}

namespace {

using iree::Status;
using iree::StatusCode;
using iree::StatusOr;
using iree::testing::status::IsOkAndHolds;
using iree::testing::status::StatusIs;
using iree::vm::ref;
using testing::Eq;

class VMBytecodeModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          iree_allocator_system(), &instance_));

    const auto* module_file_toc = iree_vm_bytecode_module_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            static_cast<iree_host_size_t>(module_file_toc->size)},
        iree_allocator_null(), iree_allocator_system(), &bytecode_module_));

    std::vector<iree_vm_module_t*> modules = {bytecode_module_};
    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
        iree_allocator_system(), &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(bytecode_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  StatusOr<std::vector<iree_vm_value_t>> RunFunction(
      const char* function_name, std::vector<iree_vm_value_t> inputs) {
    ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), inputs.size(),
                            iree_allocator_system(), &input_list));
    IREE_RETURN_IF_ERROR(iree_vm_list_resize(input_list.get(), inputs.size()));
    for (iree_host_size_t i = 0; i < inputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_set_value(input_list.get(), i, &inputs[i]));
    }

    ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             8, iree_allocator_system(),
                                             &output_list));

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));
    IREE_RETURN_IF_ERROR(
        iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/nullptr, input_list.get(), output_list.get(),
                       iree_allocator_system()));

    std::vector<iree_vm_value_t> outputs;
    outputs.resize(iree_vm_list_size(output_list.get()));
    for (iree_host_size_t i = 0; i < outputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_get_value(output_list.get(), i, &outputs[i]));
    }
    return outputs;
  }

  StatusOr<std::vector<iree_vm_ref_t>> RunFunction(
      const char* function_name, std::vector<iree_vm_ref_t> inputs) {
    ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), inputs.size(),
                            iree_allocator_system(), &input_list));
    IREE_RETURN_IF_ERROR(iree_vm_list_resize(input_list.get(), inputs.size()));
    for (iree_host_size_t i = 0; i < inputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_set_ref_retain(input_list.get(), i, &inputs[i]));
    }

    ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             8, iree_allocator_system(),
                                             &output_list));

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));
    IREE_RETURN_IF_ERROR(
        iree_vm_invoke(context_, function, IREE_VM_INVOCATION_FLAG_NONE,
                       /*policy=*/nullptr, input_list.get(), output_list.get(),
                       iree_allocator_system()));

    std::vector<iree_vm_ref_t> outputs;
    outputs.resize(iree_vm_list_size(output_list.get()));
    for (iree_host_size_t i = 0; i < outputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_get_ref_retain(output_list.get(), i, &outputs[i]));
    }
    return outputs;
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;

  enum : uint8_t {
    kOpCoreConstI32Zero = 12,
    kOpCoreConstI32 = 13,
    kOpCoreConstRefZero = 16,
    kOpCoreListAlloc = 18,
    kOpCoreAddI32 = 34,
    kOpCoreBranch = 86,
    kOpCoreReturn = 90,
    kOpCoreBlock = 121,
    kOpCoreDiscardRefs = 133,
  };

  struct BytecodeBuilder {
    std::vector<uint8_t> code;
    void EmitU8(uint8_t v) { code.push_back(v); }
    void EmitU16(uint16_t v) {
      code.push_back(static_cast<uint8_t>(v & 0xFFu));
      code.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
    }
    void EmitU32(uint32_t v) {
      code.push_back(static_cast<uint8_t>(v & 0xFFu));
      code.push_back(static_cast<uint8_t>((v >> 8) & 0xFFu));
      code.push_back(static_cast<uint8_t>((v >> 16) & 0xFFu));
      code.push_back(static_cast<uint8_t>((v >> 24) & 0xFFu));
    }
    void Align2() {
      if (code.size() & 1u) code.push_back(0);
    }
  };

  struct FunctionDescriptorLayout {
    int32_t bytecode_offset;
    int32_t bytecode_length;
    uint32_t requirements;
    uint16_t reserved0;
    uint16_t block_count;
    int16_t i32_register_count;
    int16_t ref_register_count;
  };
  static_assert(sizeof(FunctionDescriptorLayout) == 20,
                "unexpected FunctionDescriptor size");

  // Mutates function 2 (FuncIO8: cconv "0iiiiiiii_iiiiiiii", 20-byte bytecode)
  // in a copy of the embedded test module archive without relying on private
  // module implementation headers.
  iree_status_t MutateFuncIO8InArchive(const char* cconv_str, int16_t i32_regs,
                                       int16_t ref_regs,
                                       const std::vector<uint8_t>& bytecode,
                                       std::vector<uint8_t>* out_archive,
                                       size_t* out_code_offset = nullptr) {
    const auto* module_file_toc = iree_vm_bytecode_module_test_module_create();
    std::vector<uint8_t> archive(
        reinterpret_cast<const uint8_t*>(module_file_toc->data),
        reinterpret_cast<const uint8_t*>(module_file_toc->data) +
            module_file_toc->size);

    static const char kTargetCConv[] = "0iiiiiiii_iiiiiiii";
    const size_t kTargetCConvLen = sizeof(kTargetCConv) - 1;
    const size_t new_cconv_len = std::strlen(cconv_str);
    if (new_cconv_len > kTargetCConvLen || bytecode.size() > 20) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "test mutation exceeds FuncIO8 slot capacity");
    }

    auto cconv_it = std::search(archive.begin(), archive.end(), kTargetCConv,
                                kTargetCConv + kTargetCConvLen);
    if (cconv_it == archive.end() ||
        static_cast<size_t>(cconv_it - archive.begin()) < sizeof(uint32_t)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "FuncIO8 cconv string not found in archive");
    }
    const size_t cconv_offset = static_cast<size_t>(cconv_it - archive.begin());

    const uint8_t kExpectedFuncIO8Code[20] = {
        kOpCoreBlock, kOpCoreReturn, 0x08, 0x00, 0x07, 0x00, 0x06,
        0x00,         0x05,          0x00, 0x04, 0x00, 0x03, 0x00,
        0x02,         0x00,          0x01, 0x00, 0x00, 0x00,
    };
    auto code_it =
        std::search(archive.begin(), archive.end(), kExpectedFuncIO8Code,
                    kExpectedFuncIO8Code + sizeof(kExpectedFuncIO8Code));
    if (code_it == archive.end()) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "FuncIO8 bytecode not found in archive");
    }
    const size_t code_offset = static_cast<size_t>(code_it - archive.begin());

    FunctionDescriptorLayout expected_desc = {
        /*bytecode_offset=*/0,
        /*bytecode_length=*/20,
        /*requirements=*/0,
        /*reserved0=*/0,
        /*block_count=*/1,
        /*i32_register_count=*/8,
        /*ref_register_count=*/0,
    };
    const auto* desc_tail_begin =
        reinterpret_cast<const uint8_t*>(&expected_desc.bytecode_length);
    const auto* desc_tail_end =
        reinterpret_cast<const uint8_t*>(&expected_desc) +
        sizeof(expected_desc);
    auto desc_tail_it = std::search(archive.begin(), archive.end(),
                                    desc_tail_begin, desc_tail_end);
    if (desc_tail_it == archive.end() ||
        static_cast<size_t>(desc_tail_it - archive.begin()) < sizeof(int32_t)) {
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "FuncIO8 descriptor not found in archive");
    }
    const size_t desc_offset =
        static_cast<size_t>(desc_tail_it - archive.begin()) - sizeof(int32_t);
    FunctionDescriptorLayout orig_desc = {};
    std::memcpy(&orig_desc, archive.data() + desc_offset, sizeof(orig_desc));

    uint32_t cconv_len_u32 = static_cast<uint32_t>(new_cconv_len);
    std::memcpy(archive.data() + cconv_offset - sizeof(uint32_t),
                &cconv_len_u32, sizeof(uint32_t));
    std::memset(archive.data() + cconv_offset, 0, kTargetCConvLen);
    std::memcpy(archive.data() + cconv_offset, cconv_str, new_cconv_len);

    std::memset(archive.data() + code_offset, 0, sizeof(kExpectedFuncIO8Code));
    std::memcpy(archive.data() + code_offset, bytecode.data(), bytecode.size());

    FunctionDescriptorLayout desc = orig_desc;
    desc.bytecode_length = static_cast<int32_t>(bytecode.size());
    desc.block_count = 1;
    desc.i32_register_count = i32_regs;
    desc.ref_register_count = ref_regs;
    std::memcpy(archive.data() + desc_offset, &desc, sizeof(desc));

    *out_archive = std::move(archive);
    if (out_code_offset) *out_code_offset = code_offset;
    return iree_ok_status();
  }

  iree_status_t VerifyMutatedModule(const char* cconv_str, int16_t i32_regs,
                                    int16_t ref_regs,
                                    const std::vector<uint8_t>& bytecode) {
    std::vector<uint8_t> archive;
    IREE_RETURN_IF_ERROR(MutateFuncIO8InArchive(cconv_str, i32_regs, ref_regs,
                                                bytecode, &archive));
    iree_vm_module_t* module = nullptr;
    iree_status_t status = iree_vm_bytecode_module_create(
        instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
        iree_make_const_byte_span(archive.data(), archive.size()),
        iree_allocator_null(), iree_allocator_system(), &module);
    if (module) iree_vm_module_release(module);
    return status;
  }

  // Returns false in builds where per-function bytecode verification is
  // compiled out (e.g. -DIREE_VM_BYTECODE_VERIFICATION_ENABLE=0 in
  // runtime_small).
  bool IsBytecodeFunctionVerificationEnabled() {
#if defined(IREE_VM_BYTECODE_VERIFICATION_ENABLE) && \
    !IREE_VM_BYTECODE_VERIFICATION_ENABLE
    return false;
#else
    const std::vector<uint8_t> kMissingBlockMarker = {0xFFu};
    iree_status_t status = VerifyMutatedModule(
        "0v_v", /*i32_regs=*/0, /*ref_regs=*/0, kMissingBlockMarker);
    bool enabled = !iree_status_is_ok(status);
    iree_status_ignore(status);
    return enabled;
#endif
  }
};

TEST_F(VMBytecodeModuleTest, FuncIOEmpty) {
  EXPECT_THAT(RunFunction("FuncIOEmpty", std::vector<iree_vm_value_t>()),
              IsOkAndHolds(Eq(std::vector<iree_vm_value_t>())));
}

TEST_F(VMBytecodeModuleTest, FuncIO1) {
  EXPECT_THAT(RunFunction("FuncIO1", MakeValuesList({1})),
              IsOkAndHolds(Eq(MakeValuesList({1}))));
}

TEST_F(VMBytecodeModuleTest, FuncIO8) {
  EXPECT_THAT(RunFunction("FuncIO8", MakeValueRangeList(0, 7)),
              IsOkAndHolds(Eq(MakeValueRangeList(7, 0))));
}

TEST_F(VMBytecodeModuleTest, FuncIO600) {
  EXPECT_THAT(RunFunction("FuncIO600", MakeNullRefList(600)),
              IsOkAndHolds(Eq(MakeNullRefList(600))));
}

TEST_F(VMBytecodeModuleTest, VerifierControlValidBytecodeAccepted) {
  BytecodeBuilder b;
  b.EmitU8(kOpCoreBlock);
  b.EmitU8(kOpCoreConstI32);
  b.EmitU32(42);
  b.EmitU16(1);
  b.EmitU8(kOpCoreAddI32);
  b.EmitU16(0);
  b.EmitU16(1);
  b.EmitU16(0);
  b.EmitU8(kOpCoreReturn);
  b.Align2();
  b.EmitU16(1);
  b.EmitU16(0);
  IREE_EXPECT_OK(
      VerifyMutatedModule("0i_i", /*i32_regs=*/2, /*ref_regs=*/1, b.code));
}

TEST_F(VMBytecodeModuleTest, VerifierSeam01ValidateRegI32AndTypeIdBounds) {
  if (!IsBytecodeFunctionVerificationEnabled()) {
    GTEST_SKIP() << "Bytecode function verification is disabled in this build.";
  }

  // ConstI32 with ref register bit set (0x8000) in i32 result slot.
  {
    BytecodeBuilder b;
    b.EmitU8(kOpCoreBlock);
    b.EmitU8(kOpCoreConstI32);
    b.EmitU32(0x41414141u);
    b.EmitU16(0x8000u);
    b.EmitU8(kOpCoreReturn);
    b.Align2();
    b.EmitU16(0);
    EXPECT_THAT(Status(VerifyMutatedModule("0v_v", /*i32_regs=*/2,
                                           /*ref_regs=*/1, b.code)),
                StatusIs(StatusCode::kInvalidArgument));
  }

  // ConstI32 with out-of-range i32 register ordinal 10 >=
  // i32_register_count(2).
  {
    BytecodeBuilder b;
    b.EmitU8(kOpCoreBlock);
    b.EmitU8(kOpCoreConstI32);
    b.EmitU32(0x41414141u);
    b.EmitU16(10);
    b.EmitU8(kOpCoreReturn);
    b.Align2();
    b.EmitU16(0);
    EXPECT_THAT(Status(VerifyMutatedModule("0v_v", /*i32_regs=*/2,
                                           /*ref_regs=*/1, b.code)),
                StatusIs(StatusCode::kOutOfRange));
  }

  // ListAlloc with type_id=2 when module->type_count == 2 (types 0 and 1).
  {
    BytecodeBuilder b;
    b.EmitU8(kOpCoreBlock);
    b.EmitU8(kOpCoreListAlloc);
    b.EmitU32(2);
    b.EmitU16(0);
    b.EmitU16(0x8000u);
    b.EmitU8(kOpCoreReturn);
    b.Align2();
    b.EmitU16(0);
    EXPECT_THAT(Status(VerifyMutatedModule("0i_v", /*i32_regs=*/2,
                                           /*ref_regs=*/1, b.code)),
                StatusIs(StatusCode::kOutOfRange));
  }
}

TEST_F(VMBytecodeModuleTest, VerifierMapTypeOutOfBoundsReturnsUndefinedType) {
  // Build a valid 18-byte "0i_r" function in FuncIO8's slot so module creation
  // succeeds whether bytecode function verification is enabled or disabled.
  BytecodeBuilder valid_code;
  valid_code.EmitU8(kOpCoreBlock);
  for (int i = 0; i < 3; ++i) {
    valid_code.EmitU8(kOpCoreConstI32Zero);
    valid_code.EmitU16(1);
  }
  valid_code.EmitU8(kOpCoreConstRefZero);
  valid_code.EmitU16(0xC000u);
  valid_code.EmitU8(kOpCoreReturn);
  valid_code.EmitU16(1);
  valid_code.EmitU16(0xC000u);
  ASSERT_EQ(valid_code.code.size(), 18u);

  std::vector<uint8_t> archive;
  size_t code_offset = 0;
  IREE_ASSERT_OK(MutateFuncIO8InArchive("0i_r", /*i32_regs=*/2, /*ref_regs=*/1,
                                        valid_code.code, &archive,
                                        &code_offset));

  iree_vm_module_t* raw_module = nullptr;
  IREE_ASSERT_OK(iree_vm_bytecode_module_create(
      instance_, IREE_VM_BYTECODE_MODULE_FLAG_NONE,
      iree_make_const_byte_span(archive.data(), archive.size()),
      iree_allocator_null(), iree_allocator_system(), &raw_module));
  std::unique_ptr<iree_vm_module_t, void (*)(iree_vm_module_t*)> module(
      raw_module, iree_vm_module_release);

  iree_vm_context_t* raw_context = nullptr;
  std::vector<iree_vm_module_t*> modules = {module.get()};
  IREE_ASSERT_OK(iree_vm_context_create_with_modules(
      instance_, IREE_VM_CONTEXT_FLAG_NONE, modules.size(), modules.data(),
      iree_allocator_system(), &raw_context));
  std::unique_ptr<iree_vm_context_t, void (*)(iree_vm_context_t*)> context(
      raw_context, iree_vm_context_release);

  iree_vm_function_t function = {};
  IREE_ASSERT_OK(iree_vm_module_lookup_function_by_name(
      module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
      iree_make_cstring_view("FuncIO8"), &function));

  // Overwrite the zero-copy bytecode body with ListAlloc(type_id, %i0) -> %R0
  // followed by Return(%R0) to exercise iree_vm_map_type in dispatch.c for
  // negative and out-of-bounds type_id values (module->type_count == 2).
  for (uint32_t raw_type_id : {2u, 0xFFFFFFFFu, 99u}) {
    BytecodeBuilder invoke_code;
    invoke_code.EmitU8(kOpCoreBlock);
    invoke_code.EmitU8(kOpCoreListAlloc);
    invoke_code.EmitU32(raw_type_id);
    invoke_code.EmitU16(0);
    invoke_code.EmitU16(0xC000u);
    invoke_code.EmitU8(kOpCoreConstI32Zero);
    invoke_code.EmitU16(1);
    invoke_code.EmitU8(kOpCoreReturn);
    invoke_code.EmitU16(1);
    invoke_code.EmitU16(0xC000u);
    ASSERT_EQ(invoke_code.code.size(), valid_code.code.size());
    std::memcpy(archive.data() + code_offset, invoke_code.code.data(),
                invoke_code.code.size());

    ref<iree_vm_list_t> input_list;
    IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                       iree_allocator_system(), &input_list));
    iree_vm_value_t cap = iree_vm_value_make_i32(0);
    IREE_ASSERT_OK(iree_vm_list_push_value(input_list.get(), &cap));

    ref<iree_vm_list_t> output_list;
    IREE_ASSERT_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(), 1,
                                       iree_allocator_system(), &output_list));

    IREE_EXPECT_OK(iree_vm_invoke(context.get(), function,
                                  IREE_VM_INVOCATION_FLAG_NONE,
                                  /*policy=*/nullptr, input_list.get(),
                                  output_list.get(), iree_allocator_system()));
    EXPECT_EQ(iree_vm_list_size(output_list.get()), 1u);
  }
}

TEST_F(VMBytecodeModuleTest,
       VerifierSeam02ExactRegisterCountAndNegativeDescriptorCount) {
  if (IsBytecodeFunctionVerificationEnabled()) {
    // Declared i32_register_count=3 (< 4=round_up_to_pow2(3)) with 4-arg cconv.
    {
      BytecodeBuilder b;
      b.EmitU8(kOpCoreBlock);
      b.EmitU8(kOpCoreReturn);
      b.Align2();
      b.EmitU16(0);
      EXPECT_THAT(Status(VerifyMutatedModule("0iiii_v", /*i32_regs=*/3,
                                             /*ref_regs=*/1, b.code)),
                  StatusIs(StatusCode::kInvalidArgument));
    }

    // Declared i32_register_count=3 with Return reading ordinal 3.
    {
      BytecodeBuilder b;
      b.EmitU8(kOpCoreBlock);
      b.EmitU8(kOpCoreReturn);
      b.Align2();
      b.EmitU16(1);
      b.EmitU16(3);
      EXPECT_THAT(Status(VerifyMutatedModule("0v_i", /*i32_regs=*/3,
                                             /*ref_regs=*/1, b.code)),
                  StatusIs(StatusCode::kOutOfRange));
    }
  }

  // Negative i32_register_count = -1 in FunctionDescriptor (checked by
  // iree_vm_bytecode_module_flatbuffer_verify in all builds).
  {
    BytecodeBuilder b;
    b.EmitU8(kOpCoreBlock);
    b.EmitU8(kOpCoreReturn);
    b.Align2();
    b.EmitU16(0);
    EXPECT_THAT(Status(VerifyMutatedModule("0v_v", /*i32_regs=*/-1,
                                           /*ref_regs=*/1, b.code)),
                StatusIs(StatusCode::kInvalidArgument));
  }

  // Negative ref_register_count = -1 in FunctionDescriptor.
  {
    BytecodeBuilder b;
    b.EmitU8(kOpCoreBlock);
    b.EmitU8(kOpCoreReturn);
    b.Align2();
    b.EmitU16(0);
    EXPECT_THAT(Status(VerifyMutatedModule("0v_v", /*i32_regs=*/1,
                                           /*ref_regs=*/-1, b.code)),
                StatusIs(StatusCode::kInvalidArgument));
  }
}

TEST_F(VMBytecodeModuleTest, VerifierSeam03BranchRemapCrossBankRejected) {
  if (!IsBytecodeFunctionVerificationEnabled()) {
    GTEST_SKIP() << "Bytecode function verification is disabled in this build.";
  }

  BytecodeBuilder b;
  b.EmitU8(kOpCoreBlock);
  b.EmitU8(kOpCoreBranch);
  b.EmitU32(0);
  b.Align2();
  b.EmitU16(1);
  b.EmitU16(0);
  b.EmitU16(0x8000u);
  EXPECT_THAT(Status(VerifyMutatedModule("0i_v", /*i32_regs=*/2, /*ref_regs=*/2,
                                         b.code)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(VMBytecodeModuleTest, VerifierSeam04DiscardRefsNonRefRegisterRejected) {
  if (!IsBytecodeFunctionVerificationEnabled()) {
    GTEST_SKIP() << "Bytecode function verification is disabled in this build.";
  }

  BytecodeBuilder b;
  b.EmitU8(kOpCoreBlock);
  b.EmitU8(kOpCoreDiscardRefs);
  b.Align2();
  b.EmitU16(1);
  b.EmitU16(0);
  b.EmitU8(kOpCoreReturn);
  b.Align2();
  b.EmitU16(0);
  EXPECT_THAT(Status(VerifyMutatedModule("0i_v", /*i32_regs=*/2, /*ref_regs=*/1,
                                         b.code)),
              StatusIs(StatusCode::kInvalidArgument));
}

TEST_F(VMBytecodeModuleTest,
       VerifierSeam05CConvRegisterListLengthMismatchRejected) {
  if (!IsBytecodeFunctionVerificationEnabled()) {
    GTEST_SKIP() << "Bytecode function verification is disabled in this build.";
  }

  BytecodeBuilder b;
  b.EmitU8(kOpCoreBlock);
  b.EmitU8(kOpCoreReturn);
  b.Align2();
  b.EmitU16(2);
  b.EmitU16(0);
  b.EmitU16(1);
  EXPECT_THAT(Status(VerifyMutatedModule("0i_i", /*i32_regs=*/2, /*ref_regs=*/1,
                                         b.code)),
              StatusIs(StatusCode::kInvalidArgument));
}

}  // namespace
