// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/target_id.h"

typedef enum iree_hal_amdgpu_target_feature_support_bits_e {
  IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_NONE = 0u,
  IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_SRAMECC = 1u << 0,
  IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_XNACK = 1u << 1,
} iree_hal_amdgpu_target_feature_support_bits_t;
typedef uint32_t iree_hal_amdgpu_target_feature_support_flags_t;

typedef struct iree_hal_amdgpu_target_id_mapping_t {
  // Exact HSA ISA processor name.
  iree_string_view_t exact_processor;
  // Code-object processor selected for the exact processor.
  iree_string_view_t code_object_processor;
  // Feature support flags from
  // iree_hal_amdgpu_target_feature_support_bits_t.
  iree_hal_amdgpu_target_feature_support_flags_t feature_support;
} iree_hal_amdgpu_target_id_mapping_t;

static const iree_hal_amdgpu_target_id_mapping_t
    iree_hal_amdgpu_target_id_mappings[] = {
#include "iree/hal/drivers/amdgpu/util/target_id_map.inl"
};

static bool iree_hal_amdgpu_parse_decimal_digit(char c, uint32_t* out_value) {
  if (c < '0' || c > '9') return false;
  *out_value = (uint32_t)(c - '0');
  return true;
}

static bool iree_hal_amdgpu_parse_hex_digit(char c, uint32_t* out_value) {
  if (c >= '0' && c <= '9') {
    *out_value = (uint32_t)(c - '0');
    return true;
  } else if (c >= 'a' && c <= 'f') {
    *out_value = (uint32_t)(c - 'a' + 10);
    return true;
  } else if (c >= 'A' && c <= 'F') {
    *out_value = (uint32_t)(c - 'A' + 10);
    return true;
  }
  return false;
}

static bool iree_hal_amdgpu_parse_decimal_number(iree_string_view_t value,
                                                 uint32_t* out_number) {
  if (iree_string_view_is_empty(value)) return false;
  uint64_t number = 0;
  for (iree_host_size_t i = 0; i < value.size; ++i) {
    uint32_t digit = 0;
    if (!iree_hal_amdgpu_parse_decimal_digit(value.data[i], &digit)) {
      return false;
    }
    number = number * 10 + digit;
    if (number > UINT32_MAX) return false;
  }
  *out_number = (uint32_t)number;
  return true;
}

static bool iree_hal_amdgpu_gfxip_version_equal(
    iree_hal_amdgpu_gfxip_version_t lhs, iree_hal_amdgpu_gfxip_version_t rhs) {
  return lhs.major == rhs.major && lhs.minor == rhs.minor &&
         lhs.stepping == rhs.stepping;
}

static bool iree_hal_amdgpu_parse_exact_processor(
    iree_string_view_t processor,
    iree_hal_amdgpu_gfxip_version_t* out_version) {
  memset(out_version, 0, sizeof(*out_version));
  if (!iree_string_view_consume_prefix(&processor, IREE_SV("gfx"))) {
    return false;
  }

  uint32_t major0 = 0;
  uint32_t major1 = 0;
  uint32_t minor = 0;
  uint32_t stepping = 0;
  if (processor.size == 4 &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[0], &major0) &&
      major0 == 1 &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[1], &major1) &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[2], &minor) &&
      iree_hal_amdgpu_parse_hex_digit(processor.data[3], &stepping)) {
    out_version->major = 10 + major1;
    out_version->minor = minor;
    out_version->stepping = stepping;
    return true;
  }
  if (processor.size == 3 &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[0], &major0) &&
      iree_hal_amdgpu_parse_decimal_digit(processor.data[1], &minor) &&
      iree_hal_amdgpu_parse_hex_digit(processor.data[2], &stepping)) {
    out_version->major = major0;
    out_version->minor = minor;
    out_version->stepping = stepping;
    return true;
  }
  return false;
}

static const iree_hal_amdgpu_target_id_mapping_t*
iree_hal_amdgpu_target_id_lookup_mapping(iree_string_view_t exact_processor) {
  for (iree_host_size_t i = 0;
       i < IREE_ARRAYSIZE(iree_hal_amdgpu_target_id_mappings); ++i) {
    if (iree_string_view_equal(
            exact_processor,
            iree_hal_amdgpu_target_id_mappings[i].exact_processor)) {
      return &iree_hal_amdgpu_target_id_mappings[i];
    }
  }
  return NULL;
}

static void iree_hal_amdgpu_target_id_apply_known_feature_support(
    iree_hal_amdgpu_target_id_t* target_id) {
  if (target_id->kind != IREE_HAL_AMDGPU_TARGET_KIND_EXACT) return;
  const iree_hal_amdgpu_target_id_mapping_t* mapping =
      iree_hal_amdgpu_target_id_lookup_mapping(target_id->processor);
  if (mapping == NULL) return;

  if (target_id->sramecc == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY &&
      !iree_any_bit_set(mapping->feature_support,
                        IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_SRAMECC)) {
    target_id->sramecc = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED;
  }
  if (target_id->xnack == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY &&
      !iree_any_bit_set(mapping->feature_support,
                        IREE_HAL_AMDGPU_TARGET_FEATURE_SUPPORT_XNACK)) {
    target_id->xnack = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED;
  }
}

static bool iree_hal_amdgpu_parse_generic_processor(
    iree_string_view_t processor,
    iree_hal_amdgpu_gfxip_version_t* out_version) {
  memset(out_version, 0, sizeof(*out_version));
  if (!iree_string_view_consume_prefix(&processor, IREE_SV("gfx")) ||
      !iree_string_view_consume_suffix(&processor, IREE_SV("-generic"))) {
    return false;
  }

  iree_string_view_t major = iree_string_view_empty();
  iree_string_view_t minor = iree_string_view_empty();
  if (iree_string_view_split(processor, '-', &major, &minor) == -1) {
    major = processor;
  } else if (iree_string_view_find_char(minor, '-', 0) !=
             IREE_STRING_VIEW_NPOS) {
    return false;
  }

  uint32_t major_value = 0;
  uint32_t minor_value = 0;
  if (!iree_hal_amdgpu_parse_decimal_number(major, &major_value)) {
    return false;
  }
  if (!iree_string_view_is_empty(minor) &&
      !iree_hal_amdgpu_parse_decimal_number(minor, &minor_value)) {
    return false;
  }
  out_version->major = major_value;
  out_version->minor = minor_value;
  out_version->stepping = 0;
  return true;
}

static iree_status_t iree_hal_amdgpu_target_id_parse_processor(
    iree_string_view_t processor, iree_hal_amdgpu_target_id_t* out_target_id) {
  if (iree_string_view_is_empty(processor)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target ID has an empty processor name");
  }

  out_target_id->processor = processor;
  if (iree_hal_amdgpu_parse_generic_processor(processor,
                                              &out_target_id->version)) {
    out_target_id->kind = IREE_HAL_AMDGPU_TARGET_KIND_GENERIC;
    return iree_ok_status();
  }
  if (iree_hal_amdgpu_parse_exact_processor(processor,
                                            &out_target_id->version)) {
    out_target_id->kind = IREE_HAL_AMDGPU_TARGET_KIND_EXACT;
    return iree_ok_status();
  }
  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "unsupported AMDGPU target processor syntax: %.*s",
                          (int)processor.size, processor.data);
}

static iree_status_t iree_hal_amdgpu_target_id_parse_feature(
    iree_string_view_t feature,
    iree_hal_amdgpu_target_feature_state_t* inout_sramecc,
    iree_hal_amdgpu_target_feature_state_t* inout_xnack) {
  if (feature.size < 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target feature suffix is empty");
  }

  const char selector = feature.data[feature.size - 1];
  iree_hal_amdgpu_target_feature_state_t state =
      IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  if (selector == '+') {
    state = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON;
  } else if (selector == '-') {
    state = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target feature suffix missing +/-: %.*s",
                            (int)feature.size, feature.data);
  }

  iree_string_view_t name = iree_string_view_remove_suffix(feature, /*n=*/1);
  iree_hal_amdgpu_target_feature_state_t* feature_state = NULL;
  if (iree_string_view_equal(name, IREE_SV("sramecc"))) {
    feature_state = inout_sramecc;
  } else if (iree_string_view_equal(name, IREE_SV("xnack"))) {
    feature_state = inout_xnack;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported AMDGPU target feature suffix: %.*s",
                            (int)feature.size, feature.data);
  }
  if (*feature_state != IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "duplicate AMDGPU target feature suffix: %.*s",
                            (int)name.size, name.data);
  }
  *feature_state = state;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_target_id_parse(
    iree_string_view_t value, iree_hal_amdgpu_target_id_parse_flags_t flags,
    iree_hal_amdgpu_target_id_t* out_target_id) {
  IREE_ASSERT_ARGUMENT(out_target_id);
  memset(out_target_id, 0, sizeof(*out_target_id));
  out_target_id->sramecc = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;
  out_target_id->xnack = IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY;

  const iree_hal_amdgpu_target_id_parse_flags_t known_flags =
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_HSA_PREFIX |
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY |
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES;
  if (IREE_UNLIKELY(iree_any_bit_set(flags, ~known_flags))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unknown AMDGPU target ID parse flags: 0x%x",
                            flags);
  }

  const bool allow_hsa_prefix = iree_any_bit_set(
      flags, IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_HSA_PREFIX);
  const bool allow_arch_only =
      flags == IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE ||
      iree_any_bit_set(flags,
                       IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_ARCH_ONLY);
  const bool allow_feature_suffixes = iree_any_bit_set(
      flags, IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES);

  if (iree_string_view_starts_with(value, IREE_SV("amdgcn-amd-amdhsa--"))) {
    if (!allow_hsa_prefix) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "HSA ISA prefix not accepted in AMDGPU target ID: %.*s",
          (int)value.size, value.data);
    }
    value = iree_string_view_substr(value, IREE_SV("amdgcn-amd-amdhsa--").size,
                                    IREE_STRING_VIEW_NPOS);
  } else if (!allow_arch_only) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bare processor name not accepted in AMDGPU target ID: %.*s",
        (int)value.size, value.data);
  }

  iree_string_view_t processor = value;
  iree_string_view_t feature_list = iree_string_view_empty();
  if (iree_string_view_split(value, ':', &processor, &feature_list) != -1) {
    if (!allow_feature_suffixes) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "AMDGPU target feature suffixes not accepted in target ID: %.*s",
          (int)value.size, value.data);
    }
    if (iree_string_view_is_empty(feature_list) ||
        value.data[value.size - 1] == ':') {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "AMDGPU target feature suffix is empty");
    }
  }
  IREE_RETURN_IF_ERROR(
      iree_hal_amdgpu_target_id_parse_processor(processor, out_target_id));

  while (!iree_string_view_is_empty(feature_list)) {
    iree_string_view_t feature = iree_string_view_empty();
    iree_string_view_t remaining_features = iree_string_view_empty();
    if (iree_string_view_split(feature_list, ':', &feature,
                               &remaining_features) == -1) {
      feature = feature_list;
    }
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_parse_feature(
        feature, &out_target_id->sramecc, &out_target_id->xnack));
    feature_list = remaining_features;
  }
  iree_hal_amdgpu_target_id_apply_known_feature_support(out_target_id);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_target_id_parse_hsa_isa_name(
    iree_string_view_t value, iree_hal_amdgpu_target_id_t* out_target_id) {
  return iree_hal_amdgpu_target_id_parse(
      value,
      IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_HSA_PREFIX |
          IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_ALLOW_FEATURE_SUFFIXES,
      out_target_id);
}

typedef struct iree_hal_amdgpu_target_id_formatter_t {
  // Caller-provided output buffer; NULL when only querying required length.
  char* buffer;
  // Caller-provided output buffer capacity in bytes.
  iree_host_size_t capacity;
  // Required output length excluding the NUL terminator.
  iree_host_size_t length;
} iree_hal_amdgpu_target_id_formatter_t;

static void iree_hal_amdgpu_target_id_formatter_append(
    iree_hal_amdgpu_target_id_formatter_t* formatter,
    iree_string_view_t value) {
  if (formatter->buffer != NULL && formatter->capacity > 0 &&
      formatter->length < formatter->capacity - 1) {
    const iree_host_size_t available =
        formatter->capacity - 1 - formatter->length;
    const iree_host_size_t copy_length = iree_min(value.size, available);
    memcpy(formatter->buffer + formatter->length, value.data, copy_length);
    formatter->buffer[formatter->length + copy_length] = 0;
  }
  formatter->length += value.size;
}

static void iree_hal_amdgpu_target_id_formatter_append_feature(
    iree_hal_amdgpu_target_id_formatter_t* formatter, iree_string_view_t name,
    iree_hal_amdgpu_target_feature_state_t state) {
  if (state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF) {
    iree_hal_amdgpu_target_id_formatter_append(formatter, IREE_SV(":"));
    iree_hal_amdgpu_target_id_formatter_append(formatter, name);
    iree_hal_amdgpu_target_id_formatter_append(formatter, IREE_SV("-"));
  } else if (state == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON) {
    iree_hal_amdgpu_target_id_formatter_append(formatter, IREE_SV(":"));
    iree_hal_amdgpu_target_id_formatter_append(formatter, name);
    iree_hal_amdgpu_target_id_formatter_append(formatter, IREE_SV("+"));
  }
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_target_id_format(const iree_hal_amdgpu_target_id_t* target_id,
                                 iree_host_size_t buffer_capacity, char* buffer,
                                 iree_host_size_t* out_buffer_length) {
  IREE_ASSERT_ARGUMENT(target_id);
  if (IREE_UNLIKELY(iree_string_view_is_empty(target_id->processor))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target ID has no processor name");
  }

  iree_hal_amdgpu_target_id_formatter_t formatter = {
      .buffer = buffer,
      .capacity = buffer_capacity,
      .length = 0,
  };
  if (buffer != NULL && buffer_capacity > 0) buffer[0] = 0;
  iree_hal_amdgpu_target_id_formatter_append(&formatter, target_id->processor);
  iree_hal_amdgpu_target_id_formatter_append_feature(
      &formatter, IREE_SV("sramecc"), target_id->sramecc);
  iree_hal_amdgpu_target_id_formatter_append_feature(
      &formatter, IREE_SV("xnack"), target_id->xnack);
  if (out_buffer_length != NULL) {
    *out_buffer_length = formatter.length;
  }
  if (buffer != NULL && buffer_capacity <= formatter.length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "AMDGPU target ID buffer capacity exceeded");
  }
  return iree_ok_status();
}

static bool iree_hal_amdgpu_target_id_lookup_code_object_processor(
    iree_string_view_t exact_processor,
    iree_string_view_t* out_code_object_processor) {
  const iree_hal_amdgpu_target_id_mapping_t* mapping =
      iree_hal_amdgpu_target_id_lookup_mapping(exact_processor);
  if (mapping == NULL) return false;
  *out_code_object_processor = mapping->code_object_processor;
  return true;
}

IREE_API_EXPORT iree_status_t
iree_hal_amdgpu_target_id_lookup_code_object_target(
    const iree_hal_amdgpu_target_id_t* exact_target_id,
    iree_hal_amdgpu_target_id_t* out_code_object_target_id) {
  IREE_ASSERT_ARGUMENT(exact_target_id);
  IREE_ASSERT_ARGUMENT(out_code_object_target_id);
  memset(out_code_object_target_id, 0, sizeof(*out_code_object_target_id));

  if (exact_target_id->kind != IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU code-object target lookup requires an "
                            "exact processor target ID");
  }
  iree_string_view_t code_object_processor = exact_target_id->processor;
  iree_hal_amdgpu_target_id_lookup_code_object_processor(
      exact_target_id->processor, &code_object_processor);
  if (iree_string_view_equal(code_object_processor,
                             exact_target_id->processor)) {
    *out_code_object_target_id = *exact_target_id;
  } else {
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_id_parse(
        code_object_processor, IREE_HAL_AMDGPU_TARGET_ID_PARSE_FLAG_NONE,
        out_code_object_target_id));
  }
  out_code_object_target_id->sramecc = exact_target_id->sramecc;
  out_code_object_target_id->xnack = exact_target_id->xnack;
  return iree_ok_status();
}

static bool iree_hal_amdgpu_generic_version_compatible(
    iree_hal_amdgpu_gfxip_version_t code_object_version,
    iree_hal_amdgpu_gfxip_version_t agent_version) {
  if (code_object_version.major != agent_version.major) return false;
  if (code_object_version.minor > agent_version.minor) return false;
  if (code_object_version.minor == agent_version.minor &&
      code_object_version.stepping > agent_version.stepping) {
    return false;
  }
  return true;
}

static bool iree_hal_amdgpu_target_feature_compatible(
    iree_hal_amdgpu_target_feature_state_t code_object_feature,
    iree_hal_amdgpu_target_feature_state_t agent_feature) {
  if (code_object_feature == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON ||
      code_object_feature == IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF) {
    return code_object_feature == agent_feature;
  }
  return true;
}

static uint32_t iree_hal_amdgpu_generic_code_object_minimum_version(
    const iree_hal_amdgpu_target_id_t* generic_target_id) {
  return generic_target_id->kind == IREE_HAL_AMDGPU_TARGET_KIND_GENERIC ? 1 : 0;
}

IREE_API_EXPORT iree_hal_amdgpu_target_compatibility_t
iree_hal_amdgpu_target_id_check_compatible(
    const iree_hal_amdgpu_target_id_t* code_object_target_id,
    const iree_hal_amdgpu_target_id_t* agent_target_id) {
  IREE_ASSERT_ARGUMENT(code_object_target_id);
  IREE_ASSERT_ARGUMENT(agent_target_id);

  iree_hal_amdgpu_target_compatibility_t compatibility =
      IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE;
  if (code_object_target_id->kind == IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
    if (agent_target_id->kind != IREE_HAL_AMDGPU_TARGET_KIND_EXACT ||
        !iree_hal_amdgpu_gfxip_version_equal(code_object_target_id->version,
                                             agent_target_id->version)) {
      compatibility |= IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_PROCESSOR;
    }
  } else {
    iree_string_view_t agent_code_object_processor = iree_string_view_empty();
    if (agent_target_id->kind == IREE_HAL_AMDGPU_TARGET_KIND_EXACT) {
      if (!iree_hal_amdgpu_target_id_lookup_code_object_processor(
              agent_target_id->processor, &agent_code_object_processor)) {
        if (!iree_hal_amdgpu_generic_version_compatible(
                code_object_target_id->version, agent_target_id->version)) {
          compatibility |=
              IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY;
        }
      } else if (!iree_string_view_equal(code_object_target_id->processor,
                                         agent_code_object_processor)) {
        compatibility |=
            IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY;
      }
    } else if (!iree_string_view_equal(code_object_target_id->processor,
                                       agent_target_id->processor)) {
      compatibility |=
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY;
    }
    const uint32_t minimum_generic_version =
        iree_hal_amdgpu_generic_code_object_minimum_version(
            code_object_target_id);
    if (code_object_target_id->generic_version != 0 &&
        code_object_target_id->generic_version < minimum_generic_version) {
      compatibility |=
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_VERSION;
    }
  }
  if (!iree_hal_amdgpu_target_feature_compatible(code_object_target_id->sramecc,
                                                 agent_target_id->sramecc)) {
    compatibility |= IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC;
  }
  if (!iree_hal_amdgpu_target_feature_compatible(code_object_target_id->xnack,
                                                 agent_target_id->xnack)) {
    compatibility |= IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK;
  }
  return compatibility;
}

static void iree_hal_amdgpu_target_compatibility_formatter_append_reason(
    iree_hal_amdgpu_target_id_formatter_t* formatter,
    iree_host_size_t* inout_reason_count, iree_string_view_t reason) {
  if (*inout_reason_count != 0) {
    iree_hal_amdgpu_target_id_formatter_append(formatter, IREE_SV(", "));
  }
  iree_hal_amdgpu_target_id_formatter_append(formatter, reason);
  ++*inout_reason_count;
}

IREE_API_EXPORT iree_status_t iree_hal_amdgpu_target_compatibility_format(
    iree_hal_amdgpu_target_compatibility_t compatibility,
    iree_host_size_t buffer_capacity, char* buffer,
    iree_host_size_t* out_buffer_length) {
  iree_hal_amdgpu_target_id_formatter_t formatter = {
      .buffer = buffer,
      .capacity = buffer_capacity,
      .length = 0,
  };
  if (buffer != NULL && buffer_capacity > 0) buffer[0] = 0;

  iree_host_size_t reason_count = 0;
  if (compatibility == IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_COMPATIBLE) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("compatible"));
  }
  if (iree_any_bit_set(
          compatibility,
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_PROCESSOR)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("processor"));
  }
  if (iree_any_bit_set(
          compatibility,
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_FAMILY)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("generic family"));
  }
  if (iree_any_bit_set(
          compatibility,
          IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_GENERIC_VERSION)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("generic version"));
  }
  if (iree_any_bit_set(compatibility,
                       IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_SRAMECC)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("sramecc"));
  }
  if (iree_any_bit_set(compatibility,
                       IREE_HAL_AMDGPU_TARGET_COMPATIBILITY_MISMATCH_XNACK)) {
    iree_hal_amdgpu_target_compatibility_formatter_append_reason(
        &formatter, &reason_count, IREE_SV("xnack"));
  }
  if (out_buffer_length != NULL) {
    *out_buffer_length = formatter.length;
  }
  if (buffer != NULL && buffer_capacity <= formatter.length) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "AMDGPU target compatibility buffer capacity exceeded");
  }
  return iree_ok_status();
}
