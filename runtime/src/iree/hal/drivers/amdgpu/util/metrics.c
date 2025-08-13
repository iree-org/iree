#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "iree/base/api.h"

typedef uint32_t metric_flags_t;
enum metric_flag_bits_e {
  METRIC_FLAG_EMULATED = 1u << 0,          // "emulated" tag in the XML
  METRIC_FLAG_NONDETERMINISTIC = 1u << 1,  // "nondeterministic" tag in the XML
  METRIC_FLAG_LEVEL = 1u << 2,             // "level" tag in the XML
  METRIC_FLAG_C1 = 1u << 3,                // "C1" tag in the XML
  METRIC_FLAG_C2 = 1u << 4,                // "C2" tag in the XML
  METRIC_FLAG_GLOBAL = 1u << 5,            // "global" tag in the XML
  METRIC_FLAG_PER_SIMD = 1u << 6,          // "per-simd" tag in the XML
  METRIC_FLAG_PER_SQ = 1u << 7,            // "per-SQ" tag in the XML
  METRIC_FLAG_PER_BANK = 1u << 8,          // "per-Bank" tag in the XML
};

typedef uint32_t metric_type_t;
enum metric_type_e {
  // Metric is an event counter ("block" and "event" in the XML).
  METRIC_TYPE_EVENT = 0,
  // Metric is an expression ("expr" in the XML).
  METRIC_TYPE_EXPR,
};

typedef struct metric_t {
  // Metric type as determined by which attributes are present in the XML.
  metric_type_t type;
  // Name of the metric ("name" attribute in the XML).
  iree_string_view_t name;
  // Description text with any flags stripped off.
  iree_string_view_t description;
  // Flags parsed from tagged strings in the description text.
  // Flags are inconsistent in the file but generally appear as a
  // comma-delimited list in parenthesis or braces:
  //   `(per-simd, emulated)`
  //   `{level, nondeterministic, C1}`
  // Sometimes the cases are wrong. Yay.
  metric_flags_t flags;
  union {
    // Event information, if type is METRIC_TYPE_EVENT.
    struct {
      // Parsed block name ("block" attribute in the XML mapped to the enum).
      hsa_ven_amd_aqlprofile_block_name_t block;
      // Counter ID ("event" attribute in the XML).
      uint32_t counter_id;
    } event;
    // Expression, if type is METRIC_TYPE_EXPR.
    iree_string_view_t expr;
  };
} metric_t;

// Callback issued for each metric matching the architecture.
typedef iree_status_t (*metric_callback_t)(void* user_data,
                                           const metric_t* metric);

// Given the basic_counters.xml file in memory and the architecture string (like
// "gfx1032" or "gfx942") specifying the group enumerate each <metric> entry and
// call the provided metric callback function with the parsed metric.
//
// Do not allocate any memory during the enumeration. Multiple passes over the
// document may be required to first identify the group (like an arch value of
// "gfx942" leading to scanning for `<gfx942...`), then depth-first recursively
// enumerate any "base" group (like `<gfx942 base="gfx940">` causing the metrics
// from `<gfx940>` to be enumerated first followed by any contained within
// `<gfx942>`), and finally parse each metric.
//
// `<metric>` elements map into the `metric_t` struct. The parser should
// determine whether the metric is an event (has an "event" attribute) or an
// expression (has an "expr" attribute), parse the name description, extract
// any flags by looking for string tags in the description and afterward
// stripping them to store just the description without the string tags.
// Expressions are parsed like strings (though may not be double quoted).
//
// NOTE: the XML files may contain comments in the form of `## comment` that
// should be ignored. For example:
// ```xml
//   ## L2 Cache Metrics
// ```
//
// Example of an event metric:
//
// ```xml
// <metric name="SQ_INST_LEVEL_GDS" block=SQ event=98 descr="Number of in-flight
// GDS instructions. Set next counter to ACCUM_PREV and divide by INSTS_GDS for
// average latency. {level, nondeterministic, C1}"></metric>
// ```
//
// is parsed as:
//
// ```
// metric_t:
// - type: METRIC_TYPE_EVENT
// - name: "SQ_INST_LEVEL_GDS"
// - description: "Number of in-flight GDS instructions. Set next counter to
// ACCUM_PREV and divide by INSTS_GDS for average latency."
// - flags: METRIC_FLAG_LEVEL | METRIC_FLAG_NONDETERMINISTIC | METRIC_FLAG_C1
// - event:
//   - block: HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ
//   - counter_id: 98
// ```
//
// Example of an expression metric:
//
// ```xml
// <metric name="MeanOccupancyPerCU"
// expr=reduce(SQ_LEVEL_WAVES,sum)*0+reduce(SQ_ACCUM_PREV_HIRES,sum)/reduce(GRBM_GUI_ACTIVE,sum)/CU_NUM
// descr="Mean occupancy per compute unit."></metric>
// ```
//
// is parsed as:
//
// ```
// metric_t:
// - type: METRIC_TYPE_EXPRESSION
// - name: "MeanOccupancyPerCU"
// - description: "Mean occupancy per compute unit."
// - flags: 0
// - expr:
// "reduce(SQ_LEVEL_WAVES,sum)*0+reduce(SQ_ACCUM_PREV_HIRES,sum)/reduce(GRBM_GUI_ACTIVE,sum)/CU_NUM"
// ```
iree_status_t enumerate_arch_metrics(iree_string_view_t basic_counters_xml,
                                     iree_string_view_t arch,
                                     metric_callback_t callback,
                                     void* user_data);

// Skip whitespace at the beginning of a string view.
static iree_string_view_t sv_skip_whitespace(iree_string_view_t sv) {
  while (sv.size > 0 && isspace((unsigned char)*sv.data)) {
    ++sv.data;
    --sv.size;
  }
  return sv;
}

// Map a block name string to the corresponding HSA enum value.
// Example: "GRBM" -> HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM
static bool map_block_name(iree_string_view_t name,
                           hsa_ven_amd_aqlprofile_block_name_t* out) {
  // Common block mappings based on the XML files.
  if (iree_string_view_equal(name, IREE_SV("GRBM"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("SQ"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("TA"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("TD"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("TCP"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("TCC"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("TCA"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("GL2C"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("CPC"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("CPF"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF;
    return true;
  } else if (iree_string_view_equal(name, IREE_SV("SPI"))) {
    *out = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI;
    return true;
  }
  // Add more mappings as needed based on actual HSA enum values.
  return false;
}

// Parse a single flag from a string.
// Example: "per-simd" -> METRIC_FLAG_PER_SIMD
static metric_flags_t parse_flag(iree_string_view_t flag) {
  flag = iree_string_view_trim(flag);

  // Case-insensitive comparison.
  if (iree_string_view_equal_case(flag, IREE_SV("emulated")))
    return METRIC_FLAG_EMULATED;
  if (iree_string_view_equal_case(flag, IREE_SV("nondeterministic")))
    return METRIC_FLAG_NONDETERMINISTIC;
  if (iree_string_view_equal_case(flag, IREE_SV("level")))
    return METRIC_FLAG_LEVEL;
  if (iree_string_view_equal_case(flag, IREE_SV("c1"))) return METRIC_FLAG_C1;
  if (iree_string_view_equal_case(flag, IREE_SV("c2"))) return METRIC_FLAG_C2;
  if (iree_string_view_equal_case(flag, IREE_SV("global")))
    return METRIC_FLAG_GLOBAL;
  if (iree_string_view_equal_case(flag, IREE_SV("per-simd")))
    return METRIC_FLAG_PER_SIMD;
  if (iree_string_view_equal_case(flag, IREE_SV("per-sq")))
    return METRIC_FLAG_PER_SQ;
  if (iree_string_view_equal_case(flag, IREE_SV("per-bank")))
    return METRIC_FLAG_PER_BANK;

  return 0;
}

// Extract flags from a description and return the cleaned description.
// Flags appear in parentheses () or braces {}.
// Example: "Description text. (per-simd, emulated)" -> flags =
// METRIC_FLAG_PER_SIMD | METRIC_FLAG_EMULATED
static void extract_flags(iree_string_view_t* description,
                          metric_flags_t* flags) {
  *flags = 0;

  // Look for flag patterns at the end of the description.
  iree_host_size_t paren_start = IREE_STRING_VIEW_NPOS;
  iree_host_size_t brace_start = IREE_STRING_VIEW_NPOS;

  // Find the last occurrence of '(' or '{'.
  for (int i = (int)description->size - 1; i >= 0; --i) {
    if (description->data[i] == '(' && paren_start == IREE_STRING_VIEW_NPOS) {
      paren_start = i;
    } else if (description->data[i] == '{' &&
               brace_start == IREE_STRING_VIEW_NPOS) {
      brace_start = i;
    }
  }

  // Process whichever is later (closer to end).
  iree_host_size_t flag_start =
      paren_start > brace_start ? paren_start : brace_start;
  char closing = paren_start > brace_start ? ')' : '}';

  if (flag_start == IREE_STRING_VIEW_NPOS) return;

  // Find closing delimiter.
  iree_host_size_t flag_end = iree_string_view_find_char(
      (iree_string_view_t){description->data + flag_start,
                           description->size - flag_start},
      closing, 0);

  if (flag_end == IREE_STRING_VIEW_NPOS) return;

  // Extract flag content.
  iree_string_view_t flag_content = {.data = description->data + flag_start + 1,
                                     .size = flag_end - 1};

  // Parse comma-separated flags.
  while (flag_content.size > 0) {
    iree_host_size_t comma = iree_string_view_find_char(flag_content, ',', 0);
    iree_string_view_t single_flag = iree_string_view_empty();

    if (comma == IREE_STRING_VIEW_NPOS) {
      single_flag = flag_content;
      flag_content.size = 0;
    } else {
      single_flag = (iree_string_view_t){flag_content.data, comma};
      flag_content.data += comma + 1;
      flag_content.size -= comma + 1;
    }

    *flags |= parse_flag(single_flag);
  }

  // Remove the flag portion from the description.
  description->size = flag_start;
  *description = iree_string_view_trim(*description);
}

// Find an XML attribute value within a tag.
// Example: in '<metric name="SQ_WAVES" block=SQ event=4>',
// find_attribute(..., "name") returns "SQ_WAVES"
static bool find_attribute(iree_string_view_t tag_content,
                           const char* attr_name, iree_string_view_t* value) {
  iree_host_size_t pos =
      iree_string_view_find_str(tag_content, attr_name, strlen(attr_name));
  if (pos == IREE_STRING_VIEW_NPOS) return false;

  // Move past attribute name.
  tag_content.data += pos + strlen(attr_name);
  tag_content.size -= pos + strlen(attr_name);

  // Skip whitespace and '='.
  tag_content = sv_skip_whitespace(tag_content);
  if (tag_content.size == 0 || tag_content.data[0] != '=') return false;
  ++tag_content.data;
  --tag_content.size;
  tag_content = sv_skip_whitespace(tag_content);

  if (tag_content.size == 0) return false;

  // Handle quoted and unquoted values.
  if (tag_content.data[0] == '"') {
    // Quoted value.
    ++tag_content.data;
    --tag_content.size;
    iree_host_size_t end = iree_string_view_find_char(tag_content, '"', 0);
    if (end == IREE_STRING_VIEW_NPOS) return false;
    *value = (iree_string_view_t){tag_content.data, end};
  } else {
    // Unquoted value - find end by whitespace or '>'.
    size_t end = 0;
    while (end < tag_content.size &&
           !isspace((unsigned char)tag_content.data[end]) &&
           tag_content.data[end] != '>') {
      ++end;
    }
    *value = (iree_string_view_t){tag_content.data, end};
  }

  return true;
}

// Process a single metric tag and invoke the callback.
// Example: <metric name="SQ_WAVES" block=SQ event=4 descr="Count number of
// waves sent to SQs. (per-simd, emulated, global)"></metric>
static iree_status_t process_metric(iree_string_view_t tag_content,
                                    metric_callback_t callback,
                                    void* user_data) {
  metric_t metric = {0};

  // Extract name (required).
  if (!find_attribute(tag_content, "name", &metric.name)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metric missing name attribute");
  }

  // Extract description (required).
  iree_string_view_t description = iree_string_view_empty();
  if (!find_attribute(tag_content, "descr", &description)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metric missing descr attribute");
  }

  // Extract flags from description.
  metric.description = description;
  extract_flags(&metric.description, &metric.flags);

  // Determine metric type and parse type-specific attributes.
  iree_string_view_t event_str = iree_string_view_empty();
  iree_string_view_t expr_str = iree_string_view_empty();
  iree_string_view_t block_str = iree_string_view_empty();
  bool has_event = find_attribute(tag_content, "event", &event_str);
  bool has_expr = find_attribute(tag_content, "expr", &expr_str);

  if (has_event) {
    metric.type = METRIC_TYPE_EVENT;

    // Parse block name.
    if (!find_attribute(tag_content, "block", &block_str)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "event metric missing block attribute");
    }
    if (!map_block_name(block_str, &metric.event.block)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unknown block name");
    }

    // Parse event ID.
    if (!iree_string_view_atoi_uint32(event_str, &metric.event.counter_id)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "invalid event ID");
    }
  } else if (has_expr) {
    metric.type = METRIC_TYPE_EXPR;
    metric.expr = expr_str;
  } else {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "metric must have either event or expr attribute");
  }

  // Invoke callback.
  return callback(user_data, &metric);
}

// Process all metrics within a group's content.
// Searches for <metric> tags and processes each one.
static iree_status_t process_group_metrics(iree_string_view_t content,
                                           metric_callback_t callback,
                                           void* user_data) {
  while (content.size > 0) {
    // Skip comments starting with '#'.
    if (iree_string_view_starts_with(content, IREE_SV("#"))) {
      iree_host_size_t newline = iree_string_view_find_char(content, '\n', 0);
      if (newline == IREE_STRING_VIEW_NPOS) break;
      content.data += newline + 1;
      content.size -= newline + 1;
      continue;
    }

    // Find next metric tag.
    iree_host_size_t metric_start =
        iree_string_view_find_str(content, IREE_SV("<metric"), 7);
    if (metric_start == IREE_STRING_VIEW_NPOS) break;

    content.data += metric_start;
    content.size -= metric_start;

    // Find end of tag.
    iree_host_size_t tag_end =
        iree_string_view_find_str(content, IREE_SV("</metric>"), 9);
    iree_host_size_t self_closing_end =
        iree_string_view_find_str(content, IREE_SV("/>"), 2);

    iree_host_size_t end = IREE_STRING_VIEW_NPOS;
    bool self_closing = false;

    if (self_closing_end != IREE_STRING_VIEW_NPOS &&
        (tag_end == IREE_STRING_VIEW_NPOS || self_closing_end < tag_end)) {
      end = self_closing_end + 2;
      self_closing = true;
    } else if (tag_end != IREE_STRING_VIEW_NPOS) {
      end = tag_end + 9;  // Length of "</metric>".
    } else {
      // Also check for just '>'.
      iree_host_size_t simple_end = iree_string_view_find_char(content, '>', 0);
      if (simple_end != IREE_STRING_VIEW_NPOS) {
        end = simple_end + 1;
        self_closing = true;
      }
    }

    if (end == IREE_STRING_VIEW_NPOS) break;

    // Extract tag content.
    iree_string_view_t tag_content = {content.data + 7,
                                      end - 7};  // Skip "<metric".
    if (self_closing && tag_content.size > 2) {
      tag_content.size -= 2;  // Remove "/>".
    }

    // Process the metric.
    IREE_RETURN_IF_ERROR(process_metric(tag_content, callback, user_data));

    // Move past this metric.
    content.data += end;
    content.size -= end;
  }

  return iree_ok_status();
}

// Find a specific architecture group in the XML and return its content.
// Example: find_arch_group(..., "gfx942") looks for <gfx942...> ... </gfx942>
static bool find_arch_group(iree_string_view_t xml, iree_string_view_t arch,
                            iree_string_view_t* group_content,
                            iree_string_view_t* base_arch) {
  // Build the opening tag pattern.
  char pattern[64];
  snprintf(pattern, sizeof(pattern), "<%.*s", (int)arch.size, arch.data);

  iree_host_size_t start =
      iree_string_view_find_str(xml, pattern, strlen(pattern));
  if (start == IREE_STRING_VIEW_NPOS) return false;

  // Move to the tag.
  xml.data += start;
  xml.size -= start;

  // Find the end of the opening tag.
  iree_host_size_t open_end = iree_string_view_find_char(xml, '>', 0);
  if (open_end == IREE_STRING_VIEW_NPOS) return false;

  // Extract tag attributes to find base.
  iree_string_view_t tag_attrs = {xml.data + arch.size + 1,
                                  open_end - arch.size - 1};
  if (base_arch) {
    if (!find_attribute(tag_attrs, "base", base_arch)) {
      base_arch->data = NULL;
      base_arch->size = 0;
    }
  }

  // Find closing tag.
  char closing_pattern[64];
  snprintf(closing_pattern, sizeof(closing_pattern), "</%.*s>", (int)arch.size,
           arch.data);

  iree_host_size_t close_start =
      iree_string_view_find_str(xml, closing_pattern, strlen(closing_pattern));
  if (close_start == IREE_STRING_VIEW_NPOS) return false;

  // Extract content between tags.
  group_content->data = xml.data + open_end + 1;
  group_content->size = close_start - open_end - 1;

  return true;
}

// Main enumeration function.
iree_status_t enumerate_arch_metrics(iree_string_view_t basic_counters_xml,
                                     iree_string_view_t arch,
                                     metric_callback_t callback,
                                     void* user_data) {
  // Find the architecture group.
  iree_string_view_t group_content = iree_string_view_empty();
  iree_string_view_t base_arch = iree_string_view_empty();
  if (!find_arch_group(basic_counters_xml, arch, &group_content, &base_arch)) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "architecture group not found");
  }

  // If there's a base architecture, process it first (recursively).
  if (base_arch.size > 0) {
    IREE_RETURN_IF_ERROR(enumerate_arch_metrics(basic_counters_xml, base_arch,
                                                callback, user_data));
  }

  // Process metrics in this group.
  return process_group_metrics(group_content, callback, user_data);
}
