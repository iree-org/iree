#!/usr/bin/env python3
# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generates runtime VM ISA derived files."""

import argparse
import difflib
import json
import re
import sys
from pathlib import Path


_SET_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_C_TAG_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_C_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_FIELD_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENCODING_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")
_MNEMONIC_RE = re.compile(r"^[a-z][a-z0-9_.]*$")
_INSTRUCTION_KINDS = frozenset(("instruction", "marker", "prefix"))
_REGISTER_BANKS = frozenset(("i32", "i64", "f32", "f64", "ref"))


def _format_opcode(opcode):
    return f"0x{opcode:02X}"


def _c_enum_name(value):
    return re.sub(r"[^A-Za-z0-9]+", "_", value).upper()


def _c_string_literal(value):
    return json.dumps(value)


def _c_string_view(value):
    return f"IREE_VM_ISA_SV({_c_string_literal(value)})"


def _feature_expr(feature):
    if feature is None:
        return "0"
    return f"iree_vm_FeatureBits_{feature}"


def _fail(message):
    raise ValueError(message)


def _require(condition, message):
    if not condition:
        _fail(message)


def _load_schema(schema_path):
    try:
        schema = json.loads(schema_path.read_text())
    except json.JSONDecodeError as exc:
        _fail(f"{schema_path}: invalid JSON: {exc}")
    return schema


def _validate_bytecode_version(schema):
    version = schema.get("bytecode_version")
    _require(isinstance(version, dict), "bytecode_version must be an object")
    for field_name in ("major", "minor"):
        value = version.get(field_name)
        _require(
            isinstance(value, int) and value >= 0,
            f"bytecode_version.{field_name} must be a non-negative integer",
        )


def _validate_template_field(field, field_path, parameter_kinds):
    _require(isinstance(field, dict), f"{field_path} must be an object")
    name = field.get("name")
    kind = field.get("kind")
    _require(
        isinstance(name, str) and _FIELD_RE.match(name),
        f"{field_path}.name must be a lowercase identifier",
    )
    _require(
        isinstance(kind, str) and _FIELD_RE.match(kind),
        f"{field_path}.kind must be a lowercase identifier",
    )

    if kind == "register":
        bank = field.get("bank")
        _require(isinstance(bank, str), f"{field_path}.bank must be a string")
        if bank.startswith("$"):
            parameter_name = bank[1:]
            _require(
                parameter_name in parameter_kinds,
                f"{field_path}.bank references unknown parameter '{parameter_name}'",
            )
            _require(
                parameter_kinds[parameter_name] == "register_bank",
                f"{field_path}.bank must reference a register_bank parameter",
            )
        else:
            _require(
                bank in _REGISTER_BANKS,
                f"{field_path}.bank must be one of {sorted(_REGISTER_BANKS)}",
            )

        access = field.get("access")
        _require(
            access in ("read", "write"),
            f"{field_path}.access must be 'read' or 'write'",
        )

        move = field.get("move")
        _require(
            move is None or move == "allow",
            f"{field_path}.move must be 'allow' if present",
        )

    field_type = field.get("type")
    _require(
        field_type is None or isinstance(field_type, str),
        f"{field_path}.type must be a string if present",
    )
    if isinstance(field_type, str) and not field_type.startswith("$"):
        _require(
            field_type == "any" or field_type in _REGISTER_BANKS,
            f"{field_path}.type must be 'any' or one of " f"{sorted(_REGISTER_BANKS)}",
        )

    for key, value in field.items():
        if not isinstance(value, str) or not value.startswith("$"):
            continue
        parameter_name = value[1:]
        _require(
            parameter_name in parameter_kinds,
            f"{field_path}.{key} references unknown parameter '{parameter_name}'",
        )


def _validate_operation_classes(schema):
    operation_classes = schema.get("operation_classes", {})
    _require(
        isinstance(operation_classes, dict),
        "operation_classes must be an object if present",
    )

    for class_name, operation_class in operation_classes.items():
        class_path = f"operation_classes.{class_name}"
        _require(
            isinstance(class_name, str) and _FIELD_RE.match(class_name),
            "operation class names must be lowercase identifiers",
        )
        _require(isinstance(operation_class, dict), f"{class_path} must be an object")

        description = operation_class.get("description")
        _require(
            description is None or isinstance(description, str),
            f"{class_path}.description must be a string if present",
        )

        parameters = operation_class.get("parameters", {})
        _require(
            isinstance(parameters, dict),
            f"{class_path}.parameters must be an object",
        )
        parameter_kinds = {}
        for parameter_name, parameter in parameters.items():
            parameter_path = f"{class_path}.parameters.{parameter_name}"
            _require(
                isinstance(parameter_name, str) and _FIELD_RE.match(parameter_name),
                f"{parameter_path} name must be a lowercase identifier",
            )
            _require(isinstance(parameter, dict), f"{parameter_path} must be an object")
            kind = parameter.get("kind")
            _require(
                isinstance(kind, str) and _FIELD_RE.match(kind),
                f"{parameter_path}.kind must be a lowercase identifier",
            )
            description = parameter.get("description")
            _require(
                description is None or isinstance(description, str),
                f"{parameter_path}.description must be a string if present",
            )
            parameter_kinds[parameter_name] = kind

        fields = operation_class.get("fields")
        _require(isinstance(fields, list), f"{class_path}.fields must be an array")
        _require(fields, f"{class_path}.fields must not be empty")
        field_names = set()
        for index, field in enumerate(fields):
            field_path = f"{class_path}.fields[{index}]"
            _validate_template_field(field, field_path, parameter_kinds)
            field_name = field["name"]
            _require(
                field_name not in field_names,
                f"duplicate field '{field_name}' in {class_path}",
            )
            field_names.add(field_name)

    return operation_classes


def _validate_opcode_sets(schema):
    opcode_sets = schema.get("opcode_sets")
    _require(isinstance(opcode_sets, list), "opcode_sets must be an array")
    _require(opcode_sets, "opcode_sets must not be empty")

    seen_ids = set()
    seen_tags = set()
    seen_enum_names = set()
    for index, opcode_set in enumerate(opcode_sets):
        _require(
            isinstance(opcode_set, dict), f"opcode_sets[{index}] must be an object"
        )
        opcode_set_id = opcode_set.get("id")
        c_tag = opcode_set.get("c_tag")
        enum_name = opcode_set.get("enum_name")
        prefix = opcode_set.get("prefix")
        feature = opcode_set.get("feature")

        _require(
            isinstance(opcode_set_id, str) and _SET_ID_RE.match(opcode_set_id),
            f"opcode_sets[{index}].id must be a lowercase identifier",
        )
        _require(
            opcode_set_id not in seen_ids,
            f"duplicate opcode set id '{opcode_set_id}'",
        )
        seen_ids.add(opcode_set_id)

        _require(
            isinstance(c_tag, str) and _C_TAG_RE.match(c_tag),
            f"opcode_sets[{index}].c_tag must be an uppercase C tag",
        )
        _require(c_tag not in seen_tags, f"duplicate opcode set C tag '{c_tag}'")
        seen_tags.add(c_tag)

        _require(
            isinstance(enum_name, str) and _C_IDENTIFIER_RE.match(enum_name),
            f"opcode_sets[{index}].enum_name must be a C identifier",
        )
        _require(
            enum_name not in seen_enum_names,
            f"duplicate opcode set enum name '{enum_name}'",
        )
        seen_enum_names.add(enum_name)

        _require(
            prefix is None or (isinstance(prefix, str) and _SYMBOL_RE.match(prefix)),
            f"opcode_sets[{index}].prefix must be null or a VM opcode symbol",
        )
        _require(
            feature is None or (isinstance(feature, str) and _C_TAG_RE.match(feature)),
            f"opcode_sets[{index}].feature must be null or an uppercase C tag",
        )

    return opcode_sets


def _validate_encoding_spec(encoding, encoding_path, operation_classes):
    _require(isinstance(encoding, dict), f"{encoding_path} must be an object")

    encoding_class = encoding.get("class")
    fields = encoding.get("fields")
    _require(
        (encoding_class is None) != (fields is None),
        f"{encoding_path} must have exactly one of 'class' or 'fields'",
    )

    if encoding_class is not None:
        _require(
            isinstance(encoding_class, str) and encoding_class in operation_classes,
            f"{encoding_path}.class must name an operation class",
        )
        parameters = encoding.get("parameters", {})
        _require(
            isinstance(parameters, dict),
            f"{encoding_path}.parameters must be an object",
        )
        expected_parameters = operation_classes[encoding_class].get("parameters", {})
        actual_parameters = set(parameters)
        missing_parameters = sorted(set(expected_parameters) - actual_parameters)
        extra_parameters = sorted(actual_parameters - set(expected_parameters))
        _require(
            not missing_parameters,
            f"{encoding_path}.parameters missing {missing_parameters}",
        )
        _require(
            not extra_parameters,
            f"{encoding_path}.parameters has unknown {extra_parameters}",
        )
        for parameter_name, parameter_value in parameters.items():
            _require(
                isinstance(parameter_value, str) and _FIELD_RE.match(parameter_value),
                f"{encoding_path}.parameters.{parameter_name} must be "
                "a lowercase identifier",
            )
            parameter_kind = expected_parameters[parameter_name]["kind"]
            if parameter_kind == "register_bank":
                _require(
                    parameter_value in _REGISTER_BANKS,
                    f"{encoding_path}.parameters.{parameter_name} must be one "
                    f"of {sorted(_REGISTER_BANKS)}",
                )
        return

    _require(
        isinstance(fields, list),
        f"{encoding_path}.fields must be an array",
    )
    _require(fields, f"{encoding_path}.fields must not be empty")
    field_names = set()
    for index, field in enumerate(fields):
        field_path = f"{encoding_path}.fields[{index}]"
        _validate_template_field(field, field_path, {})
        field_name = field["name"]
        _require(
            field_name not in field_names,
            f"duplicate field '{field_name}' in {encoding_path}",
        )
        field_names.add(field_name)


def _validate_encodings(schema, operation_classes):
    encodings = schema.get("encodings", {})
    _require(isinstance(encodings, dict), "encodings must be an object if present")

    for encoding_id, encoding in encodings.items():
        _require(
            isinstance(encoding_id, str) and _ENCODING_ID_RE.match(encoding_id),
            "encoding ids must be lowercase identifiers",
        )
        _validate_encoding_spec(encoding, f"encodings.{encoding_id}", operation_classes)

    return encodings


def _validate_instruction_encoding(
    instruction, instruction_path, operation_classes, encodings
):
    encoding = instruction.get("encoding")
    if encoding is None:
        return None

    if isinstance(encoding, str):
        _require(
            encoding in encodings,
            f"{instruction_path}.encoding references unknown encoding '{encoding}'",
        )
        instruction["encoding"] = encodings[encoding]
        instruction["encoding_ref"] = encoding
        return encoding

    _validate_encoding_spec(encoding, f"{instruction_path}.encoding", operation_classes)
    return None


def _validate_instructions(schema, opcode_sets, operation_classes, encodings):
    opcode_set_ids = {opcode_set["id"] for opcode_set in opcode_sets}
    instructions = schema.get("instructions")
    _require(isinstance(instructions, list), "instructions must be an array")

    instructions_by_set = {opcode_set_id: {} for opcode_set_id in opcode_set_ids}
    symbols_by_set = {opcode_set_id: set() for opcode_set_id in opcode_set_ids}
    mnemonics = set()
    used_encodings = set()
    for index, instruction in enumerate(instructions):
        instruction_path = f"instructions[{index}]"
        _require(isinstance(instruction, dict), f"{instruction_path} must be an object")

        opcode_set_id = instruction.get("opcode_set")
        _require(
            isinstance(opcode_set_id, str) and opcode_set_id in opcode_set_ids,
            f"{instruction_path}.opcode_set references an unknown opcode set",
        )

        opcode = instruction.get("opcode")
        _require(
            isinstance(opcode, int) and 0 <= opcode <= 0xFF,
            f"{instruction_path}.opcode must be an integer in [0, 255]",
        )
        _require(
            opcode not in instructions_by_set[opcode_set_id],
            f"duplicate opcode {_format_opcode(opcode)} in opcode set '{opcode_set_id}'",
        )

        symbol = instruction.get("symbol")
        _require(
            isinstance(symbol, str) and _SYMBOL_RE.match(symbol),
            f"{instruction_path}.symbol must be a VM opcode symbol",
        )
        _require(
            symbol not in symbols_by_set[opcode_set_id],
            f"duplicate symbol '{symbol}' in opcode set '{opcode_set_id}'",
        )
        symbols_by_set[opcode_set_id].add(symbol)

        kind = instruction.get("kind", "instruction")
        _require(
            kind in _INSTRUCTION_KINDS,
            f"{instruction_path}.kind must be one of {sorted(_INSTRUCTION_KINDS)}",
        )

        mnemonic = instruction.get("mnemonic")
        _require(
            mnemonic is None
            or (isinstance(mnemonic, str) and _MNEMONIC_RE.match(mnemonic)),
            f"{instruction_path}.mnemonic must be null or a lowercase dotted name",
        )

        if kind == "prefix":
            target_opcode_set = instruction.get("target_opcode_set")
            _require(
                isinstance(target_opcode_set, str)
                and target_opcode_set in opcode_set_ids,
                f"{instruction_path}.target_opcode_set must name an opcode set",
            )

        has_encoding = instruction.get("encoding") is not None
        encoding_ref = _validate_instruction_encoding(
            instruction, instruction_path, operation_classes, encodings
        )
        if has_encoding:
            _require(
                kind == "instruction",
                f"{instruction_path}.encoding is only valid on instructions",
            )
            _require(
                mnemonic is not None,
                f"{instruction_path}.mnemonic is required for encoded instructions",
            )
            _require(
                mnemonic not in mnemonics,
                f"duplicate instruction mnemonic '{mnemonic}'",
            )
            mnemonics.add(mnemonic)
        if encoding_ref is not None:
            used_encodings.add(encoding_ref)

        instructions_by_set[opcode_set_id][opcode] = instruction

    unused_encodings = sorted(set(encodings) - used_encodings)
    _require(not unused_encodings, f"unused encodings: {unused_encodings}")

    return instructions_by_set


def _validate_opcode_set_prefixes(opcode_sets, instructions_by_set):
    core_instructions = instructions_by_set.get("core", {})
    core_prefixes_by_symbol = {
        instruction["symbol"]: instruction
        for instruction in core_instructions.values()
        if instruction.get("kind") == "prefix"
    }
    for opcode_set in opcode_sets:
        prefix = opcode_set.get("prefix")
        if prefix is None:
            continue
        _require(
            prefix in core_prefixes_by_symbol,
            f"opcode set '{opcode_set['id']}' prefix '{prefix}' must name "
            "a core prefix instruction",
        )
        prefix_instruction = core_prefixes_by_symbol[prefix]
        _require(
            prefix_instruction.get("target_opcode_set") == opcode_set["id"],
            f"opcode set '{opcode_set['id']}' prefix '{prefix}' targets "
            f"'{prefix_instruction.get('target_opcode_set')}'",
        )


def _load_and_validate_schema(schema_path):
    schema = _load_schema(schema_path)
    _validate_bytecode_version(schema)
    operation_classes = _validate_operation_classes(schema)
    encodings = _validate_encodings(schema, operation_classes)
    opcode_sets = _validate_opcode_sets(schema)
    instructions_by_set = _validate_instructions(
        schema, opcode_sets, operation_classes, encodings
    )
    _validate_opcode_set_prefixes(opcode_sets, instructions_by_set)
    return opcode_sets, instructions_by_set


def _resolve_parameter(value, parameters):
    if isinstance(value, str) and value.startswith("$"):
        return parameters[value[1:]]
    return value


def _resolve_encoding_fields(encoding, operation_classes):
    if "class" in encoding:
        parameters = encoding.get("parameters", {})
        fields = operation_classes[encoding["class"]]["fields"]
    else:
        parameters = {}
        fields = encoding["fields"]

    resolved_fields = []
    for field in fields:
        resolved_field = {
            key: _resolve_parameter(value, parameters) for key, value in field.items()
        }
        resolved_fields.append(resolved_field)
    return resolved_fields


def _encoded_instructions(opcode_sets, instructions_by_set):
    for opcode_set in opcode_sets:
        opcode_set_id = opcode_set["id"]
        for opcode in range(256):
            instruction = instructions_by_set[opcode_set_id].get(opcode)
            if instruction and "encoding" in instruction:
                yield opcode_set, instruction


def _generate_encoding_table_header(schema_path):
    schema = _load_schema(schema_path)
    _validate_bytecode_version(schema)
    operation_classes = _validate_operation_classes(schema)
    encodings = _validate_encodings(schema, operation_classes)
    opcode_sets = _validate_opcode_sets(schema)
    instructions_by_set = _validate_instructions(
        schema, opcode_sets, operation_classes, encodings
    )
    _validate_opcode_set_prefixes(opcode_sets, instructions_by_set)

    resolved_field_kinds = set()
    for _, instruction in _encoded_instructions(opcode_sets, instructions_by_set):
        resolved_field_kinds.update(
            field["kind"]
            for field in _resolve_encoding_fields(
                instruction["encoding"], operation_classes
            )
        )

    lines = [
        "/*===- VM ISA generated file ------------------------------*- C -*-===*\\",
        "|*                                                                            *|",
        "|* IREE VM Encoding Tables                                                    *|",
        "|*                                                                            *|",
        "|* Generated by build_tools/scripts/generate_vm_isa.py from                   *|",
        "|* runtime/src/iree/vm/bytecode/isa/isa.json.                                *|",
        "|*                                                                            *|",
        "|* Automatically generated file, do not edit!                                 *|",
        "|*                                                                            *|",
        "\\*===----------------------------------------------------------------------===*/",
        "",
        "#ifndef IREE_VM_BYTECODE_ISA_ENCODING_TABLE_H_",
        "#define IREE_VM_BYTECODE_ISA_ENCODING_TABLE_H_",
        "",
        '#include "iree/vm/bytecode/isa/isa.h"',
        "",
        "#ifdef __cplusplus",
        'extern "C" {',
        "#endif  // __cplusplus",
        "",
        "typedef enum iree_vm_isa_opcode_set_e {",
    ]
    for index, opcode_set in enumerate(opcode_sets):
        lines.append(
            f"  IREE_VM_ISA_OPCODE_SET_{_c_enum_name(opcode_set['id'])} = {index},"
        )
    lines.extend(
        [
            f"  IREE_VM_ISA_OPCODE_SET_COUNT = {len(opcode_sets)},",
            "} iree_vm_isa_opcode_set_t;",
            "",
            "typedef enum iree_vm_isa_field_kind_e {",
        ]
    )
    for index, kind in enumerate(sorted(resolved_field_kinds)):
        lines.append(f"  IREE_VM_ISA_FIELD_KIND_{_c_enum_name(kind)} = {index},")
    lines.extend(
        [
            "} iree_vm_isa_field_kind_t;",
            "",
            "typedef enum iree_vm_isa_register_bank_e {",
            "  IREE_VM_ISA_REGISTER_BANK_NONE = 0,",
        ]
    )
    for bank in sorted(_REGISTER_BANKS):
        lines.append(f"  IREE_VM_ISA_REGISTER_BANK_{_c_enum_name(bank)},")
    lines.extend(
        [
            "} iree_vm_isa_register_bank_t;",
            "",
            "typedef enum iree_vm_isa_value_type_e {",
            "  IREE_VM_ISA_VALUE_TYPE_NONE = 0,",
            "  IREE_VM_ISA_VALUE_TYPE_ANY,",
        ]
    )
    for value_type in sorted(_REGISTER_BANKS):
        lines.append(f"  IREE_VM_ISA_VALUE_TYPE_{_c_enum_name(value_type)},")
    lines.extend(
        [
            "} iree_vm_isa_value_type_t;",
            "",
            "typedef enum iree_vm_isa_field_access_e {",
            "  IREE_VM_ISA_FIELD_ACCESS_NONE = 0,",
            "  IREE_VM_ISA_FIELD_ACCESS_READ,",
            "  IREE_VM_ISA_FIELD_ACCESS_WRITE,",
            "} iree_vm_isa_field_access_t;",
            "",
            "typedef struct iree_vm_isa_opcode_set_descriptor_t {",
            "  // Schema identifier for this opcode set.",
            "  iree_string_view_t id;",
            "  // Core opcode symbol used as the extension prefix, if any.",
            "  iree_string_view_t prefix_symbol;",
            "  // Core opcode byte used as the extension prefix, or 0 for core.",
            "  uint8_t prefix_opcode;",
            "  // Feature bits required by instructions in this opcode set.",
            "  iree_vm_FeatureBits_enum_t required_features;",
            "} iree_vm_isa_opcode_set_descriptor_t;",
            "",
            "typedef struct iree_vm_isa_field_t {",
            "  // Field name from the ISA schema.",
            "  iree_string_view_t name;",
            "  // Field encoder/decoder category.",
            "  iree_vm_isa_field_kind_t kind;",
            "  // Register bank for register fields, or NONE otherwise.",
            "  iree_vm_isa_register_bank_t register_bank;",
            "  // Value type for attributes and variadic fields.",
            "  iree_vm_isa_value_type_t value_type;",
            "  // Register access mode for register fields.",
            "  iree_vm_isa_field_access_t access;",
            "  // Whether ref register fields may use MOVE semantics.",
            "  bool allows_move;",
            "} iree_vm_isa_field_t;",
            "",
            "typedef struct iree_vm_isa_instruction_t {",
            "  // Textual mnemonic used by VM assembly.",
            "  iree_string_view_t mnemonic;",
            "  // C opcode symbol without the IREE_VM_OP_* prefix.",
            "  iree_string_view_t symbol;",
            "  // Named encoding referenced by the instruction.",
            "  iree_string_view_t encoding;",
            "  // Opcode set containing this instruction.",
            "  iree_vm_isa_opcode_set_t opcode_set;",
            "  // Opcode byte within the opcode set.",
            "  uint8_t opcode;",
            "  // Feature bits required by this instruction.",
            "  iree_vm_FeatureBits_enum_t required_features;",
            "  // Number of encoded fields following the opcode byte.",
            "  iree_host_size_t field_count;",
            "  // Ordered encoded fields following the opcode byte.",
            "  const iree_vm_isa_field_t* fields;",
            "} iree_vm_isa_instruction_t;",
            "",
            "IREE_API_EXPORT const iree_vm_isa_opcode_set_descriptor_t*",
            "iree_vm_isa_opcode_set_descriptor(iree_vm_isa_opcode_set_t opcode_set);",
            "",
            "IREE_API_EXPORT iree_host_size_t iree_vm_isa_instruction_count(void);",
            "",
            "IREE_API_EXPORT const iree_vm_isa_instruction_t*",
            "iree_vm_isa_instruction_table(void);",
            "",
            "IREE_API_EXPORT const iree_vm_isa_instruction_t*",
            "iree_vm_isa_lookup_mnemonic(iree_string_view_t mnemonic);",
            "",
            "IREE_API_EXPORT const iree_vm_isa_instruction_t*",
            "iree_vm_isa_lookup_opcode(iree_vm_isa_opcode_set_t opcode_set,",
            "                          uint8_t opcode);",
            "",
            "#ifdef __cplusplus",
            '}  // extern "C"',
            "#endif  // __cplusplus",
            "",
            "#endif  // IREE_VM_BYTECODE_ISA_ENCODING_TABLE_H_",
            "",
        ]
    )
    return "\n".join(lines)


def _value_type_enum(value):
    if value is None:
        return "IREE_VM_ISA_VALUE_TYPE_NONE"
    if value == "any":
        return "IREE_VM_ISA_VALUE_TYPE_ANY"
    return f"IREE_VM_ISA_VALUE_TYPE_{_c_enum_name(value)}"


def _register_bank_enum(value):
    if value is None:
        return "IREE_VM_ISA_REGISTER_BANK_NONE"
    return f"IREE_VM_ISA_REGISTER_BANK_{_c_enum_name(value)}"


def _field_access_enum(value):
    if value is None:
        return "IREE_VM_ISA_FIELD_ACCESS_NONE"
    return f"IREE_VM_ISA_FIELD_ACCESS_{_c_enum_name(value)}"


def _field_kind_enum(value):
    return f"IREE_VM_ISA_FIELD_KIND_{_c_enum_name(value)}"


def _opcode_set_enum(value):
    return f"IREE_VM_ISA_OPCODE_SET_{_c_enum_name(value)}"


def _find_core_prefix_instruction(opcode_set, instructions_by_set):
    prefix = opcode_set.get("prefix")
    if prefix is None:
        return None
    for instruction in instructions_by_set["core"].values():
        if instruction["symbol"] == prefix:
            return instruction
    _fail(f"opcode set '{opcode_set['id']}' has no resolved prefix")


def _generate_encoding_table_source(schema_path):
    schema = _load_schema(schema_path)
    _validate_bytecode_version(schema)
    operation_classes = _validate_operation_classes(schema)
    encodings = _validate_encodings(schema, operation_classes)
    opcode_sets = _validate_opcode_sets(schema)
    instructions_by_set = _validate_instructions(
        schema, opcode_sets, operation_classes, encodings
    )
    _validate_opcode_set_prefixes(opcode_sets, instructions_by_set)

    used_encodings = []
    seen_encodings = set()
    for _, instruction in _encoded_instructions(opcode_sets, instructions_by_set):
        encoding_ref = instruction["encoding_ref"]
        if encoding_ref not in seen_encodings:
            used_encodings.append(encoding_ref)
            seen_encodings.add(encoding_ref)

    lines = [
        "/*===- VM ISA generated file ------------------------------*- C -*-===*\\",
        "|*                                                                            *|",
        "|* IREE VM Encoding Tables                                                    *|",
        "|*                                                                            *|",
        "|* Generated by build_tools/scripts/generate_vm_isa.py from                   *|",
        "|* runtime/src/iree/vm/bytecode/isa/isa.json.                                *|",
        "|*                                                                            *|",
        "|* Automatically generated file, do not edit!                                 *|",
        "|*                                                                            *|",
        "\\*===----------------------------------------------------------------------===*/",
        "",
        '#include "iree/vm/bytecode/isa/encoding_table.h"',
        "",
        "#define IREE_VM_ISA_SV(literal) {literal, sizeof(literal) - 1}",
        "",
        "static const iree_vm_isa_opcode_set_descriptor_t",
        "    iree_vm_isa_opcode_set_descriptors[] = {",
    ]
    for opcode_set in opcode_sets:
        prefix_instruction = _find_core_prefix_instruction(
            opcode_set, instructions_by_set
        )
        prefix_opcode = (
            _format_opcode(prefix_instruction["opcode"]) if prefix_instruction else "0"
        )
        prefix_symbol = opcode_set.get("prefix") or ""
        lines.extend(
            [
                "        {",
                f"            .id = {_c_string_view(opcode_set['id'])},",
                f"            .prefix_symbol = {_c_string_view(prefix_symbol)},",
                f"            .prefix_opcode = {prefix_opcode},",
                f"            .required_features = {_feature_expr(opcode_set.get('feature'))},",
                "        },",
            ]
        )
    lines.extend(["};", ""])

    for encoding_ref in used_encodings:
        fields = _resolve_encoding_fields(encodings[encoding_ref], operation_classes)
        lines.append(
            f"static const iree_vm_isa_field_t "
            f"iree_vm_isa_{encoding_ref}_fields[] = {{"
        )
        for field in fields:
            lines.extend(
                [
                    "    {",
                    f"        .name = {_c_string_view(field['name'])},",
                    f"        .kind = {_field_kind_enum(field['kind'])},",
                    f"        .register_bank = {_register_bank_enum(field.get('bank'))},",
                    f"        .value_type = {_value_type_enum(field.get('type'))},",
                    f"        .access = {_field_access_enum(field.get('access'))},",
                    "        .allows_move = "
                    f"{'true' if field.get('move') == 'allow' else 'false'},",
                    "    },",
                ]
            )
        lines.extend(["};", ""])

    lines.append(
        "static const iree_vm_isa_instruction_t iree_vm_isa_instructions[] = {"
    )
    for opcode_set, instruction in _encoded_instructions(
        opcode_sets, instructions_by_set
    ):
        encoding_ref = instruction["encoding_ref"]
        lines.extend(
            [
                "    {",
                f"        .mnemonic = {_c_string_view(instruction['mnemonic'])},",
                f"        .symbol = {_c_string_view(instruction['symbol'])},",
                f"        .encoding = {_c_string_view(encoding_ref)},",
                f"        .opcode_set = {_opcode_set_enum(opcode_set['id'])},",
                f"        .opcode = {_format_opcode(instruction['opcode'])},",
                f"        .required_features = {_feature_expr(opcode_set.get('feature'))},",
                "        .field_count = "
                f"IREE_ARRAYSIZE(iree_vm_isa_{encoding_ref}_fields),",
                f"        .fields = iree_vm_isa_{encoding_ref}_fields,",
                "    },",
            ]
        )
    lines.extend(
        [
            "};",
            "",
            "IREE_API_EXPORT const iree_vm_isa_opcode_set_descriptor_t*",
            "iree_vm_isa_opcode_set_descriptor(iree_vm_isa_opcode_set_t opcode_set) {",
            "  if ((uint32_t)opcode_set >= IREE_VM_ISA_OPCODE_SET_COUNT) {",
            "    return NULL;",
            "  }",
            "  return &iree_vm_isa_opcode_set_descriptors[opcode_set];",
            "}",
            "",
            "IREE_API_EXPORT iree_host_size_t iree_vm_isa_instruction_count(void) {",
            "  return IREE_ARRAYSIZE(iree_vm_isa_instructions);",
            "}",
            "",
            "IREE_API_EXPORT const iree_vm_isa_instruction_t*",
            "iree_vm_isa_instruction_table(void) {",
            "  return iree_vm_isa_instructions;",
            "}",
            "",
            "IREE_API_EXPORT const iree_vm_isa_instruction_t*",
            "iree_vm_isa_lookup_mnemonic(iree_string_view_t mnemonic) {",
            "  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_vm_isa_instructions);",
            "       ++i) {",
            "    const iree_vm_isa_instruction_t* instruction =",
            "        &iree_vm_isa_instructions[i];",
            "    if (iree_string_view_equal(instruction->mnemonic, mnemonic)) {",
            "      return instruction;",
            "    }",
            "  }",
            "  return NULL;",
            "}",
            "",
            "IREE_API_EXPORT const iree_vm_isa_instruction_t*",
            "iree_vm_isa_lookup_opcode(iree_vm_isa_opcode_set_t opcode_set,",
            "                          uint8_t opcode) {",
            "  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(iree_vm_isa_instructions);",
            "       ++i) {",
            "    const iree_vm_isa_instruction_t* instruction =",
            "        &iree_vm_isa_instructions[i];",
            "    if (instruction->opcode_set == opcode_set && instruction->opcode == opcode) {",
            "      return instruction;",
            "    }",
            "  }",
            "  return NULL;",
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _generate_op_table(schema_path):
    opcode_sets, instructions_by_set = _load_and_validate_schema(schema_path)

    lines = [
        "/*===- VM ISA generated file ------------------------------*- C -*-===*\\",
        "|*                                                                            *|",
        "|* IREE VM Operation Tables                                                   *|",
        "|*                                                                            *|",
        "|* Generated by build_tools/scripts/generate_vm_isa.py from                   *|",
        "|* runtime/src/iree/vm/bytecode/isa/isa.json.                                *|",
        "|*                                                                            *|",
        "|* Automatically generated file, do not edit!                                 *|",
        "|*                                                                            *|",
        "\\*===----------------------------------------------------------------------===*/",
        "",
    ]

    for opcode_set in opcode_sets:
        c_tag = opcode_set["c_tag"]
        instructions = instructions_by_set[opcode_set["id"]]

        lines.append("typedef enum {")
        for opcode in range(256):
            instruction = instructions.get(opcode)
            if instruction:
                lines.append(
                    f"  IREE_VM_OP_{c_tag}_{instruction['symbol']} = "
                    f"{_format_opcode(opcode)},"
                )
            else:
                lines.append(f"  IREE_VM_OP_{c_tag}_RSV_{_format_opcode(opcode)},")
        lines.append(f"}} {opcode_set['enum_name']};")
        lines.append("")

        lines.append(f"#define IREE_VM_OP_{c_tag}_TABLE(OPC, RSV) \\")
        for opcode in range(256):
            instruction = instructions.get(opcode)
            suffix = " \\" if opcode != 0xFF else ""
            if instruction:
                lines.append(
                    f"    OPC({_format_opcode(opcode)}, {instruction['symbol']}){suffix}"
                )
            else:
                lines.append(f"    RSV({_format_opcode(opcode)}){suffix}")
        lines.append("")
        lines.append("")

    return "\n".join(lines) + "\n"


def _check_output(output_path, generated_output):
    existing_output = output_path.read_text()
    if existing_output == generated_output:
        return 0

    diff = difflib.unified_diff(
        existing_output.splitlines(keepends=True),
        generated_output.splitlines(keepends=True),
        fromfile=str(output_path),
        tofile=f"{output_path} (generated)",
    )
    sys.stderr.writelines(diff)
    return 1


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schema",
        type=Path,
        required=True,
        help="Path to runtime/src/iree/vm/bytecode/isa/isa.json.",
    )
    parser.add_argument(
        "--op-table",
        type=Path,
        help="Path to write or check op_table.h.",
    )
    parser.add_argument(
        "--encoding-table-header",
        type=Path,
        help="Path to write encoding_table.h.",
    )
    parser.add_argument(
        "--encoding-table-source",
        type=Path,
        help="Path to write encoding_table.c.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Checks that the output path already contains the generated output.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Writes the generated op table to stdout.",
    )
    args = parser.parse_args(argv)

    try:
        outputs = []
        if args.op_table is not None:
            outputs.append((args.op_table, _generate_op_table(args.schema)))
        if args.encoding_table_header is not None:
            outputs.append(
                (
                    args.encoding_table_header,
                    _generate_encoding_table_header(args.schema),
                )
            )
        if args.encoding_table_source is not None:
            outputs.append(
                (
                    args.encoding_table_source,
                    _generate_encoding_table_source(args.schema),
                )
            )

        if args.stdout:
            sys.stdout.write(_generate_op_table(args.schema))
            return 0
        if not outputs:
            parser.error(
                "at least one output path is required unless --stdout is specified"
            )
        if args.check:
            status = 0
            for output_path, generated_output in outputs:
                status |= _check_output(output_path, generated_output)
            return status
        for output_path, generated_output in outputs:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(generated_output)
        return 0
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
