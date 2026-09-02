# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# CMake side of the renamed IREE compiler ABI (IREE_COMPILER_DYNAMIC_PLUGINS).
#
# libIREECompiler renames every llvm/mlir C++ symbol to an IREE-private
# Itanium name before link so it can co-reside with another LLVM/MLIR copy in
# the same process, while dynamic plugins rename their own archives the same
# way and resolve the shared closure from the compiler library. The rename
# decision and spelling live in build_tools/bazel/gen_rename_map.py, shared
# with the Bazel implementation so both builds produce the same ABI.
#
# llvm-objcopy consumes archives, so everything to be renamed first becomes a
# static archive. Renaming the linked shared library instead is not an option:
# rewriting .dynsym would invalidate the ELF hash sections.

# Names a file after a target, safe for use in declared outputs
# (object-library targets contain dots, e.g. `obj.MLIRCAPIIR`).
function(iree_renamed_link_sanitize_name VALUE OUT_VAR)
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _result "${VALUE}")
  set(${OUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()

# Evaluates $<BOOL:...> by its own rules: false only for the empty string and
# CMake's false constants, so a path is true. if() cannot stand in for this -
# under CMP0054 a quoted argument that is not a known constant is false, which
# would invert exactly the paths LLVM guards its link flags with.
function(iree_renamed_link_genex_bool VALUE OUT_VAR)
  string(TOUPPER "${VALUE}" _upper)
  if(_upper STREQUAL "" OR _upper STREQUAL "0" OR _upper STREQUAL "FALSE"
     OR _upper STREQUAL "OFF" OR _upper STREQUAL "N" OR _upper STREQUAL "NO"
     OR _upper STREQUAL "IGNORE" OR _upper STREQUAL "NOTFOUND"
     OR _upper MATCHES "-NOTFOUND$")
    set(${OUT_VAR} FALSE PARENT_SCOPE)
  else()
    set(${OUT_VAR} TRUE PARENT_SCOPE)
  endif()
endfunction()

# Rewrites one static archive to the renamed ABI.
#
# Parameters:
#   OUT_VAR: variable receiving the renamed archive path.
#   NAME: base name for the outputs (must be unique in this directory).
#   INPUT: archive path (generator expressions allowed).
#   DEPENDS: extra dependencies of the rename actions.
function(iree_renamed_archive OUT_VAR)
  cmake_parse_arguments(_RULE "" "NAME;INPUT" "DEPENDS" ${ARGN})
  set(_dir "${CMAKE_CURRENT_BINARY_DIR}/renamed")
  set(_map "${_dir}/${_RULE_NAME}.rename_map")
  set(_out "${_dir}/lib${_RULE_NAME}.renamed.a")
  set(_script "${IREE_ROOT_DIR}/build_tools/bazel/gen_rename_map.py")
  add_custom_command(
    OUTPUT "${_map}"
    COMMAND
      "${Python3_EXECUTABLE}" "${_script}"
      --nm "$<TARGET_FILE:llvm-nm>"
      --cxxfilt "$<TARGET_FILE:llvm-cxxfilt>"
      --input "${_RULE_INPUT}"
      --out "${_map}"
      --symbol-prefix "IREE18"
    DEPENDS "${_RULE_INPUT}" "${_script}" llvm-nm llvm-cxxfilt ${_RULE_DEPENDS}
    COMMENT "Computing llvm/mlir rename map for ${_RULE_NAME}"
  )
  add_custom_command(
    OUTPUT "${_out}"
    COMMAND
      "$<TARGET_FILE:llvm-objcopy>" "--redefine-syms=${_map}"
      "${_RULE_INPUT}" "${_out}"
    DEPENDS "${_RULE_INPUT}" "${_map}" llvm-objcopy ${_RULE_DEPENDS}
    COMMENT "Renaming llvm/mlir symbols in ${_RULE_NAME}"
  )
  set(${OUT_VAR} "${_out}" PARENT_SCOPE)
endfunction()

# Archives an object library's objects, then rewrites the archive to the
# renamed ABI. Same parameters as iree_renamed_archive with TARGET instead of
# INPUT.
function(iree_renamed_archive_from_objects OUT_VAR)
  cmake_parse_arguments(_RULE "" "NAME;TARGET" "" ${ARGN})
  set(_dir "${CMAKE_CURRENT_BINARY_DIR}/renamed")
  set(_archive "${_dir}/lib${_RULE_NAME}.objects.a")
  add_custom_command(
    OUTPUT "${_archive}"
    COMMAND "$<TARGET_FILE:llvm-ar>" rcs "${_archive}"
            "$<TARGET_OBJECTS:${_RULE_TARGET}>"
    DEPENDS "$<TARGET_OBJECTS:${_RULE_TARGET}>" ${_RULE_TARGET} llvm-ar
    COMMAND_EXPAND_LISTS
    COMMENT "Archiving objects of ${_RULE_TARGET}"
  )
  iree_renamed_archive(_renamed
    NAME "${_RULE_NAME}"
    INPUT "${_archive}"
  )
  set(${OUT_VAR} "${_renamed}" PARENT_SCOPE)
endfunction()

# Walks LINK_LIBRARIES/INTERFACE_LINK_LIBRARIES transitively and partitions
# the closure into:
#   STATIC_LIBS_VAR: non-imported STATIC_LIBRARY targets (to be renamed);
#   OTHER_VAR: everything else, passed through to the link line unchanged
#     (imported targets, raw flags, paths, unhandled generator expressions).
# Imported targets and plain C archives are safe to pass through: the rename
# only affects mangled llvm/mlir C++ names, which those do not define.
function(iree_collect_static_link_closure STATIC_LIBS_VAR OTHER_VAR)
  set(_worklist ${ARGN})
  set(_visited "")
  set(_static_libs "")
  set(_other "")
  while(_worklist)
    list(POP_FRONT _worklist _item)
    if(_item IN_LIST _visited)
      continue()
    endif()
    list(APPEND _visited "${_item}")
    # Directory-scope markers CMake embeds in LINK_LIBRARIES values.
    if(_item MATCHES "^::@")
      continue()
    endif()
    if(_item MATCHES "^\\$<LINK_ONLY:(.+)>$")
      list(APPEND _worklist "${CMAKE_MATCH_1}")
      continue()
    endif()
    # Target-existence conditionals (LLD uses these in its interface link
    # libraries) are decidable now: all closure targets are defined by the
    # time this walk runs.
    if(_item MATCHES "^\\$<TARGET_NAME_IF_EXISTS:([^>]+)>$")
      if(TARGET "${CMAKE_MATCH_1}")
        list(APPEND _worklist "${CMAKE_MATCH_1}")
      endif()
      continue()
    endif()
    if(_item MATCHES "^\\$<IF:\\$<TARGET_EXISTS:([^>]+)>,([^,]*),([^>]*)>$")
      if(TARGET "${CMAKE_MATCH_1}")
        set(_resolved "${CMAKE_MATCH_2}")
      else()
        set(_resolved "${CMAKE_MATCH_3}")
      endif()
      if(_resolved)
        list(APPEND _worklist "${_resolved}")
      endif()
      continue()
    endif()
    # iree_cc_library wires object libraries to their usage requirements via
    # a TARGET_PROPERTY indirection; read the property now instead.
    if(_item MATCHES "^\\$<TARGET_PROPERTY:([^,>]+),([A-Za-z_]+)>$")
      if(TARGET "${CMAKE_MATCH_1}")
        get_target_property(_prop_value "${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}")
        if(_prop_value)
          list(APPEND _worklist ${_prop_value})
        endif()
      endif()
      continue()
    endif()
    # MLIR's CAPI object libraries guard their deps on membership in an
    # aggregate's MLIR_AGGREGATE_EXCLUDE_LIBS. The compiler dylib is not an
    # MLIR aggregate, so the exclusion list is empty and the guard always
    # includes the library.
    if(_item MATCHES "^\\$<\\$<NOT:\\$<IN_LIST:.*MLIR_AGGREGATE_EXCLUDE_LIBS.*>:([A-Za-z0-9_]+)>$")
      list(APPEND _worklist "${CMAKE_MATCH_1}")
      continue()
    endif()
    # LLVM guards platform link flags with $<$<BOOL:probe>:-lfoo>, where the
    # probe is already a literal here (a found library path, or empty when the
    # platform does not apply). Decide it now: an empty condition contributes
    # nothing and is not worth warning about.
    if(_item MATCHES "^\\$<\\$<BOOL:(.*)>:(.*)>$")
      set(_bool_cond "${CMAKE_MATCH_1}")
      set(_bool_value "${CMAKE_MATCH_2}")
      iree_renamed_link_genex_bool("${_bool_cond}" _bool_true)
      if(_bool_true)
        list(APPEND _worklist "${_bool_value}")
      endif()
      continue()
    endif()
    if(_item MATCHES "^\\$<")
      # An unevaluated generator expression on the link line corrupts the
      # generated build files; dropping it surfaces as an undefined-symbol
      # link error instead, which is diagnosable.
      message(WARNING
        "iree_collect_static_link_closure: dropping unsupported generator "
        "expression link item: ${_item}")
      continue()
    endif()
    if(NOT TARGET "${_item}")
      list(APPEND _other "${_item}")
      continue()
    endif()
    get_target_property(_aliased "${_item}" ALIASED_TARGET)
    if(_aliased)
      list(APPEND _worklist "${_aliased}")
      continue()
    endif()
    get_target_property(_imported "${_item}" IMPORTED)
    get_target_property(_type "${_item}" TYPE)
    if(_imported)
      list(APPEND _other "${_item}")
    elseif(_type STREQUAL "STATIC_LIBRARY")
      list(APPEND _static_libs "${_item}")
    elseif(_type STREQUAL "INTERFACE_LIBRARY" OR _type STREQUAL "OBJECT_LIBRARY")
      # Recurse only. Object libraries are the walk's seeds; their objects are
      # archived and whole-archive linked by the caller, so linking the
      # library target itself would duplicate them unrenamed.
    else()
      # Shared libraries, executables: pass through.
      list(APPEND _other "${_item}")
    endif()
    foreach(_prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_deps "${_item}" ${_prop})
      if(_deps)
        list(APPEND _worklist ${_deps})
      endif()
    endforeach()
  endwhile()
  set(${STATIC_LIBS_VAR} "${_static_libs}" PARENT_SCOPE)
  set(${OTHER_VAR} "${_other}" PARENT_SCOPE)
endfunction()

# Assembles the renamed libIREECompiler link. Runs deferred at the end of the
# top-level directory so the transitive closure walk sees every target: the
# API directory is processed before most of the compiler tree, so aliases like
# iree::compiler::Codegen::Common do not exist yet when it is configured (the
# non-renamed assembly sidesteps the same problem with generate-time
# GENEX_EVAL indirection).
#
# Consumes global properties set by compiler/src/iree/compiler/API:
#   IREE_RENAMED_SHARED_IMPL_OBJECT_LIBS: the API-root object libraries.
#   IREE_RENAMED_SHARED_IMPL_BINARY_DIR: directory owning the rename actions.
function(iree_renamed_shared_impl_finalize)
  get_property(_object_libs GLOBAL PROPERTY IREE_RENAMED_SHARED_IMPL_OBJECT_LIBS)
  set(_target iree_compiler_API_SharedImpl)

  # The API-root object libraries become renamed archives and are linked
  # whole-archive: they are the exported surface, nothing references them.
  set(_whole_archives)
  foreach(_object_lib ${_object_libs})
    iree_renamed_link_sanitize_name("${_object_lib}" _base)
    iree_renamed_archive_from_objects(_renamed
      NAME "${_base}"
      TARGET "${_object_lib}"
    )
    list(APPEND _whole_archives "${_renamed}")
  endforeach()

  # Everything the object libraries pull in transitively is renamed too;
  # otherwise the renamed references above would not resolve. Selective
  # archive extraction is preserved (no whole-archive), so the library
  # contains the same objects the unrenamed link would select.
  iree_collect_static_link_closure(_closure_static_libs _closure_other
    ${_object_libs}
    MLIRExportSMTLIB
    MLIRTargetLLVMIRImport
  )
  set(_closure_archives)
  foreach(_lib ${_closure_static_libs})
    iree_renamed_link_sanitize_name("${_lib}" _base)
    iree_renamed_archive(_renamed
      NAME "closure_${_base}"
      INPUT "$<TARGET_FILE:${_lib}>"
      DEPENDS "${_lib}"
    )
    list(APPEND _closure_archives "${_renamed}")
  endforeach()

  if(APPLE)
    set(_whole_archive_flags)
    foreach(_archive ${_whole_archives})
      list(APPEND _whole_archive_flags "-Wl,-force_load,${_archive}")
    endforeach()
    # ld64 resolves archives independent of command-line order.
    target_link_options(${_target} PRIVATE ${_whole_archive_flags})
    target_link_libraries(${_target} PRIVATE
      ${_closure_archives}
      ${_closure_other}
    )
  else()
    # Group the closure so single-pass ELF linkers are insensitive to the
    # discovery order of the archives.
    target_link_libraries(${_target} PRIVATE
      "-Wl,--whole-archive" ${_whole_archives} "-Wl,--no-whole-archive"
      "-Wl,--start-group" ${_closure_archives} "-Wl,--end-group"
      ${_closure_other}
    )
  endif()

  # The archives live in this (deferred, top-level) directory's scope; a
  # target-level dependency orders their generation before the link.
  add_custom_target(${_target}_renamed_archives
    DEPENDS ${_whole_archives} ${_closure_archives}
  )
  add_dependencies(${_target} ${_target}_renamed_archives)
  set_property(TARGET ${_target} APPEND PROPERTY
    LINK_DEPENDS ${_whole_archives} ${_closure_archives}
  )
endfunction()
