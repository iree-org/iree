# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT CMAKE_CROSSCOMPILING)
  add_executable(generate_embed_data)
  target_sources(generate_embed_data PRIVATE generate_embed_data_main.cc)
  set_target_properties(generate_embed_data PROPERTIES OUTPUT_NAME generate_embed_data)

  target_link_libraries(generate_embed_data
    absl::flags
    absl::flags_parse
    absl::strings
  )

  install(TARGETS generate_embed_data
          COMPONENT generate_embed_data
          RUNTIME DESTINATION bin)
endif()
