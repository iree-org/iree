/* -*- c -*-
 *
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2013      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2016      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * Function: - OS, CPU and compiler dependent configuration
 */

#ifndef OSHMEM_CONFIG_H
#define OSHMEM_CONFIG_H

/* Need to include a bunch of infrastructure from the OMPI layer */
#include "ompi_config.h"

#define OSHMEM_IDENT_STRING OPAL_IDENT_STRING

#if defined(__WINDOWS__)

#  if defined(_USRDLL)    /* building shared libraries (.DLL) */
#    if defined(OSHMEM_EXPORTS)
#      define OSHMEM_DECLSPEC        __declspec(dllexport)
#      define OSHMEM_MODULE_DECLSPEC
#    else
#      define OSHMEM_DECLSPEC        __declspec(dllimport)
#      if defined(OSHMEM_MODULE_EXPORTS)
#        define OSHMEM_MODULE_DECLSPEC __declspec(dllexport)
#      else
#        define OSHMEM_MODULE_DECLSPEC __declspec(dllimport)
#      endif  /* defined(OSHMEM_MODULE_EXPORTS) */
#    endif  /* defined(OSHMEM_EXPORTS) */
#  else          /* building static library */
#    if defined(OSHMEM_IMPORTS)
#      define OSHMEM_DECLSPEC        __declspec(dllimport)
#    else
#      define OSHMEM_DECLSPEC
#    endif  /* defined(OSHMEM_IMPORTS) */
#    define OSHMEM_MODULE_DECLSPEC
#  endif  /* defined(_USRDLL) */

#else

#  if OPAL_C_HAVE_VISIBILITY
#    ifndef OSHMEM_DECLSPEC
#      define OSHMEM_DECLSPEC            __opal_attribute_visibility__("default")
#    endif
#    ifndef OSHMEM_MODULE_DECLSPEC
#      define OSHMEM_MODULE_DECLSPEC     __opal_attribute_visibility__("default")
#    endif
#  else
#    ifndef OSHMEM_DECLSPEC
#      define OSHMEM_DECLSPEC
#    endif
#    ifndef OSHMEM_MODULE_DECLSPEC
#      define OSHMEM_MODULE_DECLSPEC
#    endif
#  endif
#endif  /* defined(__WINDOWS__) */

#endif
