/*
 * Copyright (c) 2013      Mellanox Technologies, Inc.
 *                         All rights reserved.
 * Copyright (c) 2015 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 */
#ifndef SHMEM_API_LOGGER_H
#define SHMEM_API_LOGGER_H

#include "oshmem_config.h"
#include "opal/util/output.h"

#include "oshmem/util/oshmem_util.h"

OSHMEM_DECLSPEC extern int shmem_api_logger_output;

#ifdef __BASE_FILE__
#define __SPML_FILE__ __BASE_FILE__
#else
#define __SPML_FILE__ __FILE__
#endif

#ifdef OPAL_ENABLE_DEBUG
#define SHMEM_API_VERBOSE(level, ...) \
    oshmem_output_verbose(level, shmem_api_logger_output, \
        "%s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)
#else
#define SHMEM_API_VERBOSE(level, ...)
#endif

#define SHMEM_API_ERROR(...) \
    oshmem_output(shmem_api_logger_output, \
        "Error: %s:%d - %s()", __SPML_FILE__, __LINE__, __func__, __VA_ARGS__)

#endif /*SHMEM_API_LOGGER_H*/
