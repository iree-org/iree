/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2011 Cisco Systems, Inc.  All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * This file should be included by any file that needs full
 * version information for the OMPI project
 */

#ifndef OMPI_VERSIONS_H
#define OMPI_VERSIONS_H

#define OMPI_MAJOR_VERSION 4
#define OMPI_MINOR_VERSION 1
#define OMPI_RELEASE_VERSION 4
#define OMPI_GREEK_VERSION ""
#define OMPI_WANT_REPO_REV @OMPI_WANT_REPO_REV@
#define OMPI_REPO_REV "v4.1.4"
#ifdef OMPI_VERSION
/* If we included version.h, we want the real version, not the
   stripped (no-r number) version */
#undef OMPI_VERSION
#endif
#define OMPI_VERSION "4.1.4"

#endif
