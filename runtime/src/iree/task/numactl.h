/* Copyright (C) 2003,2004,2005 Andi Kleen, SuSE Labs.
   Command line NUMA policy control.

   numactl is free software; you can redistribute it and/or
   modify it under the terms of the GNU General Public
   License as published by the Free Software Foundation; version
   2.

   numactl is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should find a copy of v2 of the GNU General Public License somewhere
   on your Linux system; if not, write to the Free Software Foundation,
   Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA */
#ifndef IREE_TASK_NUMACTL_H_
#define IREE_TASK_NUMACTL_H_

#define _GNU_SOURCE
#include <getopt.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdarg.h>
#include <ctype.h>
#include "third_party/numactl/numa.h"
#include "third_party/numactl/numaif.h"
#include "third_party/numactl/numaint.h"
#include "third_party/numactl/util.h"
#include "third_party/numactl/shm.h"

#define CPUSET 0
#define ALL 1

int exitcode;

char *fmt_mem(unsigned long long mem, char *buf);
void print_distances(int maxnode);
void print_node_cpus(int node);
void hardware_test(void);

#endif  // IREE_TASK_NUMACTL_H_