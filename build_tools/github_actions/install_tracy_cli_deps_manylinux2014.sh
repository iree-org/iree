#!/bin/bash
# Installs deps on a manylinux2014 CentOS docker container needed for
# building Tracy CLI capture tool.

set -e

td="$(cd $(dirname $0) && pwd)"
yum -y install capstone-devel libzstd-devel
$td/install_tbb_manylinux2014.sh
