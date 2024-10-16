# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Installs dependencies on Windows necessary to build IREE Python wheels.

$PYTHON_VERSIONS = @(
  "3.13" #,
  "3.12" #,
  "3.11" #,
  # "3.10",
  # "3.9"
)

$PYTHON_VERSIONS_NO_DOT = @(
  "313" #,
  "312" #,
  "311" #,
  # "310",
  # "39"
)

# These can be discovered at https://www.python.org/downloads/windows/
$PYTHON_INSTALLER_URLS = @(
  "https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe" #,
  "https://www.python.org/ftp/python/3.12.6/python-3.12.6-amd64.exe" #,
  "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" #,
  # "https://www.python.org/ftp/python/3.10.5/python-3.10.5-amd64.exe",
  # "https://www.python.org/ftp/python/3.9.13/python-3.9.13-amd64.exe"
)

# Multiple Python install locations are valid, so we use the `py` helper to
# check for versions and call into them. Some valid install locations are:
#   C:\Python39\python.exe
#   C:\Program Files\Python39\python.exe
#   C:\Users\[NAME]\AppData\Local\Programs\Python\Python39\python.exe
# See https://docs.python.org/3/using/windows.html#python-launcher-for-windows.
$INSTALLED_VERSIONS_OUTPUT = py --list | Out-String

for($i=0 ; $i -lt $PYTHON_VERSIONS.Length; $i++) {
  $PYTHON_VERSION = $PYTHON_VERSIONS[$i]
  $PYTHON_VERSION_NO_DOT = $PYTHON_VERSIONS_NO_DOT[$i]
  $PYTHON_INSTALLER_URL = $PYTHON_INSTALLER_URLS[$i]
  Write-Host "-- Installing Python ${PYTHON_VERSION} from ${PYTHON_INSTALLER_URL}"

  if ("${INSTALLED_VERSIONS_OUTPUT}" -like "*${PYTHON_VERSION}*") {
    Write-Host "::  Python version already installed. Not reinstalling."
  } else {
    $DOWNLOAD_ROOT = "$env:TEMP/iree_python_install"
    $DOWNLOAD_FILENAME = $PYTHON_INSTALLER_URL.Substring($PYTHON_INSTALLER_URL.LastIndexOf("/") + 1)
    $DOWNLOAD_PATH = "${DOWNLOAD_ROOT}/$DOWNLOAD_FILENAME"

    # Create download folder as needed.
    md -Force ${DOWNLOAD_ROOT} | Out-Null

    Write-Host "::  Downloading $PYTHON_INSTALLER_URL -> $DOWNLOAD_PATH"
    curl $PYTHON_INSTALLER_URL -o $DOWNLOAD_PATH

    Write-Host "::  Running installer: $DOWNLOAD_PATH"
    # https://docs.python.org/3/using/windows.html#installing-without-ui
    & "$DOWNLOAD_PATH" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
  }

  Write-Host "::  Python version $PYTHON_VERSION installed:"
  & py -${PYTHON_VERSION} --version
  & py -${PYTHON_VERSION} -m pip --version

  Write-Host "::  Installing system pip packages"
  & py -${PYTHON_VERSION} -m pip install --upgrade pip
}

Write-Host "*** All done ***"
