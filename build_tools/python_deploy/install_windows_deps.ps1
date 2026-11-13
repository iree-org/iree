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
)

$PYTHON_VERSIONS_NO_DOT = @(
  "313" #,
  "312" #,
  "311" #,
)

# These can be discovered at https://www.python.org/downloads/windows/
$PYTHON_INSTALLER_URLS = @(
  "https://www.python.org/ftp/python/3.13.1/python-3.13.1-amd64.exe" #,
  "https://www.python.org/ftp/python/3.12.8/python-3.12.8-amd64.exe" #,
  "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" #,
)

# SHA-256 of each installer in $PYTHON_INSTALLER_URLS.
$PYTHON_INSTALLER_SHA256S = @(
  "6b33fa9a439a86f553f9f60e538ccabc857d2f308bc77c477c04a46552ade81f" #,
  "71bd44e6b0e91c17558963557e4cdb80b483de9b0a0a9717f06cf896f95ab598" #,
  "5ee42c4eee1e6b4464bb23722f90b45303f79442df63083f05322f1785f5fdde" #,
)

# The python.org installers are Authenticode-signed by this subject.
$PYTHON_SIGNER = "CN=Python Software Foundation, O=Python Software Foundation, *"

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
  $PYTHON_INSTALLER_SHA256 = $PYTHON_INSTALLER_SHA256S[$i]
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

    $DOWNLOAD_SHA256 = (Get-FileHash -Algorithm SHA256 $DOWNLOAD_PATH).Hash
    if ($DOWNLOAD_SHA256 -ne $PYTHON_INSTALLER_SHA256) {
      throw "SHA-256 of $DOWNLOAD_PATH is $DOWNLOAD_SHA256, expected $PYTHON_INSTALLER_SHA256"
    }
    $SIGNATURE = Get-AuthenticodeSignature $DOWNLOAD_PATH
    if ($SIGNATURE.Status -ne "Valid" -or $SIGNATURE.SignerCertificate.Subject -notlike $PYTHON_SIGNER) {
      throw "$DOWNLOAD_PATH is not signed by the Python Software Foundation: $($SIGNATURE.Status), $($SIGNATURE.SignerCertificate.Subject)"
    }

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
