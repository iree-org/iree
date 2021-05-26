@ECHO OFF
REM Copyright 2020 The IREE Authors
REM
REM Licensed under the Apache License v2.0 with LLVM Exceptions.
REM See https://llvm.org/LICENSE.txt for license information.
REM SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

SET RUNNER_PATH=%~dp0
powershell.exe -NoProfile -File "%RUNNER_PATH%\run_lit.ps1" %*
EXIT /B %ERRORLEVEL%
