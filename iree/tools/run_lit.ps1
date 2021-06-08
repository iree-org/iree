# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

param(
  [Parameter(Position=0, Mandatory)]
  [ValidateNotNullOrEmpty()]
  [string]
    $test_file,
  [Parameter(Position=1, ValueFromRemainingArguments=$true)]
  [string[]]
    $test_data = @()
)

# NOTE: to debug first run `$DebugPreference = 'Continue'` in your shell.
# $DebugPreference = "Continue"

trap {
  Write-Error $_
  exit 1
}

# Search the system path for a suitable bash.exe.
# Note that C:\Windows\system32\bash.exe is actually WSL -- which will not
# work. Why???
$pathFolders = $env:Path.Split(";") 
foreach ($_ in $pathFolders) {
  if (-not ($_ -like "*:\Windows\*")) {
    Write-Debug "Checking for bash.exe in: $_"
    $possibleBashExe = "$_\bash.exe"
    if (Test-Path $possibleBashExe -PathType leaf) {
      $bashExe = $possibleBashExe
      break
    }
  }
}

if (-not $bashExe) {
  Write-Host -ForegroundColor Red "Could not find bash.exe on path (excluding \Windows\system32)"
  $pathFolders -join "`r`n" |  Write-Host -ForegroundColor Red
  exit 1
}
Write-Debug "Using bash.exe: $bashExe"

# Get all of the directories we'll want to put on our path for the test.
$test_dirs = [System.Collections.ArrayList]@()
foreach ($test_path in $test_data) {
  $test_dir = Split-Path -Path $test_path -Parent
  $test_dirs.Add($test_dir) | Out-Null
}
Write-Debug "Test data directories: $test_dirs"
$test_dirs.AddRange($env:Path.Split(";"))
$env:Path = $test_dirs -join ";"
Write-Debug "Test PATH:"
Write-Debug "$env:PATH"

$test_lines = Get-Content -Path $test_file
foreach ($test_line in $test_lines) {
  if (!$test_line.StartsWith("// RUN:")) {
    continue
  }
  $test_line = $test_line.Substring("// RUN: ".Length)
  $test_line = $test_line -replace "%s", $test_file
  Write-Host -ForegroundColor Blue "Running test command:"
  Write-Host -ForegroundColor Yellow "$test_line"
  & $bashExe -c $test_line | Out-Default
  if ($LASTEXITCODE -gt 0) {
    Write-Host -ForegroundColor Red "Test failed with $LASTEXITCODE, command:"
    Write-Host -ForegroundColor Yellow "$test_line"
    exit $LASTEXITCODE
  }
}

Write-Debug "All run commands completed successfully"
exit 0
