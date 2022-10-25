#Write-Host "Installing python"

#Start-Process winget install Python.Python.3.10 '/quiet InstallAllUsers=1 PrependPath=1' -wait -NoNewWindow

#Write-Host "python installation completed successfully"

#Write-Host "Reload environment variables"
#$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
#Write-Host "Reloaded environment variables"

Write-Host "Installing Build Dependencies"
python -m venv .\iree_venv\
.\iree_venv\Scripts\activate
pip install -r runtime\bindings\python\iree\runtime\build_requirements.txt
Write-Host "Build Deps installation completed successfully" 

Write-Host "Building ..." 
$env:CMAKE_GENERATOR='Ninja'
$env:IREE_HAL_DRIVER_CUDA = 'ON'
$env:IREE_HAL_DRIVER_VULKAN = 'ON'
$env:IREE_EXTERNAL_HAL_DRIVERS = 'OFF'
pip wheel -v -w ../bindist compiler/
pip wheel -v -w ../bindist runtime/

Write-Host "Build completed successfully" 
