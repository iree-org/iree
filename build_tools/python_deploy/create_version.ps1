param(
    [Parameter(Mandatory=$True, Position=1, ValueFromPipeline=$false)]
    [System.String]
    $version
)

Write-Host $version

# Create version info object
$verinfoprop = @{
    'package-suffix'= ""
    'package-version'= $version
    'iree-revision'= $(git rev-parse HEAD)
}

$info = New-Object -TypeName PSObject -Prop $verinfoprop

# Convert info to JSON
$info = $info | ConvertTo-JSON

# Output to JSON file
$info | Out-File "version_info.json" -Encoding "ASCII"
