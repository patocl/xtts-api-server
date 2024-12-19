# Enable strict error handling
$ErrorActionPreference = "Stop"

# Define color variables
$green = [ConsoleColor]::Green
$cyan = [ConsoleColor]::Cyan
$yellow = [ConsoleColor]::Yellow
$red = [ConsoleColor]::Red

function Write-ColoredText {
    param (
        [string]$Message,
        [ConsoleColor]$Color
    )
    $currentColor = $Host.UI.RawUI.ForegroundColor
    $Host.UI.RawUI.ForegroundColor = $Color
    Write-Host $Message
    $Host.UI.RawUI.ForegroundColor = $currentColor
}

try {
    # Build Docker Compose services
    Write-ColoredText "Building Docker Compose services..." $cyan
    docker compose build
    Write-ColoredText "Build completed successfully!" $green

    # Check if containers already exist
    $containers = docker compose ps -q
    if ($containers) {
        Write-ColoredText "Stopping and removing existing containers..." $yellow
        docker compose down
    }

    # Start Docker Compose services in detached mode
    Write-ColoredText "Starting Docker Compose services in detached mode..." $cyan
    docker compose up -d
    Write-ColoredText "Services are up and running!" $green
} catch {
    # Handle any errors that occur during the build or startup
    Write-ColoredText "An error occurred: $($_.Exception.Message)" $red
    Write-ColoredText "Aborting. Please check the logs above for details." $red
    exit 1
}
