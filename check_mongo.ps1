# MongoDB Service Check Script
# Run: .\check_mongo.ps1

Write-Host "`n" + ("=" * 60)
Write-Host "MONGODB SERVICE CHECK"
Write-Host ("=" * 60)

# Check if MongoDB service exists
$service = Get-Service -Name MongoDB -ErrorAction SilentlyContinue

if ($service) {
    Write-Host "`n✅ MongoDB Service Found"
    Write-Host "   Name: $($service.Name)"
    Write-Host "   Status: $($service.Status)"
    Write-Host "   Start Type: $($service.StartType)"
    
    if ($service.Status -eq "Running") {
        Write-Host "`n✅ MongoDB is RUNNING"
        
        # Check if port 27017 is listening
        Write-Host "`nChecking MongoDB port..."
        $port = netstat -ano | Select-String "27017" | Select-Object -First 1
        
        if ($port) {
            Write-Host "✅ MongoDB is listening on port 27017"
            Write-Host "   $port"
        } else {
            Write-Host "⚠️  Port 27017 not found in netstat (might be normal)"
        }
        
        Write-Host "`n" + ("=" * 60)
        Write-Host "🎉 MongoDB is ready! You can start the backend server."
        Write-Host ("=" * 60)
        
    } else {
        Write-Host "`n⚠️  MongoDB service exists but is NOT running"
        Write-Host "`n💡 To start MongoDB:"
        Write-Host "   net start MongoDB"
        Write-Host "   OR: Start-Service MongoDB"
    }
} else {
    Write-Host "`n❌ MongoDB Service NOT Found"
    Write-Host "`n💡 MongoDB might not be installed or not running as a service."
    Write-Host "`nOptions:"
    Write-Host "1. Install MongoDB as Windows Service:"
    Write-Host "   choco install mongodb"
    Write-Host ""
    Write-Host "2. Or run MongoDB manually:"
    Write-Host "   mongod --dbpath C:\data\db"
    Write-Host ""
    Write-Host "3. Or download from:"
    Write-Host "   https://www.mongodb.com/try/download/community"
}

Write-Host "`n" + ("=" * 60)
Write-Host "CONNECTION STRING"
Write-Host ("=" * 60)
Write-Host "mongodb://localhost:27017/plantAI"
Write-Host ("=" * 60)
