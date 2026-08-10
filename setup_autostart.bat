@echo off
:: Right-click this file and choose "Run as administrator"
echo Setting up India Trader auto-start at 9:00 AM IST daily...
echo.

powershell -Command "$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument '/c \"\"C:\Users\soura\Desktop\India trader\start.bat\"\"'; $trigger = New-ScheduledTaskTrigger -Daily -At '09:00AM'; $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun; Register-ScheduledTask -TaskName 'IndiaTrader_9AM' -Action $action -Trigger $trigger -Settings $settings -Force -RunLevel Highest"

if %ERRORLEVEL%==0 (
    echo.
    echo  SUCCESS! India Trader will now open every day at 9:00 AM automatically.
    echo  Task name: IndiaTrader_9AM
    echo  You can view/edit it in Task Scheduler ^(search in Start Menu^).
) else (
    echo.
    echo  FAILED. Make sure you right-clicked and chose "Run as administrator".
)
echo.
pause
