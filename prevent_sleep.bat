@echo off
echo Configuring power settings to prevent sleep...

REM Never sleep on AC power
powercfg /change standby-timeout-ac 0

REM Never turn off display on AC power  
powercfg /change monitor-timeout-ac 0

REM Never hibernate
powercfg /change hibernate-timeout-ac 0

echo.
echo [OK] Power settings updated!
echo     - Screen: Never turn off
echo     - Sleep:  Never
echo     - Hibernate: Never
echo.
echo Your laptop will stay awake when plugged in.
echo.
pause
