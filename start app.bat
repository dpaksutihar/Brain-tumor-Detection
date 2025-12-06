REM ===============================
REM Start Flask API and React app
REM ===============================

REM Start Flask backend
echo Starting Flask backend...
start cmd /k "cd /d mri-app && python PythonServer.py"

REM Wait a few seconds to ensure Flask starts
timeout /t 5 /nobreak

REM Start React frontend
echo Starting React frontend...
start cmd /k "cd /d mri-app && npm start"


REM Done
echo Both backend and frontend are running.

