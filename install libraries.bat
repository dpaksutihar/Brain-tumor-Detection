@echo off
echo ===========================================
echo   Installing Python and Node.js Packages
echo ===========================================

REM --- Python packages ---
echo Installing Python packages...
pip install tensorflow==2.16.1
pip install flask==2.3.3
pip install flask-cors==3.1.13
pip install scikit-learn==1.3.0
pip install pandas==2.1.1
pip install numpy==1.26.0
pip install opencv-python==4.8.1.78

echo -------------------------------------------
echo Python packages installed.
echo -------------------------------------------

REM --- Node packages ---
echo Installing Node.js packages...
npm install cors
npm install concurrently
npm install axios

echo -------------------------------------------
echo Node packages installed.
echo -------------------------------------------

echo All installations completed successfully!

REM Keep the output window open and printing
cmd /k
