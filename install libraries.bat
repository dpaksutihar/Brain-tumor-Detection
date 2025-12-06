echo ===========================================
echo   Installing Python and Node.js Packages
echo ===========================================

REM --- Python packages ---
echo Installing Python packages...
pip install tensorflow==2.18.1
pip install numpy==1.26.4
pip install pandas==2.2.2
pip install scikit-learn==1.5.2
pip install opencv-python==4.10.0.84
pip install flask==3.0.3
pip install flask-cors==5.0.0


echo -------------------------------------------
echo Python packages installed.
echo -------------------------------------------

REM --- Node packages ---
echo Installing Node.js packages...
npm install
npm install cors
npm install concurrently
npm install axios

echo -------------------------------------------
echo Node packages installed.
echo -------------------------------------------

echo All installations completed successfully!

PAUSE