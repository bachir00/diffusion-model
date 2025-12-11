@echo off
chcp 65001 >nul
echo ========================================
echo   DIFFUSION MODEL TRAINING
echo ========================================
echo.
echo Activation de l'environnement swimming_pool...
call conda activate swimming_pool

echo.
echo Reprise de l'entrainement...
echo.
python resume_training.py

echo.
echo ========================================
echo   REPRISE TERMINE
echo ========================================
pause
