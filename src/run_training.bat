
@echo off
chcp 65001 >nul
echo ========================================
echo   DIFFUSION MODEL TRAINING
echo ========================================
echo.
echo Activation de l'environnement swimming_pool...
call conda activate swimming_pool

echo.
echo Lancement de l'entrainement...
echo.
python train.py

echo.
echo ========================================
echo   ENTRAINEMENT TERMINE
echo ========================================
pause
