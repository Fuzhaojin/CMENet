REM Author: Zhaojjin Fu
REM Date: 2026-04-30
REM CMENet Project - Prediction Script

@echo off
setlocal enabledelayedexpansion

REM CMENet single-case prediction for Windows CMD.
REM Usage:
REM   predict.bat T1_path T1C_path TOF_path [output_dir] [weight_path]
REM Or run predict.bat without arguments and enter paths interactively.

cd /d "%~dp0"

set "T1=%~1"
set "T1C=%~2"
set "TOF=%~3"
set "OUTPUT_DIR=%~4"
set "WEIGHT=%~5"

if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=.\pred_output"
if "%WEIGHT%"=="" set "WEIGHT=results\A compare_result\CMENet_NoK_caro_network4_NK_seg\checkpoints\best-epoch1319-loss0.8286.pth"

set "DEVICE=cuda:0"
set "NORM_DATASET=caro"
set "THRESHOLD=0.5"

if "%T1%"=="" (
    echo ========================================
    echo   CMENet single-case prediction
    echo ========================================
    set /p "T1=Input T1 file path: "
    set /p "T1C=Input T1C file path: "
    set /p "TOF=Input TOF file path: "
    set /p "INPUT_OUTPUT=Output dir [default: !OUTPUT_DIR!]: "
    if not "!INPUT_OUTPUT!"=="" set "OUTPUT_DIR=!INPUT_OUTPUT!"
    set /p "INPUT_WEIGHT=Weight path [default: !WEIGHT!]: "
    if not "!INPUT_WEIGHT!"=="" set "WEIGHT=!INPUT_WEIGHT!"
)

echo.
echo ========================================
echo   Start prediction
echo ========================================
echo   T1     : %T1%
echo   T1C    : %T1C%
echo   TOF    : %TOF%
echo   Output : %OUTPUT_DIR%
echo   Weight : %WEIGHT%
echo ========================================
echo.

python predict.py ^
    --t1 "%T1%" ^
    --t1c "%T1C%" ^
    --tof "%TOF%" ^
    --output "%OUTPUT_DIR%" ^
    --weight "%WEIGHT%" ^
    --device "%DEVICE%" ^
    --norm_dataset "%NORM_DATASET%" ^
    --threshold "%THRESHOLD%"

echo.
echo [DONE] Prediction finished. Output: %OUTPUT_DIR%
pause
