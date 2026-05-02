REM Author: Zhaojjin Fu
REM Date: 2026-04-30
REM CMENet Project - Prediction Script

@echo off
setlocal enabledelayedexpansion

REM CMENet dataset prediction for Windows CMD.
REM Usage:
REM   predict_dataset.bat dataset_dir [name_pattern] [output_dir] [weight_path]
REM dataset_dir can be either:
REM   1. data\c          (contains image\T1, image\T1C, image\TOF)
REM   2. data\c\image    (contains T1, T1C, TOF directly)
REM name_pattern is optional. Default: auto.
REM name_pattern:
REM   auto   : automatically detect image / T1 / direct
REM   image  : T1\xxx_image.nii, T1C\xxx_image.nii, TOF\xxx_image.nii
REM   T1     : T1\xxx_T1.nii,    T1C\xxx_T1C.nii,   TOF\xxx_TOF.nii
REM   direct : T1\xxxT1.nii,     T1C\xxxT1C.nii,    TOF\xxxTOF.nii

cd /d "%~dp0"

set "DATASET_DIR=%~1"
set "NAME_PATTERN=%~2"
set "OUTPUT_DIR=%~3"
set "WEIGHT=%~4"
set "TXT_FILE="
set "INTERACTIVE=0"

if "%OUTPUT_DIR%"=="" set "OUTPUT_DIR=.\batch_output"
if "%WEIGHT%"=="" set "WEIGHT=results\A compare_result\CMENet_NoK_caro_network4_NK_seg\checkpoints\best-epoch1319-loss0.8286.pth"

set "DEVICE=cuda:0"
set "NORM_DATASET=caro"
set "THRESHOLD=0.5"

if "%DATASET_DIR%"=="" (
    set "INTERACTIVE=1"
    echo ========================================
    echo   CMENet dataset prediction
    echo ========================================
    echo Supported dataset dir examples:
    echo   data\c        contains image\T1, image\T1C, image\TOF
    echo   data\c\image  contains T1, T1C, TOF directly
    echo name_pattern options: auto / image / T1 / direct
    echo Default is auto. For data\c, auto will detect image pattern.
    set /p "DATASET_DIR=Input dataset root dir [default: data\c]: "
    if "!DATASET_DIR!"=="" set "DATASET_DIR=data\c"
)

if "%NAME_PATTERN%"=="" (
    set /p "NAME_PATTERN=Input name_pattern auto / image / T1 / direct [default: auto]: "
    if "!NAME_PATTERN!"=="" set "NAME_PATTERN=auto"
)

if "%INTERACTIVE%"=="1" (
    set /p "INPUT_OUTPUT=Output dir [default: !OUTPUT_DIR!]: "
    if not "!INPUT_OUTPUT!"=="" set "OUTPUT_DIR=!INPUT_OUTPUT!"
    set /p "INPUT_WEIGHT=Weight path [default: !WEIGHT!]: "
    if not "!INPUT_WEIGHT!"=="" set "WEIGHT=!INPUT_WEIGHT!"
)

echo.
echo ========================================
echo   Start dataset prediction
echo ========================================
echo   Dataset      : %DATASET_DIR%
echo   Name pattern : %NAME_PATTERN%
echo   Output       : %OUTPUT_DIR%
echo   Weight       : %WEIGHT%
echo ========================================
echo.

python predict_dataset.py ^
    --dataset_dir "%DATASET_DIR%" ^
    --name_pattern "%NAME_PATTERN%" ^
    --output "%OUTPUT_DIR%" ^
    --weight "%WEIGHT%" ^
    --device "%DEVICE%" ^
    --norm_dataset "%NORM_DATASET%" ^
    --threshold "%THRESHOLD%"

echo.
echo [DONE] Dataset prediction finished. Output: %OUTPUT_DIR%
pause
