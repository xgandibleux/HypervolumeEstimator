@echo off
rem =============================================================================
rem Build script for the fpli_hv shared library (Windows, MinGW-w64)
rem
rem Requires MinGW-w64 with gcc available in PATH.
rem Download from: https://www.mingw-w64.org/
rem
rem Compiles hv.c and avl.c from the hv-1.3-src distribution into:
rem   src\fpli_hv.dll
rem
rem Run once from the project root before launching mainExp1_UKP.jl or
rem mainExp1_UFLP.jl:
rem   build_lib.bat
rem =============================================================================

set SRC=src HV\hv-1.3-src
set OUT=src\fpli_hv.dll

gcc -O3 -DVARIANT=4 -DDEBUG=0 ^
    -shared ^
    "%SRC%\hv.c" "%SRC%\avl.c" ^
    -o "%OUT%"

if %ERRORLEVEL% == 0 (
    echo Built: %OUT%
) else (
    echo ERROR: compilation failed. Check that gcc ^(MinGW-w64^) is in PATH.
    exit /b 1
)
