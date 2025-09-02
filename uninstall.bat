@echo off
chcp 65001 > nul
echo ========================================
echo    CL EVA02 ONNX Tagger 卸载脚本
echo ========================================
echo.
echo 警告: 此操作将删除所有项目文件和缓存数据!
echo 包括:
echo   - 虚拟环境 (venv/)
echo   - 模型缓存 (model_cache/)
echo   - 批处理输出 (batch_output/)
echo   - Python 缓存文件
echo.
set /p confirm="确定要卸载吗? (y/N): "

if /i "%confirm%" neq "y" (
    echo 取消卸载操作
    pause
    exit /b 0
)

echo.
echo 开始卸载...

REM 删除虚拟环境
if exist "venv\" (
    echo 删除虚拟环境...
    rmdir /s /q "venv"
    echo 虚拟环境已删除
)

REM 删除模型缓存
if exist "model_cache\" (
    echo 删除模型缓存...
    rmdir /s /q "model_cache"
    echo 模型缓存已删除
)

REM 删除批处理输出
if exist "batch_output\" (
    echo 删除批处理输出...
    rmdir /s /q "batch_output"
    echo 批处理输出已删除
)

REM 删除Python缓存文件
echo 删除Python缓存文件...
del /q *.pyc 2>nul
del /q *.pyo 2>nul
del /q *.pyd 2>nul
for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"

echo.
echo 卸载完成!
echo 项目文件 (app.py, requirements.txt 等) 仍然保留。
echo 如果您想完全删除整个项目，请手动删除项目文件夹。
echo.
pause