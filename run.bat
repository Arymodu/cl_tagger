@echo off
chcp 65001 > nul
echo ========================================
echo    CL EVA02 ONNX Tagger 设置脚本
echo ========================================
echo.

REM 检查Python是否已安装
python --version >nul 2>&1
if errorlevel 1 (
    echo 未检测到Python，请先安装Python 3.8或更高版本
    echo 可以从 https://www.python.org/downloads/ 下载
    pause
    exit /b 1
)

REM 显示Python版本
echo 检测到Python版本:
python --version
echo.

REM 检查虚拟环境是否存在，如果不存在则创建
if not exist "venv\" (
    echo 创建虚拟环境...
    python -m venv venv
    echo 虚拟环境创建完成
) else (
    echo 虚拟环境已存在
)
echo.

REM 激活虚拟环境
echo 激活虚拟环境...
call venv\Scripts\activate.bat
echo.

REM 安装依赖
echo 安装依赖包...
pip install --upgrade pip
pip install -r requirements.txt
echo.

REM 检查是否需要安装onnxruntime-gpu
echo 检查GPU支持...
python -c "import onnxruntime as ort; print('可用提供程序:', ort.get_available_providers())"
echo.

REM 启动应用
echo 启动应用...
echo 应用将在 http://localhost:7870 上运行
echo 按 Ctrl+C 停止服务
echo.
python app.py

pause