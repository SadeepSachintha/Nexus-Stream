@echo off
setlocal enabledelayedexpansion
title Nexus Stream - Real-Time Financial Fraud Detection Engine
set "PYTHONPATH=%~dp0"

:: Locate Virtual Environment
set "PYTHON=.venv\Scripts\python.exe"
set "STREAMLIT=.venv\Scripts\streamlit.exe"
set "PYTEST=.venv\Scripts\pytest.exe"

if not exist "%PYTHON%" (
    echo [!] Virtual environment not found at .venv\Scripts\python.exe
    echo [*] Checking system python...
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON=python"
        set "STREAMLIT=streamlit"
        set "PYTEST=pytest"
    ) else (
        echo [ERROR] Python was not found in PATH or .venv. Please install Python 3.10+.
        pause
        exit /b 1
    )
)

:: Check CLI Arguments
set "IS_CLI_MODE=0"
if not "%~1"=="" (
    set "IS_CLI_MODE=1"
    if /i "%~1"=="api" goto start_api
    if /i "%~1"=="dashboard" goto start_dashboard
    if /i "%~1"=="simulate" goto start_simulate
    if /i "%~1"=="processor" goto start_processor
    if /i "%~1"=="train" goto start_train
    if /i "%~1"=="test" goto start_test
    if /i "%~1"=="benchmark" goto start_benchmark
    if /i "%~1"=="load-test" goto start_load_test
    if /i "%~1"=="load-ui" goto start_load_ui
    if /i "%~1"=="docker-up" goto start_docker_up
    if /i "%~1"=="docker-down" goto start_docker_down
    echo [!] Unknown argument '%~1'. Available: api, dashboard, simulate, processor, train, test, benchmark, load-test, load-ui, docker-up, docker-down
    exit /b 1
)

:menu
cls
echo ========================================================================
echo   [NEXUS STREAM] Real-Time Financial Fraud Detection Platform
echo ========================================================================
echo.
echo   CORE SERVICES:
echo     [1] Start FastAPI Inference Gateway          (http://localhost:8000)
echo     [2] Launch Streamlit Dispute ^& Ops Portal    (http://localhost:8501)
echo     [3] Run Synthetic Transaction Simulator      (50 tx/s -^> API)
echo     [4] Run Standalone Stream Processor          (Local In-Memory Mode)
echo.
echo   MACHINE LEARNING ^& TESTING:
echo     [5] Train Cost-Sensitive XGBoost Model       (PR-AUC ^& Baseline Export)
echo     [6] Run Full Automated Pytest Suite         (27 Unit/Integration Tests)
echo     [7] Run Concurrency Benchmark CLI            (Sub-15ms SLA Verification)
echo     [8] Run Headless Locust Load Test            (30 Concurrent Users)
echo     [9] Launch Locust Load Testing Web UI        (http://localhost:8089)
echo.
echo   CONTAINER CLUSTER (DOCKER):
echo     [10] Launch Full Distributed Cluster         (Redpanda, Flink, Grafana)
echo     [11] Stop Distributed Cluster
echo.
echo     [0] Exit
echo ========================================================================
set /p choice="  Enter option [0-11]: "

if "%choice%"=="1" goto start_api
if "%choice%"=="2" goto start_dashboard
if "%choice%"=="3" goto start_simulate
if "%choice%"=="4" goto start_processor
if "%choice%"=="5" goto start_train
if "%choice%"=="6" goto start_test
if "%choice%"=="7" goto start_benchmark
if "%choice%"=="8" goto start_load_test
if "%choice%"=="9" goto start_load_ui
if "%choice%"=="10" goto start_docker_up
if "%choice%"=="11" goto start_docker_down
if "%choice%"=="0" goto exit_script

echo.
echo [!] Invalid selection. Please choose an option from 0 to 11.
ping 127.0.0.1 -n 2 >nul
goto menu

:start_api
echo.
echo [*] Starting Nexus Stream FastAPI Gateway on http://localhost:8000...
echo [*] Base Gateway Landing:     http://localhost:8000
echo [*] Interactive Swagger Docs:  http://localhost:8000/docs
echo [*] ReDoc Specification:      http://localhost:8000/redoc
echo [*] Prometheus Metrics:        http://localhost:8000/metrics
echo.
"%PYTHON%" -m uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 --reload
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
goto end

:start_dashboard
echo.
echo [*] Launching Streamlit Dispute Operations Portal on http://localhost:8501...
echo.
"%STREAMLIT%" run src/dashboard/app.py --server.port 8501
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
goto end

:start_simulate
echo.
echo [*] Launching Synthetic Financial Transaction Generator (50 tx/sec)...
echo.
"%PYTHON%" -m src.simulator.generate_transactions --rate 50 --target api
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
goto end

:start_processor
echo.
echo [*] Starting Stream Feature Enrichment Processor in local mode...
echo.
"%PYTHON%" -m src.streaming.processor --local
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
goto end

:start_train
echo.
echo [*] Synthesizing 50,000 transactions and training cost-sensitive XGBoost model...
echo.
"%PYTHON%" -m src.inference.train
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
echo.
pause
goto menu

:start_test
echo.
echo [*] Running full automated pytest test suite...
echo.
"%PYTEST%" tests/ -v
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
echo.
pause
goto menu

:start_benchmark
echo.
echo [*] Running high-throughput concurrency benchmark CLI...
echo.
"%PYTHON%" -m tests.load.benchmark_cli --requests 200 --concurrency 10
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
echo.
pause
goto menu

:start_load_test
echo.
echo [*] Running headless Locust load test (30 concurrent users)...
echo.
"%PYTHON%" -m locust -f tests/load/locustfile.py --headless -u 30 -r 10 --run-time 20s --host http://localhost:8000
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
echo.
pause
goto menu

:start_load_ui
echo.
echo [*] Launching Locust Web UI on http://localhost:8089...
echo.
"%PYTHON%" -m locust -f tests/load/locustfile.py --host http://localhost:8000 --web-port 8089
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
goto end

:start_docker_up
echo.
echo [*] Starting full multi-container Docker cluster...
echo.
docker compose up -d
echo.
docker compose ps
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
echo.
pause
goto menu

:start_docker_down
echo.
echo [*] Stopping and tearing down Docker cluster...
echo.
docker compose down -v
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
echo.
pause
goto menu

:end
if "%IS_CLI_MODE%"=="1" exit /b %errorlevel%
pause
goto menu

:exit_script
echo.
echo [*] Exiting Nexus Stream. Goodbye!
exit /b 0
