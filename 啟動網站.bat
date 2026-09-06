@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo [1/4] 檢查 Python 虛擬環境...
if exist ".venv\Scripts\python.exe" goto install_dependencies

where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv .venv
) else (
    where python >nul 2>nul
    if errorlevel 1 goto python_not_found
    python -m venv .venv
)

if not exist ".venv\Scripts\python.exe" goto setup_failed

:install_dependencies
echo [2/4] 檢查並安裝必要套件...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto dependency_failed

echo [3/4] 檢查環境設定...
if not exist ".env" copy /Y ".env.example" ".env" >nul

echo [4/4] 啟動網站...
echo 網址：http://127.0.0.1:7860
echo 關閉此視窗或按 Ctrl+C 可停止服務。
".venv\Scripts\python.exe" src\app.py
if errorlevel 1 goto app_failed
goto end

:python_not_found
echo.
echo [錯誤] 找不到 Python。請先安裝 Python 3.11 以上版本，並勾選 Add Python to PATH。
goto failed

:setup_failed
echo.
echo [錯誤] 無法建立 .venv 虛擬環境。
goto failed

:dependency_failed
echo.
echo [錯誤] 套件安裝失敗，請檢查網路連線與上方錯誤訊息。
goto failed

:app_failed
echo.
echo [錯誤] 網站無法啟動，請查看上方錯誤訊息。

:failed
pause
exit /b 1

:end
endlocal
