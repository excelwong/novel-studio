@echo off
cd /d %~dp0
python novel_writer_studio.py
if errorlevel 1 pause
