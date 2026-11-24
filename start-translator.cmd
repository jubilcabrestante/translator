@echo off
rem Double-click to run the translator on this PC for the Learnut app.
rem Keep the window open while it runs; press Ctrl+C to stop it.
rem Everything it does is described at the top of scripts\start_translator.ps1.
title Learnut translator
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start_translator.ps1" -PauseOnError %*
