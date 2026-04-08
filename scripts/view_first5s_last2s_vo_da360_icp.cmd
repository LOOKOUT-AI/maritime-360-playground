@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
start "" "%ROOT%\maritime_output\first5s_last2s_vo_da360_icp\background_boat_view.html"
endlocal
