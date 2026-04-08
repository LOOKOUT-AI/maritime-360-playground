@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
start "" "%ROOT%\maritime_output\first5s_last2s_masked_anchor1m_split_max30\background_boat_view.html"
endlocal
