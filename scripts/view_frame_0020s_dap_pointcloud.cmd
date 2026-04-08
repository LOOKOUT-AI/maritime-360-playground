@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
start "" "%ROOT%\maritime_output\frame_0020s_dap_vitl\debug\image_erp_pointcloud.html"
endlocal
