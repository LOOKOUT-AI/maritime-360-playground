@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\view_pointcloud.py" --input "%ROOT%\maritime_output\frame_0020s_large_ellipse_pc\debug\image_erp_pointcloud.ply"
endlocal
