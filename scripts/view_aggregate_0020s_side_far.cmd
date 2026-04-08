@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
start "" "%ROOT%\maritime_output\aggregate_0020s_w1s_step2\aggregated_pointcloud_side_far_upright.html"
endlocal
