@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
start "" "%ROOT%\maritime_output\first5s_last2s_masked_anchor1m_prev\aggregated_pointcloud_side_far_upright.html"
endlocal
