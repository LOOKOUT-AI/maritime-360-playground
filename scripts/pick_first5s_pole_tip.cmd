@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
"%ROOT%\.venv\Scripts\python.exe" "%ROOT%\scripts\pick_known_distance_point.py" ^
  --image "%ROOT%\maritime_input\first5s\frame_0050_for_masks.jpg" ^
  --output "%ROOT%\maritime_input\first5s\pole_tip_point.json" ^
  --label "pole_tip"
endlocal
