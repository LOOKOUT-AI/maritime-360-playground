@echo off
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
"%ROOT%\..\sam2\.venv\Scripts\python.exe" "%ROOT%\scripts\bootstrap_masks_sam2.py" ^
  --image "%ROOT%\maritime_input\first5s\frame_0050_for_masks.jpg" ^
  --sam2_root "%ROOT%\..\sam2" ^
  --checkpoint "%ROOT%\..\sam2\checkpoints\sam2.1_hiera_small.pt" ^
  --output_dir "%ROOT%\maritime_input\first5s\sam2_bootstrap"
endlocal
