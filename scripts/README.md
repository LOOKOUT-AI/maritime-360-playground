# Maritime Scripts

These scripts are experimental utilities built during exploratory prototyping. They are useful for understanding behavior and generating quick local results, but they are not a polished or fully validated pipeline.

The main custom scripts added on top of upstream DA360 are:

- `maritime_da360.py`: wraps DA360 for the downward-looking elliptical fisheye setup.
- `maritime_dap.py`: same adaptation, but using the DAP monocular depth backend.
- `aggregate_maritime_pointcloud.py`: short-window temporal fusion with masking, scale anchoring, and split background/boat clouds.
- `aggregate_maritime_pointcloud_vo.py`: more SLAM-like fusion with virtual horizon views, feature matching, PnP, and ICP refinement.
- `benchmark_auto_masks.py`: compares automatic boat and water mask generation methods on sampled frames.
- `propagate_auto_masks_sam2.py`: propagates selected automatic masks through a video with SAM2.
- `view_pointcloud.py` and `view_background_boat_pointcloud.py`: browser-based point-cloud viewers.

Utility scripts:

- `bootstrap_masks_sam2.py`: first-pass SAM2 mask generation on a single frame.
- `pick_known_distance_point.py`: click a reference point with known distance to the camera.
- `*.cmd`: convenience launchers for common local viewers and tools.
