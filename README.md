# ARK-Perception-Tasks-2025-Mayank-Tiwari-25MI10016


This repository contains my solutions for the ARK Perception Team selection tasks. The tasks cover computer vision fundamentals including noise filtering, edge detection, the Hough Transform, and background subtraction.


Task 2.1 — Vision-Based Line Follower
Couldn't complete this one. The Simulink Support Package for Parrot Minidrones doesn't support Apple Silicon Macs, which is what I'm on. The algorithm is planned out — grayscale the camera feed, threshold it, find the line centroid, compute the angle offset — but couldn't actually run it without the package.

Task 2.2 — Noise Filtering
Two separate scripts for the two images.
Iron Man (iron_man.py) — line drawing with salt noise. Used median blur at kernel size 3. Salt noise pixels are always value 255 so they're never the median in their neighbourhood, which means they get removed cleanly without touching the actual lines. Gaussian blur was tried first and just smeared the noise around instead of removing it.
Scenery (scnery.py) — colour photograph with salt noise. Used bilateral filter so the colours stay intact and edges stay sharp. The script processes the full colour image directly — an earlier version accidentally converted to grayscale first which stripped all the colour out.
Both scripts print the SNR for each method tried and save comparison charts to the output folder.

Task 2.3 — Medial Axis Detection
medial_axis.py — processes all 3 surgical tool videos and draws the medial axis in green on each frame.
Pipeline: background subtraction with MOG2 → morphological cleaning → Sobel edge detection → custom Hough Transform → average the two strongest lines to get the medial axis.
The Hough Transform is written from scratch with no cv2.HoughLines. The first version used nested Python loops and took about a minute per frame. Replaced the loops with NumPy broadcasting which computes the entire rho matrix in one operation — dropped it down to under 100ms per frame.
Output videos: 
