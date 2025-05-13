# 🚗 Computer Vision for Autonomous Vehicles

This project implements a real-time **intelligent road navigation system** using computer vision and image processing in OpenCV. It segments roads and lanes, detects obstacles, and makes steering decisions (stop, slow down, turn left/right, or go) based on object and road geometry.

## 🧠 Features

* **Road Segmentation**: Uses HSV color filtering and morphological operations to extract the road from a video frame.
* **Lane Detection**: Detects yellow-colored lanes in the frame using HSV thresholding.
* **Obstacle Detection**: Applies dilation-based edge detection and contour analysis to identify potential obstacles on the road.
* **Decision Making**: Determines whether the vehicle should stop, slow down, or turn based on the location of detected obstacles and road geometry.
* **Visual Feedback**: Provides annotated frames in real-time to visualize segmentation, detections, and decisions.

---

## 🗂 Project Structure

```
.
├── Project.py                # Core script for frame processing and decision-making
├── Pi_frames_reciever.py     # Script to connect Pi to Laptop
├── Demo.mp4                  # Demonstration Video
├── README.md                 # This file
```

---

## 🛠 Requirements

* Python 3.7+
* OpenCV (`cv2`)
* NumPy
* Raspberry pi 3+

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## ▶️ How It Works

1. **Video Frame Extraction**

   * The system reads each frame from the input video.

2. **Road & Lane Segmentation**

   * Road is extracted using HSV thresholding for red/pink hues and morphological filtering.
   * Lanes are detected by isolating yellow hues in HSV space.

3. **Edge and Obstacle Detection**

   * Edges are extracted via dilation and thresholding.
   * Obstacles are detected using contour properties like area, solidity, and extent.

4. **Decision Logic**

   * If obstacles are within a certain range, the system suggests stopping or turning.
   * If no obstacle is detected, curvature analysis is used to suggest direction.

---

## 🧪 Key Thresholds and Parameters

* `THRESH_STOP`: Y-coordinate threshold to issue a stop command.
* `MIN_AREA`, `MIN_SOLIDITY`, `MIN_EXTENT`: Filters for valid obstacle contours.
* `TOLERANCE_TURN`: Threshold to decide turn direction based on road centroid.

Adjust these values in `Project.py` for different datasets or camera setups.

---

## 🖼 Sample Output

> Annotated video frames with:

* Red bounding boxes for detected obstacles
* Text overlays indicating navigation decisions
* Yellow boudning boxes for possible obstacles
* Green rectangles indicating the region of interest for slowing down and turning

---

## 📌 Notes

* Designed for videos with clear road visibility and colored lane/obstacle patterns.
* The `cv2.imshow()` functions are used for live debugging. You can disable them for headless processing.
* This is a prototype, developed as part of Digital Image Processing Course; real-world deployment would require sensor fusion (e.g., LIDAR/GPS + camera) and safety enhancements.

---
