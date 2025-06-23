***FCIM.FIA - Fundamentals of Artificial Intelligence***

> **Lab 5:** *Computer Vision* \
> **Performed by:** *Dumitru Moraru*, group *FAF-212* \
> **Verified by:** Elena Graur, asist. univ.

Imports and Utils


```python
import cv2
import numpy as np
import math
from scipy.spatial.distance import cdist
```

# Part 1 Imports and Global Configuration
This initial section loads the necessary libraries and defines the core parameters that control the detection and tracking behavior. These act as the main settings for the system.


```python
# --- Constants and Configuration ---
VIDEO_PATH = 'cars.mp4' 
PIXELS_PER_METER = 20
SPEED_LIMIT_KMH = 60

# --- Detection Tuning Parameters ---
MIN_CONTOUR_AREA = 1500
MIN_ASPECT_RATIO = 0.6
MAX_ASPECT_RATIO = 1.5
MAX_TRACKING_DISTANCE = 60
FRAMES_TO_DISAPPEAR = 5
```

Explanation
* The system ensures the existence of a directory for storing face data.
* New face encodings are saved as `.npy` (NumPy array) files.

# Part 2 The VehicleTracker Class
This class is the "brain" of the tracking logic. It is responsible for maintaining a memory of detected vehicles from one frame to the next.


```python
class VehicleTracker:
    """A simple class to track vehicles using centroid tracking."""
    def __init__(self):
        self.tracked_objects = {}
        self.next_id = 0

    def register(self, centroid, rect):
        """Registers a new vehicle."""
        self.tracked_objects[self.next_id] = {
            'centroid': centroid,
            'rect': rect,
            'speed_kmh': 0,
            'frames_without_detection': 0
        }
        self.next_id += 1

    def deregister(self, object_id):
        """Removes a vehicle from tracking."""
        # Make sure the key exists before trying to delete it
        if object_id in self.tracked_objects:
            del self.tracked_objects[object_id]

    def update(self, detected_centroids, fps):
        """Updates the state of tracked vehicles."""

        # If there are no detected centroids, increment disappearance counter for all tracked objects.
        if not detected_centroids:
            for obj_id in list(self.tracked_objects.keys()):
                self.tracked_objects[obj_id]['frames_without_detection'] += 1
                if self.tracked_objects[obj_id]['frames_without_detection'] > FRAMES_TO_DISAPPEAR:
                    self.deregister(obj_id)
            return self.tracked_objects

        # If no objects are being tracked yet, register all new detections.
        if not self.tracked_objects:
            for centroid, rect in detected_centroids:
                self.register(centroid, rect)
            return self.tracked_objects

        # Prepare to match detected centroids to existing tracked objects.
        object_ids = list(self.tracked_objects.keys())
        previous_centroids = np.array([obj['centroid'] for obj in self.tracked_objects.values()])
        current_centroids = np.array([c[0] for c in detected_centroids])

        # Ensure there are centroids to compare before calculating distance
        if len(previous_centroids) == 0 or len(current_centroids) == 0:
            if len(current_centroids) > 0:
                 for centroid, rect in detected_centroids:
                    is_new = True
                    for obj in self.tracked_objects.values():
                        if np.array_equal(obj['centroid'], centroid):
                           is_new = False
                           break
                    if is_new:
                        self.register(centroid, rect)
            return self.tracked_objects


        # Calculate distances between each previous centroid and each current centroid.
        D = cdist(previous_centroids, current_centroids)

        # Find the best match for each tracked object.
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            if D[row, col] > MAX_TRACKING_DISTANCE:
                continue

            obj_id = object_ids[row]
            new_centroid, new_rect = detected_centroids[col]
            old_centroid = self.tracked_objects[obj_id]['centroid']

            distance_pixels = math.hypot(new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
            distance_meters = distance_pixels / PIXELS_PER_METER

            if fps > 0:
                speed_mps = distance_meters * fps
                speed_kmh = speed_mps * 3.6
                self.tracked_objects[obj_id]['speed_kmh'] = 0.9 * self.tracked_objects[obj_id]['speed_kmh'] + 0.1 * speed_kmh
            else:
                self.tracked_objects[obj_id]['speed_kmh'] = 0

            self.tracked_objects[obj_id]['centroid'] = new_centroid
            self.tracked_objects[obj_id]['rect'] = new_rect
            self.tracked_objects[obj_id]['frames_without_detection'] = 0

            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(0, D.shape[0])).difference(used_rows)
        for row in unused_rows:
            obj_id = object_ids[row]
            self.tracked_objects[obj_id]['frames_without_detection'] += 1
            if self.tracked_objects[obj_id]['frames_without_detection'] > FRAMES_TO_DISAPPEAR:
                self.deregister(obj_id)

        unused_cols = set(range(0, D.shape[1])).difference(used_cols)
        for col in unused_cols:
            self.register(detected_centroids[col][0], detected_centroids[col][1])

        return self.tracked_objects
```

* It calculates the distance between every previously tracked object and every newly detected object.
* It finds the best matches based on the smallest distances.
* For each successful match, it updates the vehicle's position and calculates its new speed based on the distance traveled since the last frame.
* It registers any new, unmatched detections as new vehicles.
* It increments a "disappeared" counter for any tracked vehicles that were not found in the current frame and deregisters them if they've been missing for too long.

# Part 3 Main Application Logic
This is the primary execution block of the script, containing the main `while` loop that processes the video frame by frame.


```python
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
tracker = VehicleTracker()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    fg_mask = bg_subtractor.apply(blurred)
    _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    # Refined morphological operations for better noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    dilated_mask = cv2.dilate(opened_mask, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_centroids = []
    for cnt in contours:
        contour_area = cv2.contourArea(cnt)
        if contour_area > MIN_CONTOUR_AREA:
            rect = cv2.boundingRect(cnt)
            x, y, w, h = rect

            # *** NEW: Filter by Aspect Ratio ***
            if h > 0: # Avoid division by zero
                aspect_ratio = float(w) / h
                if aspect_ratio > MIN_ASPECT_RATIO and aspect_ratio < MAX_ASPECT_RATIO:
                    M = cv2.moments(cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        detected_centroids.append(((cx, cy), rect))

    tracked_objects = tracker.update(detected_centroids, video_fps)

    speeding_count = 0
    for obj_id, data in tracked_objects.items():
        rect = data['rect']
        speed = data['speed_kmh']
        x, y, w, h = rect

        is_speeding = speed > SPEED_LIMIT_KMH
        box_color = (0, 0, 255) if is_speeding else (0, 255, 0)

        if is_speeding:
            speeding_count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
        speed_text = f"{speed:.1f} km/h"
        cv2.putText(frame, speed_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    total_vehicles = len(tracked_objects)
    info_text_total = f"Detected Vehicles: {total_vehicles}"
    info_text_speeding = f"Speeding: {speeding_count}"
    cv2.putText(frame, info_text_total, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(frame, info_text_speeding, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Vehicle Speed Detection', frame)
    # cv2.imshow('Foreground Mask', dilated_mask)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

-   **Image Pre-processing**: The frame is converted to grayscale and blurred. This reduces detail and noise, making the subsequent motion detection more reliable.
-   **Background Subtraction**: The `fg_mask` is created, which is a black-and-white image where white pixels represent foreground motion.
-   **Mask Cleaning**: Morphological operations (`MORPH_OPEN` and `dilate`) are applied to the mask. This cleans up small noise artifacts and fills holes in the detected objects, resulting in cleaner, more solid shapes.
-   **Contour Detection**: `cv2.findContours` is used to find the outlines of all the white shapes in the cleaned mask.
-   **Contour Filtering**: The script iterates through every detected contour and filters them based on the tuning parameters (`MIN_CONTOUR_AREA` and aspect ratio). Only contours that are large enough and have a car-like shape are kept. The centroids of these valid contours are stored.
-   **Tracker Update**: The list of valid centroids is passed to `tracker.update()`. The tracker performs its matching logic and returns the updated state of all tracked vehicles.
-   **Visualization**: The code loops through the tracked objects from the tracker. For each vehicle, it draws a bounding box (red for speeding, green otherwise) and displays its calculated speed on the original frame. Summary text (total detected vehicles, speeding count) is also added.
-   **Display**: `cv2.imshow(...)` displays the final processed frame in a window.

# Conclusions:
This project successfully a vehicle speed detection system using Python and OpenCV. The program effectively identifies, tracks, and calculates the speed of cars in video footage by using background subtraction and centroid tracking. While the system is efficient, its accuracy depends on stable lighting and camera calibration. Future improvements could involve implementing a Region of Interest (ROI) to focus detection or upgrading to a deep learning model like YOLO for more robust performance.

# Bibliography:

1) https://opencv.org/
2) https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html
3) https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/
4) https://pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
5) https://www.myzhar.com/blog/tutorials/tutorial-opencv-ball-tracker-using-kalman-filter/
