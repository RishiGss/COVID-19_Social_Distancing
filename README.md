# COVID-19_Social_Distancing
Real-time social distancing monitoring and individual tracking with live video feed from CCTV cameras. 

This solution helps in detecting and tracking if people in public places are violating/following social distancing measure. Unless the vaccine is developed, Social Distancing is the only saviour to control the spread of the pandemic!!!. Also, an individual can be tracked if needed.

You can watch the demo of this project in this [video](https://www.youtube.com/watch?v=pCI6UVSg5tI "COVID19 Social Distancing")

### Prerequsites:
1. Nvidia GPU with CUDA support.
2. Update Nvidia GPU drivers and CUDA development Kit.
3. MicroSoft Visual Studio 2017 or newer version.
4. Python 3. (Optional but Recommended)
5. C-Make.
6. Build OpenCV with GPU support

### Building OpenCV from Source with GPU support
To build OpenCV from Source with GPU support, watch this [video](https://www.youtube.com/watch?v=TT3_dlPL4vo "Build OpenCV with CUDA support")

### Steps to reproduce this project:
1. Clone this repository.
2. Create a python/conda environment and install the required packages.
3. Download the weigths file.
   File links:
   [YOLOv3 weights](https://pjreddie.com/media/files/yolov3.weights)   [YOLOv3 tiny weights](https://pjreddie.com/media/files/yolov3-tiny.weights)
4. Run **social_distance_monitoring** for real-time social distance monitoring.
5. Run  **opencv_object_tracking** for tracking an individual in the video.
