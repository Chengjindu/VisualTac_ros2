# Project VisualTac – ROS2 Workspace

Visual-tactile sensing pipeline for the GelWedge sensor, implemented in ROS 2.

This repository contains:

- ROS 2 package: `gelwedge_demo_ros2`
- Pre- and post-processing utilities
- Calibration and startup scripts
- Host launch scripts for real-time operation using MJPEG or GStreamer streams

## 📁 Project Structure

```text
Project_VisualTac/
│
├── ros2_ws/
│   ├── src/
│   │   ├── gelwedge_demo_ros2/         # Main ROS2 package
│   │   └── find_marker/                # C++/Cython extension for fast marker tracking
│   └── ...
│
├── calibrate_matrix.sh
├── env_loader.sh
├── start_all.sh
├── start_calibration.sh
├── start_config.json
├── start_host.sh
├── requirements_ros2.txt
└── README.md
```

### 🚀 1. Installation Guide (for collaborators)

Follow these steps to set up the environment.

1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Project_VisualTac.git
cd Project_VisualTac
```

2. Create and activate Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

To activate again later:
```bash
source .venv/bin/activate
```
3. Install Python dependencies
```bash
pip install -r requirements_ros2.txt
```

Installed packages:
```bash
numpy
opencv-python
PyYAML
python-dateutil
pyparsing
```
⚠️ ROS 2 Python APIs (rclpy) come from the ROS 2 installation — not pip.


4. Install and source ROS 2

Ensure ROS 2 (Humble/Foxy/Galactic) is installed.
```bash
source /opt/ros/humble/setup.bash
```

5. Build the ROS2 workspace
```bash
cd ros2_ws
colcon build
source install/setup.bash
```

Optional: auto-source the overlay:
```bash
echo "source ~/Projects/Project_VisualTac/ros2_ws/install/setup.bash" >> ~/.bashrc
```

### 📷 2. System Configuration

The file start_config.json controls network and camera settings.

Example:
```bash
{
  "host_ip": "172.17.167.108",
  "pi_user": "chengjindu",
  "pi_ip": "172.17.167.150",
  "stream_type": "mjpg",
  "mjpg_url": "http://172.17.167.150:8080/?action=stream",
  "gst_port": 5000
}
```

Fields:

- host_ip – Host PC running ROS 2 
- pi_user / pi_ip – Raspberry Pi camera device 
- stream_type – "mjpg" or "gst"
- mjpg_url – MJPEG URL 
- gst_port – GStreamer UDP port

## ▶️ 3. Running the VisualTac System

All commands below assume you are in the project root (Project_VisualTac/) and your ROS 2 + workspace environments are sourced.

A. Start everything (host + ROS2 + camera)
```bash
./start_all.sh
```

This starts:

- Environment loader 
- Camera/streaming 
- ROS 2 nodes 
- Marker tracking 
- Tactile processing pipeline

B. Calibration mode
```bash
./start_calibration.sh
```

C. Host visualization only
```bash
./start_host.sh
```