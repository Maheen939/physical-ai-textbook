# Software Setup

This guide will walk you through installing all required software for the Physical AI & Humanoid Robotics course.

## Installation Overview

1. Ubuntu 22.04 LTS
2. Python 3.10+ and development tools
3. ROS 2 (Humble or Iron)
4. Gazebo Simulation
5. NVIDIA Isaac Sim (Module 3)
6. Additional tools and libraries

**Estimated Time**: 2-3 hours

---

## 1. Ubuntu 22.04 LTS Setup

### Option A: Native Installation (Recommended)
- Download Ubuntu 22.04 LTS from [ubuntu.com](https://ubuntu.com/download/desktop)
- Create bootable USB with [Rufus](https://rufus.ie/) (Windows) or `dd` (Linux/Mac)
- Dual-boot or dedicated install

### Option B: WSL2 (Windows Users)
```bash
wsl --install -d Ubuntu-22.04
```

### Option C: Virtual Machine
- Use VirtualBox or VMware
- Allocate: 4+ CPU cores, 16+ GB RAM, 100+ GB storage

---

## 2. System Update and Essential Tools

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    python3-pip \
    python3-venv \
    software-properties-common
```

---

## 3. Python 3.10+ Setup

Ubuntu 22.04 comes with Python 3.10. Verify:

```bash
python3 --version  # Should be 3.10 or higher
```

Create a virtual environment for the course:

```bash
python3 -m venv ~/physical-ai-env
source ~/physical-ai-env/bin/activate
pip install --upgrade pip
```

Install essential Python libraries:

```bash
pip install numpy scipy matplotlib opencv-python pytest jupyter
```

---

## 4. ROS 2 Installation (Humble)

### Add ROS 2 Repository

```bash
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### Install ROS 2 Humble Desktop

```bash
sudo apt update
sudo apt install -y ros-humble-desktop
```

### Install Development Tools

```bash
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-xacro
```

### Initialize rosdep

```bash
sudo rosdep init
rosdep update
```

### Setup Environment

Add to `~/.bashrc`:

```bash
source /opt/ros/humble/setup.bash
```

Apply changes:

```bash
source ~/.bashrc
```

### Verify Installation

```bash
ros2 --help
ros2 run demo_nodes_cpp talker  # In terminal 1
ros2 run demo_nodes_py listener  # In terminal 2
```

---

## 5. Gazebo Simulation

### Option A: Gazebo Classic (Simpler, Good for Beginners)

```bash
sudo apt install -y ros-humble-gazebo-ros-pkgs
gazebo --version  # Should be Gazebo 11.x
```

### Option B: Gazebo Sim (Newer, More Features)

```bash
sudo apt install -y ros-humble-ros-gz
gz sim --version  # Should be Gazebo Harmonic/Garden
```

### Verify Gazebo

```bash
gazebo  # Launch Gazebo (Classic)
# OR
gz sim  # Launch Gazebo Sim
```

---

## 6. Additional ROS 2 Packages

Install packages needed for later modules:

```bash
sudo apt install -y \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \
    ros-humble-moveit \
    ros-humble-control-msgs \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers
```

---

## 7. NVIDIA CUDA and GPU Setup (for NVIDIA Isaac)

**Note**: Only needed if you have an NVIDIA RTX GPU. Skip if using cloud.

### Install NVIDIA Drivers

```bash
sudo apt install -y nvidia-driver-535  # Or latest
sudo reboot
```

### Verify GPU

```bash
nvidia-smi
```

### Install CUDA Toolkit 12.x

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-3-local_12.3.0-545.23.06-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-3-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-3
```

Add to `~/.bashrc`:

```bash
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

---

## 8. NVIDIA Isaac Sim (Module 3)

**Note**: Postpone this until Module 3 (Week 8). Requires 30+ GB storage.

Instructions will be provided in [Chapter 9: NVIDIA Isaac Introduction](./module-3/09-nvidia-isaac-intro.md).

Quick reference:
- Install NVIDIA Omniverse Launcher
- Install Isaac Sim 2023.1.1 or later
- Requires RTX GPU

---

## 9. Intel RealSense SDK (Optional)

If using RealSense D435i camera:

```bash
sudo apt install -y ros-humble-realsense2-camera
```

---

## 10. Additional Development Tools

### Visual Studio Code

```bash
sudo snap install code --classic
```

Recommended Extensions:
- Python
- ROS
- C/C++
- Docker

### Git Configuration

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## 11. Clone Course Repository

```bash
cd ~
git clone https://github.com/YOUR_USERNAME/physical-ai-textbook.git
cd physical-ai-textbook/code_examples
```

---

## 12. Verification

Run the environment check script:

```bash
cd ~/physical-ai-textbook/code_examples/module-1/chapter-01
./check_environment.sh
```

This will verify:
- ✅ Ubuntu version
- ✅ Python installation
- ✅ ROS 2 installation
- ✅ Gazebo installation
- ✅ GPU capabilities (if applicable)

---

## Troubleshooting

### ROS 2 not sourced
Add to `~/.bashrc`:
```bash
source /opt/ros/humble/setup.bash
```

### Gazebo black screen
```bash
export SVGA_VGPU10=0  # For VM users
```

### CUDA not found
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

More troubleshooting: [Appendix: Troubleshooting](./appendix/troubleshooting.md)

---

## Next Steps

Software setup complete! Proceed to [Module 1: Introduction to Physical AI](./module-1/01-introduction-physical-ai.md).
