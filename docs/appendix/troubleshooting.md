# Troubleshooting Guide

**Common Issues and Solutions for Physical AI Development**

This comprehensive troubleshooting guide covers the most frequent issues students encounter when working with ROS 2, Gazebo, Isaac Sim, and the full Physical AI stack. Issues are organized by technology and module for quick reference.

---

## Table of Contents

1. [ROS 2 Installation and Setup](#ros-2-installation-and-setup)
2. [Gazebo Classic and Gazebo Sim](#gazebo-classic-and-gazebo-sim)
3. [NVIDIA Isaac Sim](#nvidia-isaac-sim)
4. [GPU and CUDA Issues](#gpu-and-cuda-issues)
5. [Network and DDS Configuration](#network-and-dds-configuration)
6. [Python Dependencies](#python-dependencies)
7. [URDF and Robot Description](#urdf-and-robot-description)
8. [Navigation (Nav2)](#navigation-nav2)
9. [MoveIt2 Motion Planning](#moveit2-motion-planning)
10. [Computer Vision and Perception](#computer-vision-and-perception)
11. [WSL2 Specific Issues](#wsl2-specific-issues)

---

## ROS 2 Installation and Setup

### Issue: ROS 2 commands not found after installation

**Symptoms**:
```bash
ros2: command not found
```

**Solutions**:

1. **Source ROS 2 environment**:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

2. **Add to .bashrc for permanent fix**:
   ```bash
   echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

3. **Verify installation**:
   ```bash
   ros2 --version
   # Should show: ros2 cli version 0.x.x
   ```

---

### Issue: rosdep init fails with permission error

**Symptoms**:
```bash
sudo rosdep init
ERROR: cannot download default sources list from:
https://raw.githubusercontent.com/ros/rosdistro/master/rosdep/sources.list.d/20-default.list
```

**Solutions**:

1. **Check internet connection and try again**:
   ```bash
   ping raw.githubusercontent.com
   sudo rosdep init
   ```

2. **If already initialized**:
   ```bash
   sudo rm -rf /etc/ros/rosdep
   sudo rosdep init
   rosdep update
   ```

3. **Network proxy issues**:
   ```bash
   export http_proxy=http://your-proxy:port
   export https_proxy=http://your-proxy:port
   sudo -E rosdep init
   ```

---

### Issue: Colcon build fails with missing dependencies

**Symptoms**:
```bash
colcon build
--- stderr: my_robot_pkg
CMake Error: Could not find package "rclcpp"
```

**Solutions**:

1. **Install missing ROS 2 dependencies**:
   ```bash
   cd ~/ros2_ws
   rosdep install --from-paths src --ignore-src -r -y
   ```

2. **Source workspace before building**:
   ```bash
   source /opt/ros/humble/setup.bash
   colcon build
   ```

3. **Check package.xml has correct dependencies**:
   ```xml
   <depend>rclcpp</depend>
   <depend>std_msgs</depend>
   ```

---

### Issue: Workspace overlay not working

**Symptoms**: Changes to code don't take effect, or old versions run.

**Solutions**:

1. **Source workspace after building**:
   ```bash
   source ~/ros2_ws/install/setup.bash
   ```

2. **Check source order** (local workspace should be last):
   ```bash
   # In ~/.bashrc:
   source /opt/ros/humble/setup.bash
   source ~/ros2_ws/install/setup.bash
   ```

3. **Clean and rebuild**:
   ```bash
   cd ~/ros2_ws
   rm -rf build/ install/ log/
   colcon build
   ```

---

## Gazebo Classic and Gazebo Sim

### Issue: Gazebo black screen or not rendering

**Symptoms**: Gazebo window opens but shows only black screen, or crashes immediately.

**Solutions**:

1. **For Virtual Machine users**:
   ```bash
   export SVGA_VGPU10=0
   gazebo
   ```

2. **Update graphics drivers** (NVIDIA):
   ```bash
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

3. **Force software rendering**:
   ```bash
   export LIBGL_ALWAYS_SOFTWARE=1
   gazebo
   ```

4. **Check OpenGL support**:
   ```bash
   glxinfo | grep OpenGL
   # Should show version 3.3 or higher
   ```

---

### Issue: Gazebo crashes on launch

**Symptoms**:
```bash
gazebo
gzclient: symbol lookup error: /usr/lib/libgazebo_common.so
```

**Solutions**:

1. **Clear Gazebo cache**:
   ```bash
   rm -rf ~/.gazebo/
   gazebo
   ```

2. **Reinstall Gazebo**:
   ```bash
   sudo apt remove gazebo*
   sudo apt autoremove
   sudo apt install ros-humble-gazebo-ros-pkgs
   ```

3. **Check for conflicting installations**:
   ```bash
   dpkg -l | grep gazebo
   # Remove any non-ROS Gazebo installations
   ```

---

### Issue: Robot falls through ground plane

**Symptoms**: Robot spawns and immediately falls into void.

**Solutions**:

1. **Add collision to ground plane**:
   ```xml
   <collision>
     <geometry>
       <plane>
         <normal>0 0 1</normal>
       </plane>
     </geometry>
   </collision>
   ```

2. **Set ground plane as static**:
   ```xml
   <static>true</static>
   ```

3. **Increase robot position spawn height**:
   ```python
   spawn_args = ['-z', '0.5']  # 0.5 meters above ground
   ```

---

### Issue: Gazebo physics instability (robot exploding)

**Symptoms**: Robot shakes violently or parts fly apart when simulation starts.

**Solutions**:

1. **Increase mass and inertia values** in URDF:
   ```xml
   <inertial>
     <mass value="10.0"/>  <!-- Increase from very small values -->
     <inertia ixx="0.1" ixy="0.0" ixz="0.0"
              iyy="0.1" iyz="0.0" izz="0.1"/>
   </inertial>
   ```

2. **Add joint damping**:
   ```xml
   <joint name="my_joint" type="revolute">
     <dynamics damping="0.7" friction="0.5"/>
   </joint>
   ```

3. **Reduce physics update rate**:
   ```xml
   <physics type="ode">
     <max_step_size>0.001</max_step_size>
     <real_time_update_rate>1000</real_time_update_rate>
   </physics>
   ```

4. **Check for overlapping collision meshes**:
   - Visualize collision meshes in Gazebo (View → Collisions)
   - Ensure collision geometries don't overlap

---

### Issue: Gazebo sensors not publishing data

**Symptoms**: Camera or LIDAR topics exist but no data published.

**Solutions**:

1. **Check plugin is loaded** in URDF:
   ```xml
   <gazebo reference="camera_link">
     <sensor type="camera" name="camera1">
       <update_rate>30.0</update_rate>
       <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
         <ros>
           <namespace>/robot</namespace>
         </ros>
       </plugin>
     </sensor>
   </gazebo>
   ```

2. **Verify topic name**:
   ```bash
   ros2 topic list
   ros2 topic echo /camera/image_raw
   ```

3. **Check simulation is running** (not paused):
   - Press spacebar in Gazebo to play/pause

---

## NVIDIA Isaac Sim

### Issue: Isaac Sim won't launch

**Symptoms**:
```bash
./isaac-sim.sh
Error: Could not find isaac sim
```

**Solutions**:

1. **Set correct path**:
   ```bash
   export ISAAC_SIM_PATH="$HOME/.local/share/ov/pkg/isaac_sim-2023.1.1"
   cd $ISAAC_SIM_PATH
   ./isaac-sim.sh
   ```

2. **Check installation completed**:
   ```bash
   ls $ISAAC_SIM_PATH
   # Should show isaac-sim.sh, python.sh, etc.
   ```

3. **Reinstall via Omniverse Launcher** if files missing.

---

### Issue: Isaac Sim crashes with RTX error

**Symptoms**:
```
Failed to initialize RTX
CUDA error: device not supported
```

**Solutions**:

1. **Verify RTX GPU**:
   ```bash
   nvidia-smi
   # Must show RTX 20-series or newer
   ```

2. **Update NVIDIA drivers** (minimum 525):
   ```bash
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

3. **Check CUDA compatibility**:
   ```bash
   nvcc --version
   # Should be CUDA 11.8 or 12.x
   ```

---

### Issue: URDF import fails in Isaac Sim

**Symptoms**: Robot doesn't appear after import, or import dialog shows error.

**Solutions**:

1. **Check URDF file paths are absolute**:
   ```xml
   <!-- BAD -->
   <mesh filename="meshes/robot.stl"/>

   <!-- GOOD -->
   <mesh filename="package://my_robot/meshes/robot.stl"/>
   ```

2. **Set ROS_PACKAGE_PATH**:
   ```bash
   export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/ros2_ws/src
   ```

3. **Verify URDF is valid**:
   ```bash
   check_urdf my_robot.urdf
   ```

4. **Use Isaac's URDF converter script**:
   ```python
   import omni.isaac.urdf as urdf_importer

   config = urdf_importer.ImportConfig()
   config.merge_fixed_joints = True
   config.fix_base = False

   urdf_importer.import_urdf(
       urdf_path="/absolute/path/to/robot.urdf",
       import_config=config
   )
   ```

---

### Issue: Isaac Sim extremely slow

**Symptoms**: Simulation runs at less than 5 FPS, interface is laggy.

**Solutions**:

1. **Enable GPU physics**:
   - Edit → Preferences → Physics → Enable GPU acceleration

2. **Reduce rendering quality**:
   - Viewport → Rendering Mode → RTX Real-Time (not Path Traced)
   - Disable ambient occlusion and reflections

3. **Limit physics substeps**:
   ```python
   from omni.isaac.core import SimulationContext
   simulation_context = SimulationContext(
       physics_dt=1.0/60.0,
       rendering_dt=1.0/60.0
   )
   ```

4. **Check VRAM usage**:
   ```bash
   nvidia-smi
   # If near max VRAM, reduce texture resolution or scene complexity
   ```

---

### Issue: Isaac Sim ROS 2 bridge not working

**Symptoms**: No ROS 2 topics appear when running Isaac Sim.

**Solutions**:

1. **Enable ROS 2 bridge extension**:
   ```bash
   cd $ISAAC_SIM_PATH
   ./python.sh -m isaacsim.extsman.enableext omni.isaac.ros2_bridge
   ```

2. **Source ROS 2 before launching Isaac**:
   ```bash
   source /opt/ros/humble/setup.bash
   ./isaac-sim.sh
   ```

3. **Check FastDDS configuration** (see Network and DDS section below)

4. **Verify ROS_DOMAIN_ID matches**:
   ```bash
   echo $ROS_DOMAIN_ID
   # Should be same in Isaac Sim and ROS 2 terminals
   ```

---

## GPU and CUDA Issues

### Issue: NVIDIA driver not loading

**Symptoms**:
```bash
nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

**Solutions**:

1. **Reinstall driver**:
   ```bash
   sudo apt purge nvidia-*
   sudo apt autoremove
   sudo apt install nvidia-driver-535
   sudo reboot
   ```

2. **Check secure boot** (may block unsigned drivers):
   ```bash
   mokutil --sb-state
   # If enabled, disable secure boot in BIOS
   ```

3. **Blacklist nouveau driver** (open-source alternative):
   ```bash
   sudo bash -c "echo blacklist nouveau > /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
   sudo bash -c "echo options nouveau modeset=0 >> /etc/modprobe.d/blacklist-nvidia-nouveau.conf"
   sudo update-initramfs -u
   sudo reboot
   ```

---

### Issue: CUDA not found

**Symptoms**:
```bash
nvcc --version
nvcc: command not found
```

**Solutions**:

1. **Add CUDA to PATH**:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Install CUDA toolkit**:
   ```bash
   sudo apt install nvidia-cuda-toolkit
   ```

3. **Verify installation**:
   ```bash
   nvcc --version
   nvidia-smi
   ```

---

### Issue: Out of GPU memory

**Symptoms**:
```
CUDA out of memory. Tried to allocate X MiB
```

**Solutions**:

1. **Monitor VRAM usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

2. **Reduce batch size** in ML models:
   ```python
   batch_size = 8  # Reduce from 32
   ```

3. **Close other GPU applications** (browsers, other simulations)

4. **Use mixed precision training**:
   ```python
   model.half()  # Use FP16 instead of FP32
   ```

---

## Network and DDS Configuration

### Issue: ROS 2 nodes can't discover each other

**Symptoms**: `ros2 node list` shows only local nodes, not remote ones.

**Solutions**:

1. **Check ROS_DOMAIN_ID is same** on all machines:
   ```bash
   export ROS_DOMAIN_ID=0
   ```

2. **Verify network connectivity**:
   ```bash
   ping <other-machine-ip>
   ```

3. **Check firewall settings**:
   ```bash
   sudo ufw status
   # If active, allow ROS 2 ports
   sudo ufw allow 7400:7500/udp
   sudo ufw allow 7400:7500/tcp
   ```

4. **Use FastDDS with correct network interface**:
   ```xml
   <!-- fastdds.xml -->
   <dds>
     <profiles>
       <transport_descriptors>
         <transport_descriptor>
           <transport_id>udp_transport</transport_id>
           <type>UDPv4</type>
           <interfaceWhiteList>
             <address>192.168.1.0/24</address>
           </interfaceWhiteList>
         </transport_descriptor>
       </transport_descriptors>
     </profiles>
   </dds>
   ```

---

### Issue: High network latency with ROS 2

**Symptoms**: Slow message delivery, dropped messages.

**Solutions**:

1. **Use QoS settings for reliability**:
   ```python
   from rclpy.qos import QoSProfile, ReliabilityPolicy

   qos = QoSProfile(depth=10)
   qos.reliability = ReliabilityPolicy.RELIABLE

   self.publisher_ = self.create_publisher(String, 'topic', qos)
   ```

2. **Reduce message size**:
   - Use compressed image transport
   - Reduce sensor resolution/rate

3. **Check network bandwidth**:
   ```bash
   iperf3 -s  # On one machine
   iperf3 -c <server-ip>  # On other machine
   ```

---

### Issue: DDS incompatibility between Isaac Sim and ROS 2

**Symptoms**: Isaac Sim and ROS 2 nodes can't communicate.

**Solutions**:

1. **Use FastDDS on both**:
   ```bash
   # Install FastDDS
   sudo apt install ros-humble-rmw-fastrtps-cpp

   # Set RMW implementation
   export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
   ```

2. **Use consistent ROS 2 distribution** (Humble everywhere)

3. **Set FastDDS environment**:
   ```bash
   export FASTRTPS_DEFAULT_PROFILES_FILE=/path/to/fastdds.xml
   ```

---

## Python Dependencies

### Issue: ImportError for Python packages

**Symptoms**:
```python
ImportError: No module named 'cv2'
ModuleNotFoundError: No module named 'numpy'
```

**Solutions**:

1. **Install missing package**:
   ```bash
   pip3 install opencv-python numpy scipy matplotlib
   ```

2. **Use virtual environment**:
   ```bash
   python3 -m venv ~/physical-ai-env
   source ~/physical-ai-env/bin/activate
   pip install -r requirements.txt
   ```

3. **For Isaac Sim Python**:
   ```bash
   cd $ISAAC_SIM_PATH
   ./python.sh -m pip install opencv-python
   ```

---

### Issue: Conflicting Python versions

**Symptoms**: ROS 2 uses Python 3.10, but system has 3.8.

**Solutions**:

1. **Check ROS 2 Python version**:
   ```bash
   python3 --version
   # Should match ROS 2 requirement (3.10 for Humble)
   ```

2. **Use update-alternatives**:
   ```bash
   sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
   ```

3. **Always use python3 explicitly** in shebangs:
   ```python
   #!/usr/bin/env python3
   ```

---

### Issue: PyTorch CUDA version mismatch

**Symptoms**:
```python
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Solutions**:

1. **Check CUDA version**:
   ```bash
   nvcc --version
   nvidia-smi  # Shows compatible CUDA version
   ```

2. **Reinstall PyTorch with correct CUDA**:
   ```bash
   # For CUDA 11.8
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

   # For CUDA 12.1
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verify PyTorch CUDA**:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.version.cuda)
   ```

---

## URDF and Robot Description

### Issue: URDF parsing errors

**Symptoms**:
```
Error: XML parsing error: expected '>'
```

**Solutions**:

1. **Validate URDF syntax**:
   ```bash
   check_urdf my_robot.urdf
   ```

2. **Check for common XML errors**:
   - Missing closing tags
   - Unescaped special characters in names
   - Invalid attribute values

3. **Use xacro for complex URDFs**:
   ```bash
   xacro robot.urdf.xacro > robot.urdf
   check_urdf robot.urdf
   ```

---

### Issue: Robot visualization looks wrong

**Symptoms**: Robot appears distorted, scaled incorrectly, or parts missing.

**Solutions**:

1. **Check mesh file paths**:
   ```xml
   <mesh filename="package://my_robot/meshes/base.stl" scale="0.001 0.001 0.001"/>
   ```

2. **Verify origin transforms**:
   ```xml
   <joint name="joint1" type="revolute">
     <origin xyz="0 0 0.1" rpy="0 0 0"/>
   </joint>
   ```

3. **Check joint limits**:
   ```xml
   <limit lower="-1.57" upper="1.57" effort="10" velocity="1.0"/>
   ```

4. **Visualize in RViz2**:
   ```bash
   ros2 launch urdf_tutorial display.launch.py model:=my_robot.urdf
   ```

---

## Navigation (Nav2)

### Issue: Nav2 fails to plan path

**Symptoms**:
```
[planner_server]: Failed to create plan
```

**Solutions**:

1. **Check map is loaded**:
   ```bash
   ros2 topic echo /map --once
   ```

2. **Verify start and goal are in free space**:
   - Open RViz2, check robot and goal positions
   - Not inside walls or obstacles

3. **Increase planner timeout**:
   ```yaml
   planner_server:
     ros__parameters:
       expected_planner_frequency: 20.0
       planner_plugins: ["GridBased"]
       GridBased:
         plugin: "nav2_navfn_planner/NavfnPlanner"
         tolerance: 0.5
         use_astar: false
   ```

4. **Check costmap configuration**:
   ```bash
   ros2 topic echo /local_costmap/costmap
   ```

---

### Issue: Robot oscillates or gets stuck

**Symptoms**: Robot spins in place or repeatedly backs up.

**Solutions**:

1. **Tune DWB controller parameters**:
   ```yaml
   controller_server:
     ros__parameters:
       controller_frequency: 20.0
       FollowPath:
         plugin: "dwb_core::DWBLocalPlanner"
         min_vel_x: 0.0
         max_vel_x: 0.5
         max_vel_theta: 1.0
         min_speed_xy: 0.0
         max_speed_xy: 0.5
   ```

2. **Increase obstacle clearance**:
   ```yaml
   local_costmap:
     ros__parameters:
       inflation_layer:
         inflation_radius: 0.5
   ```

3. **Check AMCL localization quality**:
   ```bash
   ros2 topic echo /amcl_pose
   # Covariance should be low
   ```

---

## MoveIt2 Motion Planning

### Issue: MoveIt2 fails to plan

**Symptoms**:
```
[move_group]: Unable to find valid motion plan
```

**Solutions**:

1. **Check target pose is reachable**:
   ```python
   # Verify IK solution exists
   from moveit_commander import MoveGroupCommander
   arm = MoveGroupCommander("arm")
   target_pose = ...
   plan = arm.plan()  # Should not fail
   ```

2. **Increase planning time**:
   ```python
   arm.set_planning_time(10.0)  # 10 seconds
   ```

3. **Use different planner**:
   ```python
   arm.set_planner_id("RRTConnect")  # Or OMPL, BiTRRT, etc.
   ```

4. **Check collision objects**:
   ```python
   scene = PlanningSceneInterface()
   scene.remove_world_object("obstacle")
   ```

---

### Issue: Execution fails with controller error

**Symptoms**:
```
[follow_joint_trajectory] Aborted: Solution found but controller failed during execution
```

**Solutions**:

1. **Check joint limits in URDF match controller config**:
   ```yaml
   # ros2_controllers.yaml
   arm_controller:
     joints:
       - joint1
       - joint2
     limits:
       joint1:
         max_velocity: 2.0
         max_acceleration: 5.0
   ```

2. **Reduce trajectory speed**:
   ```python
   arm.set_max_velocity_scaling_factor(0.5)
   arm.set_max_acceleration_scaling_factor(0.5)
   ```

3. **Check joint states published**:
   ```bash
   ros2 topic echo /joint_states
   ```

---

## Computer Vision and Perception

### Issue: Camera not publishing images

**Symptoms**: No images on `/camera/image_raw` topic.

**Solutions**:

1. **Check camera driver running**:
   ```bash
   ros2 node list
   # Should show camera node
   ```

2. **Verify USB connection** (for physical cameras):
   ```bash
   ls /dev/video*
   # Should show /dev/video0, etc.
   ```

3. **Launch camera driver**:
   ```bash
   # RealSense
   ros2 launch realsense2_camera rs_launch.py

   # USB webcam
   ros2 run usb_cam usb_cam_node
   ```

4. **Check topic name**:
   ```bash
   ros2 topic list | grep image
   ```

---

### Issue: YOLO detection not working

**Symptoms**: No detections or very low confidence.

**Solutions**:

1. **Check model file loaded**:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolov8n.pt')  # Verify path
   ```

2. **Verify input image format**:
   ```python
   # BGR to RGB conversion needed
   image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
   results = model(image_rgb)
   ```

3. **Adjust confidence threshold**:
   ```python
   results = model(image, conf=0.25)  # Lower threshold
   ```

4. **Use correct model size for your hardware**:
   - YOLOv8n (nano): Fast, lower accuracy
   - YOLOv8m (medium): Balanced
   - YOLOv8x (extra large): Slow, higher accuracy

---

## WSL2 Specific Issues

### Issue: GUI applications don't display

**Symptoms**: Gazebo or RViz2 window doesn't open.

**Solutions**:

1. **Install WSLg** (Windows 11):
   ```bash
   # Should work out of the box on Windows 11
   gazebo
   ```

2. **Use X Server** (Windows 10):
   ```bash
   # Install VcXsrv or Xming on Windows
   export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0.0
   gazebo
   ```

---

### Issue: USB devices not accessible

**Symptoms**: RealSense or webcam not detected in WSL2.

**Solutions**:

1. **Use usbipd-win**:
   ```powershell
   # On Windows PowerShell (Admin)
   winget install usbipd
   usbipd wsl list
   usbipd wsl attach --busid <BUSID>
   ```

2. **Check device in WSL**:
   ```bash
   lsusb
   ls /dev/video*
   ```

---

### Issue: Poor GPU performance in WSL2

**Symptoms**: CUDA slower than expected.

**Solutions**:

1. **Install WSL2 GPU drivers**:
   - Download from NVIDIA website (WSL-specific)
   - Don't install CUDA in WSL (use Windows CUDA)

2. **Verify GPU access**:
   ```bash
   nvidia-smi
   ```

3. **Use native Linux for production** (WSL2 is for development only)

---

## General Debugging Tips

### Enable verbose logging

```bash
# ROS 2
ros2 run my_package my_node --ros-args --log-level debug

# Gazebo
gazebo --verbose

# Isaac Sim
# Check logs at: ~/.nvidia-omniverse/logs/Isaac-Sim/
```

### Monitor system resources

```bash
# CPU and RAM
htop

# GPU
nvidia-smi -l 1  # Update every second

# Disk I/O
iotop
```

### Check ROS 2 system health

```bash
ros2 wtf  # "Where's the failure?" diagnostic tool
ros2 doctor --report
```

---

## Getting Help

If issues persist after trying these solutions:

1. **Check official documentation**:
   - [ROS 2 Documentation](https://docs.ros.org/en/humble/)
   - [Gazebo Answers](https://answers.gazebosim.org/)
   - [NVIDIA Isaac Forums](https://forums.developer.nvidia.com/c/omniverse/isaac-sim/326)

2. **Search existing issues**:
   - GitHub Issues for relevant packages
   - ROS Discourse
   - Stack Overflow

3. **Ask for help**:
   - Post detailed error messages
   - Include system specs (OS, GPU, ROS version)
   - Describe steps to reproduce
   - Share relevant code snippets

4. **Course resources**:
   - Check course Discord/forum
   - Review module documentation
   - Consult with instructors

---

**Remember**: Most issues have been encountered and solved by others. Always search for error messages before asking for help!

**Last Updated**: December 2024
