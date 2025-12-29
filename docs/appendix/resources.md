# Resources

**Comprehensive Resource Guide for Physical AI and Humanoid Robotics**

This curated collection provides essential learning materials, documentation, tools, and community resources for your Physical AI journey. Resources are organized by category for easy navigation.

---

## Table of Contents

1. [Official Documentation](#official-documentation)
2. [Video Tutorials and Courses](#video-tutorials-and-courses)
3. [Research Papers](#research-papers)
4. [Books](#books)
5. [Community Forums and Discord](#community-forums-and-discord)
6. [Hardware Purchasing Guides](#hardware-purchasing-guides)
7. [Software Tools and Libraries](#software-tools-and-libraries)
8. [Simulation Assets](#simulation-assets)
9. [Datasets](#datasets)
10. [Blogs and Newsletters](#blogs-and-newsletters)

---

## Official Documentation

### ROS 2

**ROS 2 Documentation Hub**
- URL: https://docs.ros.org/en/humble/
- Description: Complete ROS 2 Humble documentation including installation, tutorials, and API reference
- Best for: All ROS 2 development

**ROS 2 Design Documentation**
- URL: https://design.ros2.org/
- Description: Architectural decisions, design patterns, and rationale behind ROS 2
- Best for: Understanding ROS 2 internals

**ROS Index**
- URL: https://index.ros.org/
- Description: Searchable directory of ROS packages with documentation and usage examples
- Best for: Finding existing packages

**Navigation 2 (Nav2)**
- URL: https://navigation.ros.org/
- Description: Complete Nav2 documentation with tutorials, behavior tree explanations, and tuning guides
- Best for: Autonomous navigation

**MoveIt2**
- URL: https://moveit.ros.org/
- Description: Motion planning framework documentation with tutorials and API reference
- Best for: Manipulation and motion planning

---

### NVIDIA Isaac

**Isaac Sim Documentation**
- URL: https://docs.omniverse.nvidia.com/isaacsim/latest/
- Description: Complete Isaac Sim user guide, Python API, and tutorials
- Best for: Simulation development

**Isaac ROS Documentation**
- URL: https://nvidia-isaac-ros.github.io/
- Description: GPU-accelerated ROS packages for perception and navigation
- Best for: Real-time perception

**Omniverse Documentation**
- URL: https://docs.omniverse.nvidia.com/
- Description: Omniverse platform documentation including USD, connectors, and extensions
- Best for: USD workflows

**Isaac SDK Documentation**
- URL: https://developer.nvidia.com/isaac-sdk
- Description: Isaac GEMs, navigation algorithms, and deployment tools
- Best for: Edge deployment

---

### Gazebo

**Gazebo Classic Documentation**
- URL: https://classic.gazebosim.org/tutorials
- Description: Tutorials for Gazebo 11 (classic), widely used with ROS 2
- Best for: Traditional simulation

**Gazebo Sim (New Generation)**
- URL: https://gazebosim.org/docs
- Description: Documentation for Gazebo Harmonic/Garden (modern version)
- Best for: Modern simulation features

---

### Additional Tools

**OpenCV Documentation**
- URL: https://docs.opencv.org/4.x/
- Description: Computer vision library documentation with Python and C++ tutorials
- Best for: Image processing

**PyTorch Documentation**
- URL: https://pytorch.org/docs/stable/index.html
- Description: Deep learning framework documentation and tutorials
- Best for: Neural network development

**Universal Scene Description (USD)**
- URL: https://openusd.org/release/index.html
- Description: Pixar's USD format documentation
- Best for: 3D scene composition

**YOLO Documentation**
- URL: https://docs.ultralytics.com/
- Description: YOLOv8 and YOLOv11 documentation with examples
- Best for: Object detection

---

## Video Tutorials and Courses

### YouTube Channels

**Articulated Robotics**
- URL: https://www.youtube.com/@ArticulatedRobotics
- Description: Josh Newans' excellent ROS 2 tutorials, hardware integration, and navigation
- Topics: ROS 2 basics, URDF, Nav2, hardware
- Recommended: ROS 2 Control tutorial series

**The Construct**
- URL: https://www.youtube.com/@TheConstruct
- Description: Professional ROS training with simulation environments
- Topics: ROS 2, Gazebo, navigation, manipulation
- Recommended: 5 Days to Master ROS 2

**NVIDIA Developer**
- URL: https://www.youtube.com/@NVIDIADeveloper
- Description: NVIDIA official channel with Isaac Sim tutorials and AI robotics content
- Topics: Isaac Sim, Isaac ROS, Omniverse
- Recommended: Isaac Sim tutorial playlist

**ROS Developers Podcast**
- URL: https://www.youtube.com/@TheConstructSim
- Description: Interviews with ROS core developers and industry experts
- Topics: ROS development, industry applications
- Recommended: State of ROS series

**Robotics Back-End**
- URL: https://www.youtube.com/@RoboticsBackEnd
- Description: Practical ROS 2 programming tutorials
- Topics: Python ROS 2, nodes, topics, services
- Recommended: ROS 2 for Beginners

---

### Online Courses

**ROS 2 Online Courses (The Construct)**
- URL: https://www.theconstructsim.com/robotigniteacademy_learnros/
- Cost: Subscription-based (~$30/month)
- Description: Comprehensive ROS 2 courses with browser-based simulation
- Courses: ROS 2 Basics, Nav2, MoveIt2, Perception

**Coursera: Modern Robotics Specialization**
- URL: https://www.coursera.org/specializations/modernrobotics
- Cost: Free to audit, ~$49/month for certificate
- Institution: Northwestern University
- Topics: Kinematics, dynamics, motion planning
- Duration: 6 courses, ~8 months

**edX: Autonomous Mobile Robots**
- URL: https://www.edx.org/learn/robotics
- Cost: Free to audit
- Institution: ETH Zurich
- Topics: SLAM, path planning, control

**NVIDIA Deep Learning Institute**
- URL: https://www.nvidia.com/en-us/training/
- Cost: Some free, paid certifications available
- Topics: Isaac Sim, Isaac ROS, AI for robotics
- Recommended: Getting Started with AI on Jetson Nano

---

## Research Papers

### Foundational Papers

**Vision-Language-Action Models**

1. **"RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control"**
   - Authors: Brohan et al. (Google DeepMind)
   - Year: 2023
   - URL: https://arxiv.org/abs/2307.15818
   - Key contribution: End-to-end VLA model trained on web data

2. **"PaLM-E: An Embodied Multimodal Language Model"**
   - Authors: Driess et al. (Google)
   - Year: 2023
   - URL: https://arxiv.org/abs/2303.03378
   - Key contribution: Large multimodal model for embodied tasks

3. **"Code as Policies: Language Model Programs for Embodied Control"**
   - Authors: Liang et al. (Google)
   - Year: 2023
   - URL: https://arxiv.org/abs/2209.07753
   - Key contribution: Using LLMs to generate robot control code

---

### Humanoid Locomotion

4. **"Learning to Walk via Deep Reinforcement Learning"**
   - Authors: Haarnoja et al. (Berkeley)
   - Year: 2018
   - URL: https://arxiv.org/abs/1812.11103
   - Key contribution: Learning bipedal walking from scratch

5. **"DeepGait: Planning and Control of Quadrupedal Gaits using Deep Reinforcement Learning"**
   - Authors: Tan et al. (Google)
   - Year: 2020
   - URL: https://arxiv.org/abs/2004.00766
   - Key contribution: Learned locomotion for quadrupeds (applies to bipeds)

---

### Manipulation and Grasping

6. **"DexNet 2.0: Deep Learning to Plan Robust Grasps"**
   - Authors: Mahler et al. (Berkeley)
   - Year: 2017
   - URL: https://arxiv.org/abs/1703.09312
   - Key contribution: Data-driven grasp planning

7. **"Learning Synergies between Pushing and Grasping with Self-supervised Deep Reinforcement Learning"**
   - Authors: Zeng et al. (Princeton)
   - Year: 2018
   - URL: https://arxiv.org/abs/1803.09956
   - Key contribution: Combined manipulation primitives

---

### Sim-to-Real Transfer

8. **"Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"**
   - Authors: Peng et al. (Berkeley)
   - Year: 2018
   - URL: https://arxiv.org/abs/1710.06537
   - Key contribution: Domain randomization techniques

9. **"Learning Dexterous In-Hand Manipulation"**
   - Authors: OpenAI et al.
   - Year: 2019
   - URL: https://arxiv.org/abs/1808.00177
   - Key contribution: Massively parallel sim-to-real for manipulation

---

### Navigation and SLAM

10. **"ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial and Multi-Map SLAM"**
    - Authors: Campos et al.
    - Year: 2021
    - URL: https://arxiv.org/abs/2007.11898
    - Key contribution: State-of-the-art visual SLAM

---

## Books

### Robotics Fundamentals

**"Modern Robotics: Mechanics, Planning, and Control"**
- Authors: Kevin M. Lynch and Frank C. Park
- Publisher: Cambridge University Press, 2017
- ISBN: 978-1107156302
- Description: Comprehensive modern robotics textbook with online videos
- Free PDF: http://hades.mech.northwestern.edu/index.php/Modern_Robotics
- Best for: Kinematics, dynamics, control theory

**"Introduction to Autonomous Mobile Robots"**
- Authors: Roland Siegwart, Illah Nourbakhsh, Davide Scaramuzza
- Publisher: MIT Press, 2011 (2nd Edition)
- ISBN: 978-0262015356
- Description: Classic text on mobile robotics covering perception, localization, and planning
- Best for: Mobile robot fundamentals

**"Robotics: Modelling, Planning and Control"**
- Authors: Bruno Siciliano, Lorenzo Sciavicco, Luigi Villani, Giuseppe Oriolo
- Publisher: Springer, 2010
- ISBN: 978-1846286414
- Description: Detailed treatment of manipulators and control
- Best for: Manipulation, inverse kinematics

---

### ROS 2 Programming

**"A Concise Introduction to Robot Programming with ROS 2"**
- Authors: Francisco Martín Rico, David García Castro
- Publisher: CRC Press, 2022
- ISBN: 978-1032116945
- Description: Practical ROS 2 programming guide
- Best for: Learning ROS 2 from scratch

**"Programming Robots with ROS: A Practical Introduction"**
- Authors: Morgan Quigley, Brian Gerkey, William D. Smart
- Publisher: O'Reilly, 2015
- ISBN: 978-1449323899
- Description: ROS 1 book, but concepts apply to ROS 2
- Best for: Understanding ROS philosophy

---

### AI and Machine Learning for Robotics

**"Probabilistic Robotics"**
- Authors: Sebastian Thrun, Wolfram Burgard, Dieter Fox
- Publisher: MIT Press, 2005
- ISBN: 978-0262201629
- Description: Essential text on probabilistic methods in robotics
- Best for: SLAM, localization, Kalman filters

**"Deep Learning for Robot Perception and Cognition"**
- Authors: Marios Xanthidis, Ashutosh Saxena
- Publisher: Academic Press, 2022
- ISBN: 978-0323857871
- Description: Modern deep learning applications in robotics
- Best for: Vision, learning-based control

---

## Community Forums and Discord

### Official Forums

**ROS Discourse**
- URL: https://discourse.ros.org/
- Description: Official ROS discussion forum for Q&A, announcements, and community discussions
- Active: 10,000+ users
- Response time: Usually within hours

**NVIDIA Isaac Forum**
- URL: https://forums.developer.nvidia.com/c/omniverse/isaac-sim/326
- Description: Official support forum for Isaac Sim and Isaac ROS
- Active: Moderated by NVIDIA developers
- Response time: 1-2 business days

**Gazebo Answers**
- URL: https://answers.gazebosim.org/
- Description: Q&A site for Gazebo simulation
- Active: Community-driven
- Note: Transitioning to GitHub Discussions

---

### Discord and Slack Communities

**ROS 2 Community Discord**
- Invite: Available through https://www.ros.org/community/
- Size: 5,000+ members
- Channels: General, hardware, simulation, AI
- Best for: Quick questions, real-time help

**Robotics Worldwide Discord**
- Invite: https://discord.gg/robotics
- Size: 15,000+ members
- Channels: Robotics, AI, hardware, jobs
- Best for: General robotics discussions

**NVIDIA Omniverse Discord**
- Invite: https://discord.gg/nvidiaomniverse
- Size: 10,000+ members
- Channels: Isaac Sim, USD, extensions
- Best for: Isaac Sim community support

**Panaverse Community**
- Invite: Contact course instructors
- Size: Growing
- Channels: Physical AI course support, projects, showcases
- Best for: Course-specific help

---

### Reddit Communities

**r/ROS**
- URL: https://www.reddit.com/r/ROS/
- Members: 15,000+
- Activity: Daily posts
- Best for: ROS news, project showcases

**r/robotics**
- URL: https://www.reddit.com/r/robotics/
- Members: 250,000+
- Activity: Very active
- Best for: General robotics discussions

**r/MachineLearning**
- URL: https://www.reddit.com/r/MachineLearning/
- Members: 2.5M+
- Activity: Research papers, implementations
- Best for: ML techniques for robotics

---

## Hardware Purchasing Guides

### Development Computers

**GPU Workstation for Isaac Sim**
- **Budget Option** (~$1,500):
  - GPU: NVIDIA RTX 4060 Ti 16GB
  - CPU: AMD Ryzen 7 7700X
  - RAM: 32GB DDR5
  - Storage: 1TB NVMe SSD

- **Recommended** (~$2,500):
  - GPU: NVIDIA RTX 4070 Ti 12GB
  - CPU: Intel i7-13700K
  - RAM: 64GB DDR5
  - Storage: 2TB NVMe SSD

- **Professional** (~$4,000+):
  - GPU: NVIDIA RTX 4090 24GB
  - CPU: AMD Ryzen 9 7950X or Intel i9-13900K
  - RAM: 128GB DDR5
  - Storage: 4TB NVMe SSD

**Where to Buy**:
- Pre-built: Dell Precision, HP Z-series, Lenovo ThinkStation
- Custom: NZXT, CyberpowerPC, iBuyPower
- DIY: Newegg, Micro Center, Amazon

---

### Edge Computing

**NVIDIA Jetson Orin Nano Developer Kit**
- Price: ~$499
- Performance: 40 TOPS
- RAM: 8GB
- Best for: Learning, prototyping
- Where to buy: NVIDIA store, Seeed Studio, SparkFun

**NVIDIA Jetson Orin NX**
- Price: ~$699 (16GB) / ~$899 (32GB)
- Performance: 100 TOPS
- Best for: Production deployments
- Where to buy: NVIDIA partners, Arrow Electronics

**Raspberry Pi 5**
- Price: ~$60-$80
- Performance: Limited (no GPU acceleration)
- Best for: Simple robots, learning Linux
- Where to buy: Raspberry Pi retailers worldwide

---

### Cameras and Sensors

**Intel RealSense D435i**
- Price: ~$299
- Type: RGB-D camera with IMU
- Range: 0.3m to 3m
- Best for: Indoor object detection, VSLAM
- Where to buy: Intel, Amazon, Mouser

**Intel RealSense L515**
- Price: ~$849
- Type: LiDAR-based depth camera
- Range: 0.25m to 9m
- Best for: High-precision scanning
- Where to buy: Intel, authorized resellers

**Logitech C920 Webcam**
- Price: ~$70
- Type: USB webcam (1080p)
- Best for: Budget computer vision projects
- Where to buy: Amazon, Best Buy

**RPLiDAR A1**
- Price: ~$99
- Type: 2D LIDAR (360°)
- Range: 12m
- Best for: Budget 2D SLAM
- Where to buy: Seeed Studio, Amazon

**YDLIDAR X4**
- Price: ~$89
- Type: 2D LIDAR (360°)
- Range: 10m
- Best for: Alternative budget LIDAR
- Where to buy: Amazon, RobotShop

---

### Robot Platforms

**TurtleBot 4**
- Price: ~$1,595 (Standard) / ~$2,195 (Lite with sensors)
- Platform: Mobile robot with Create 3 base
- Compute: Raspberry Pi 4
- Best for: ROS 2 learning, research
- Where to buy: Clearpath Robotics

**NVIDIA Carter Robot**
- Price: Reference design (build yourself)
- Platform: Differential drive with RealSense
- Compute: Jetson AGX Orin
- Best for: Isaac ROS development
- Where to buy: Build from reference design

**Unitree Go1**
- Price: ~$2,700 (Edu version)
- Platform: Quadruped robot
- Best for: Advanced locomotion research
- Where to buy: Unitree Robotics

---

### Manipulators

**Trossen Robotics PincherX 100**
- Price: ~$799
- DOF: 4 + gripper
- Payload: 50g
- Best for: Learning manipulation
- Where to buy: Trossen Robotics

**INTERBOTIX ViperX 300**
- Price: ~$2,999
- DOF: 6 + gripper
- Payload: 750g
- Best for: Research manipulation
- Where to buy: Trossen Robotics

**Universal Robots UR3e**
- Price: ~$23,000
- DOF: 6
- Payload: 3kg
- Best for: Industrial research
- Where to buy: Universal Robots distributors

---

## Software Tools and Libraries

### Development Tools

**Visual Studio Code**
- URL: https://code.visualstudio.com/
- Cost: Free
- Extensions: ROS, Python, C++, Remote SSH
- Best for: ROS 2 development

**PyCharm**
- URL: https://www.jetbrains.com/pycharm/
- Cost: Free (Community) / Paid (Professional)
- Best for: Python-heavy projects

**CLion**
- URL: https://www.jetbrains.com/clion/
- Cost: Paid (free for students)
- Best for: C++ ROS 2 development

---

### Version Control

**Git**
- URL: https://git-scm.com/
- Description: Distributed version control
- Essential for: All software projects

**GitHub**
- URL: https://github.com/
- Description: Git hosting with CI/CD
- Best for: Open-source collaboration

**GitLab**
- URL: https://gitlab.com/
- Description: Alternative to GitHub with built-in CI/CD
- Best for: Private repositories, DevOps

---

### Visualization and Analysis

**Foxglove Studio**
- URL: https://foxglove.dev/
- Cost: Free (open-source)
- Description: Modern robotics visualization tool
- Best for: ROS bag playback, real-time debugging

**PlotJuggler**
- URL: https://github.com/facontidavide/PlotJuggler
- Cost: Free
- Description: Time-series data visualization
- Best for: Analyzing sensor data, tuning controllers

**Blender**
- URL: https://www.blender.org/
- Cost: Free
- Description: 3D modeling and animation
- Best for: Creating robot meshes, animations

---

## Simulation Assets

**NVIDIA Isaac Sim Assets**
- URL: https://docs.omniverse.nvidia.com/isaacsim/latest/features/environment_setup/assets/usd_assets_overview.html
- Description: Pre-built robots, environments, props
- Includes: Franka, UR robots, warehouses, apartments

**Gazebo Model Database**
- URL: https://github.com/osrf/gazebo_models
- Description: Collection of robot and environment models
- Includes: Common robots, furniture, objects

**TurboSquid (Commercial)**
- URL: https://www.turbosquid.com/
- Cost: Paid per asset
- Description: High-quality 3D models
- Best for: Photorealistic environments

**Sketchfab (Free & Paid)**
- URL: https://sketchfab.com/
- Cost: Many free, some paid
- Description: Community 3D models
- Best for: Props, objects, robot parts

---

## Datasets

**ImageNet**
- URL: https://www.image-net.org/
- Size: 14M+ images
- Best for: Pre-training vision models

**COCO (Common Objects in Context)**
- URL: https://cocodataset.org/
- Size: 330K images, 1.5M objects
- Best for: Object detection training

**RoboNet**
- URL: https://www.robonet.wiki/
- Size: 15M video frames from 7 robot platforms
- Best for: Manipulation research

**Something-Something V2**
- URL: https://developer.qualcomm.com/software/ai-datasets/something-something
- Size: 220K videos of hand interactions
- Best for: Action recognition, manipulation

---

## Blogs and Newsletters

**NVIDIA Robotics Blog**
- URL: https://blogs.nvidia.com/blog/category/robotics/
- Frequency: Weekly
- Content: Isaac updates, customer stories, research

**ROS News**
- URL: https://www.openrobotics.org/blog
- Frequency: Monthly
- Content: ROS releases, community highlights

**The Robot Report**
- URL: https://www.therobotreport.com/
- Frequency: Daily
- Content: Industry news, funding, product launches

**IEEE Spectrum Robotics**
- URL: https://spectrum.ieee.org/topic/robotics/
- Frequency: Daily
- Content: Research highlights, industry trends

**Robohub**
- URL: https://robohub.org/
- Frequency: Daily
- Content: Research, interviews, career advice

---

## Quick Links

### Essential Downloads
- [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html)
- [Isaac Sim](https://developer.nvidia.com/isaac-sim)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Git](https://git-scm.com/downloads)

### Package Repositories
- [ROS Index](https://index.ros.org/)
- [Isaac ROS GitHub](https://github.com/NVIDIA-ISAAC-ROS)
- [PyPI (Python)](https://pypi.org/)

### Learning Platforms
- [The Construct](https://www.theconstructsim.com/)
- [Coursera](https://www.coursera.org/search?query=robotics)
- [YouTube](https://www.youtube.com/results?search_query=ros2+tutorial)

---

**Note**: Links and prices accurate as of December 2024. Check official sources for latest information.

**Contributing**: Know a great resource not listed here? Submit a pull request to the course repository!

**Last Updated**: December 2024
