# Glossary

**Physical AI and Humanoid Robotics Terminology**

This glossary provides definitions for key terms, concepts, and acronyms used throughout the Physical AI & Humanoid Robotics course. Terms are organized alphabetically for quick reference.

---

## A

**Action** - In ROS 2, a communication pattern for long-running tasks that provides feedback during execution and a final result. Consists of goal, feedback, and result messages.

**Action Server** - ROS 2 node that executes actions and provides feedback to action clients. Example: Nav2 navigation server.

**Actuation** - The mechanism by which a robot produces physical movement or force, typically through motors, servos, or hydraulics.

**AMCL (Adaptive Monte Carlo Localization)** - Probabilistic localization algorithm that uses particle filters to estimate a robot's pose within a known map.

**Articulation** - In physics simulation, a hierarchical system of rigid bodies connected by joints. Represents robot mechanisms with multiple degrees of freedom.

**Articulation Root** - The base link of an articulated system in PhysX/Isaac Sim, serving as the reference point for the entire kinematic chain.

---

## B

**Bag File** - ROS 2 data recording format (`.db3` or `.mcap`) that captures topic messages for later playback and analysis.

**Base Link** - The primary reference frame of a robot, typically located at the robot's center of mass or base platform.

**Bipedal** - Having two legs. Bipedal robots face unique balance and stability challenges compared to wheeled or quadruped robots.

---

## C

**Callback** - A function automatically executed when an event occurs, such as receiving a message on a subscribed topic.

**Center of Mass (CoM)** - The point where the mass of an object is concentrated. Critical for balance in humanoid robots.

**Center of Pressure (CoP)** - The point on the ground where the sum of all contact forces acts. Important for stability analysis.

**Collision Mesh** - Simplified geometric representation of an object used for efficient physics collision detection (distinct from visual mesh).

**Colcon** - Build tool for ROS 2 workspaces, replacing catkin from ROS 1.

**Controller** - Algorithm or system that computes commands to actuators based on desired behavior and sensor feedback. Examples: PID controller, trajectory controller.

**Coordinate Frame** - A reference system defining position and orientation in 3D space. Robots use multiple coordinate frames (world, base, joint, etc.).

**Costmap** - 2D grid representation of the environment used by navigation systems, where each cell has a cost representing traversability.

**CUDA (Compute Unified Device Architecture)** - NVIDIA's parallel computing platform enabling GPU-accelerated computation.

---

## D

**DDS (Data Distribution Service)** - Middleware standard used by ROS 2 for real-time, peer-to-peer communication between nodes.

**Denavit-Hartenberg (DH) Parameters** - Standard method for defining coordinate frames and transformations in robot kinematics using four parameters per joint.

**Depth Camera** - Camera that measures distance to each pixel, producing a depth map or point cloud. Example: Intel RealSense D435i.

**Differential Drive** - Mobile robot configuration with two independently driven wheels, enabling rotation by varying wheel speeds.

**DOF (Degrees of Freedom)** - Number of independent parameters defining a system's configuration. A humanoid robot typically has 20-40+ DOF.

**Domain Randomization** - Training technique that varies simulation parameters (lighting, textures, physics) to improve sim-to-real transfer.

**DWB (Dynamic Window Approach)** - Local trajectory planner used in Nav2 that samples velocity commands within kinematic constraints.

**Dynamics** - Study of forces and torques causing motion. Robot dynamics relates joint torques to accelerations.

---

## E

**Embodied Intelligence** - Intelligence that emerges from the interaction of computation (brain), physical structure (body), and environment.

**End Effector** - The device at the end of a robotic manipulator, such as a gripper, tool, or sensor.

**EKF (Extended Kalman Filter)** - Algorithm for state estimation that combines predictions with noisy sensor measurements.

---

## F

**FastDDS (Fast RTPS)** - Popular DDS implementation used as ROS 2 middleware, developed by eProsima.

**Force/Torque Sensor** - Device measuring forces and torques applied to it, commonly used in wrists for force control.

**Forward Kinematics** - Computing end effector position/orientation from joint angles. Contrast with inverse kinematics.

**Frame** - See Coordinate Frame.

---

## G

**Gazebo** - 3D robot simulator with physics engine, sensor simulation, and ROS integration. Gazebo Classic uses OGRE rendering; Gazebo Sim uses newer architecture.

**Grasp Planning** - Algorithm for computing gripper poses and configurations to successfully pick up objects.

**Ground Truth** - True, accurate value of a quantity, used as reference for evaluating estimates or detections.

---

## H

**Humanoid** - Robot with human-like form, typically featuring bipedal legs, arms, torso, and head.

**HRBP (Humanoid Robot Planning Problem)** - Challenge of planning stable, collision-free motions for humanoid robots under balance constraints.

---

## I

**IMU (Inertial Measurement Unit)** - Sensor combining accelerometer and gyroscope to measure linear acceleration and angular velocity.

**Inertia** - Resistance of an object to changes in motion. Inertia tensor describes rotational inertia in 3D.

**Inverse Kinematics (IK)** - Computing joint angles required to achieve desired end effector pose. Computationally harder than forward kinematics.

**Isaac GEM (GPU-accelerated Extension Modules)** - High-performance algorithms in NVIDIA Isaac SDK.

**Isaac ROS** - Collection of GPU-accelerated ROS 2 packages for perception, navigation, and manipulation.

**Isaac Sim** - NVIDIA's photorealistic robot simulator built on Omniverse platform with RTX ray tracing and PhysX.

---

## J

**Jacobian** - Matrix relating joint velocities to end effector velocity. Essential for velocity control and singularity analysis.

**Joint** - Connection between two links allowing relative motion. Types: revolute (rotational), prismatic (linear), fixed, continuous, planar, spherical.

**Joint Space** - Configuration space where robot state is represented by joint angles/positions.

**Joint State** - Current position, velocity, and effort (torque/force) of all robot joints.

---

## K

**Kinematics** - Study of motion without considering forces. Describes geometric relationships between joints and end effector.

**Kinematic Chain** - Series of rigid bodies (links) connected by joints forming a robot mechanism.

---

## L

**Launch File** - Python or XML file specifying how to start multiple ROS 2 nodes with specific parameters and configurations.

**LIDAR (Light Detection and Ranging)** - Sensor using laser pulses to measure distances, creating 2D or 3D point clouds.

**Link** - Rigid body element in a robot, connected to other links via joints.

**Localization** - Process of determining robot's position and orientation in an environment.

---

## M

**Manipulation** - Control of a robot arm/hand to interact with objects through grasping, placing, pushing, etc.

**Map** - Representation of environment used for navigation. Types include occupancy grids, point clouds, and semantic maps.

**Motion Planning** - Computing collision-free path from current to goal configuration, considering kinematic and dynamic constraints.

**MoveIt2** - Motion planning framework for ROS 2 providing IK, collision checking, trajectory planning, and control.

---

## N

**Nav2 (Navigation 2)** - Navigation framework for ROS 2 providing path planning, control, recovery behaviors, and waypoint following.

**Node** - Independent process in ROS 2 that performs computation. Nodes communicate via topics, services, and actions.

**Nucleus** - Asset database and collaboration server in NVIDIA Omniverse ecosystem.

---

## O

**Occupancy Grid** - 2D grid map where each cell represents probability of obstacle presence.

**Odometry** - Estimate of robot position/velocity over time based on motion sensors (wheel encoders, IMU, visual odometry).

**OMPL (Open Motion Planning Library)** - Library of sampling-based motion planning algorithms used by MoveIt2.

**Omniverse** - NVIDIA's platform for 3D collaboration and simulation, foundation for Isaac Sim.

---

## P

**Package** - Organizational unit in ROS 2 containing nodes, launch files, configuration, and dependencies.

**Path Planning** - Computing sequence of configurations connecting start to goal, avoiding obstacles.

**PhysX** - NVIDIA's physics engine providing GPU-accelerated rigid body dynamics, articulations, and collision detection.

**Physical AI** - AI systems that operate in the physical world through embodiment, requiring perception, planning, and actuation.

**PID Controller** - Proportional-Integral-Derivative controller using error feedback for control. Widely used despite simplicity.

**Planner** - Algorithm computing paths or trajectories. Examples: A*, RRT, PRM, TEB.

**Point Cloud** - Set of 3D points representing object or environment surfaces, typically from LIDAR or depth cameras.

**Pose** - Position (x, y, z) and orientation (roll, pitch, yaw or quaternion) of an object.

**Prim (Primitive)** - Basic element in USD scene graph, representing geometry, transforms, cameras, lights, etc.

**Proprioception** - Sense of body configuration and state. In robots, knowledge of joint angles and velocities.

---

## Q

**QoS (Quality of Service)** - ROS 2 communication policies specifying reliability, durability, history, and other message delivery characteristics.

**Quaternion** - Four-component representation of 3D rotation, avoiding gimbal lock issues of Euler angles. Format: (w, x, y, z).

---

## R

**RealSense** - Intel's depth camera product line, popular in robotics. D435i includes RGB-D camera and IMU.

**Replicator** - NVIDIA tool for synthetic data generation in Isaac Sim, producing labeled training datasets.

**RGB-D** - Image format combining color (RGB) and depth (D) information at each pixel.

**RNDF (Road Network Definition File)** - Format for defining navigable areas, used in some autonomous vehicle systems.

**Robot State Publisher** - ROS 2 node that publishes transform tree of robot based on URDF and joint states.

**ROS (Robot Operating System)** - Middleware and framework for robot software development. ROS 2 is the modern version with real-time capabilities.

**ROS 2** - Second generation of ROS, supporting real-time, embedded systems, multiple platforms, and modern DDS communication.

**RVDML (Robot Vision and Decision Making Language)** - Specialized language for robot vision processing (less common).

**RViz2** - 3D visualization tool for ROS 2, displaying sensor data, robot models, paths, and debug information.

---

## S

**SDF (Simulation Description Format)** - XML format for describing simulation worlds, robots, and physics in Gazebo.

**Sensor Fusion** - Combining data from multiple sensors to produce more accurate state estimates than any single sensor.

**Service** - ROS 2 communication pattern for request-reply interactions. Client sends request, server processes and returns response.

**Sim-to-Real Transfer** - Process of deploying policies/models trained in simulation to physical robots.

**Singularity** - Configuration where robot Jacobian loses rank, causing loss of controllability in certain directions.

**SLAM (Simultaneous Localization and Mapping)** - Building map while simultaneously localizing within it.

**Subscriber** - ROS 2 entity that receives messages published to a topic.

---

## T

**Task Space** - Cartesian space where end effector position/orientation is specified, as opposed to joint space.

**TF2 (Transform Library 2)** - ROS 2 library managing coordinate frame relationships and transformations over time.

**Topic** - Named bus for asynchronous message passing in ROS 2. Publishers send messages, subscribers receive them.

**Trajectory** - Time-parameterized path specifying position, velocity, and acceleration at each time point.

**Transform** - Relationship between two coordinate frames, consisting of translation and rotation.

---

## U

**URDF (Unified Robot Description Format)** - XML format describing robot kinematics, dynamics, visualization, and collision properties.

**USD (Universal Scene Description)** - Pixar's framework for 3D scene composition, used by Isaac Sim and Omniverse.

**USDA** - USD ASCII format, human-readable text files.

**USDC** - USD Crate format, binary files for efficient loading.

---

## V

**Visual Odometry** - Estimating robot motion by analyzing sequential camera images.

**VLA (Vision-Language-Action)** - Model that maps visual input and language instructions to robot actions.

**VSLAM (Visual SLAM)** - SLAM using camera images as primary sensor input.

**Vulkan** - Cross-platform graphics and compute API, used by modern simulators.

---

## W

**Waypoint** - Intermediate goal position along a path or trajectory.

**Workspace** - Set of all end effector poses reachable by a manipulator. Dexterous workspace allows all orientations.

**Wrench** - Combined force and torque vector, used in force control and dynamics.

---

## X

**Xacro (XML Macros)** - Extension to URDF adding macros, constants, and mathematical expressions for more maintainable robot descriptions.

---

## Y

**YAML** - Human-readable data serialization language used for ROS 2 parameter files and configuration.

**Yaw** - Rotation around vertical (z) axis, one of three Euler angles (roll, pitch, yaw).

**YOLO (You Only Look Once)** - Family of real-time object detection models. YOLOv8 is commonly used in robotics.

---

## Z

**Zero Moment Point (ZMP)** - Point on ground where sum of all moments equals zero. Used to ensure bipedal robot stability.

**Z-up** - Coordinate convention where z-axis points upward (vertical). Alternative is y-up. ROS uses z-up.

---

## Common Acronyms

- **API** - Application Programming Interface
- **CAD** - Computer-Aided Design
- **CLI** - Command Line Interface
- **CoM** - Center of Mass
- **DAE** - COLLADA (Digital Asset Exchange)
- **DH** - Denavit-Hartenberg
- **DOF** - Degrees of Freedom
- **FPS** - Frames Per Second
- **GUI** - Graphical User Interface
- **HRI** - Human-Robot Interaction
- **IK** - Inverse Kinematics
- **IMU** - Inertial Measurement Unit
- **LLM** - Large Language Model
- **ML** - Machine Learning
- **ODE** - Open Dynamics Engine
- **OMPL** - Open Motion Planning Library
- **PID** - Proportional-Integral-Derivative
- **RGBD** - Red-Green-Blue-Depth
- **ROS** - Robot Operating System
- **RRT** - Rapidly-exploring Random Tree
- **RTX** - Ray Tracing Texel eXtreme (NVIDIA)
- **SDK** - Software Development Kit
- **SLAM** - Simultaneous Localization and Mapping
- **STL** - Stereolithography (3D mesh format)
- **TF** - Transform
- **TOPS** - Tera Operations Per Second
- **URDF** - Unified Robot Description Format
- **USD** - Universal Scene Description
- **VLA** - Vision-Language-Action
- **VRAM** - Video Random Access Memory
- **VSLAM** - Visual SLAM
- **WSL** - Windows Subsystem for Linux
- **XML** - Extensible Markup Language
- **ZMP** - Zero Moment Point

---

## Mathematical Terms

**Covariance** - Measure of how two variables vary together, used in uncertainty representation.

**Denavit-Hartenberg Convention** - Standard for assigning coordinate frames to robot links.

**Eigenvalue** - Scalar representing how much eigenvector is scaled by linear transformation. Used in stability analysis.

**Euler Angles** - Three angles (roll, pitch, yaw) representing 3D rotation. Suffers from gimbal lock.

**Gaussian Distribution** - Normal probability distribution, commonly used for sensor noise models.

**Homogeneous Transform** - 4×4 matrix encoding both rotation and translation for 3D transforms.

**Jacobian Matrix** - Matrix of partial derivatives relating input velocities to output velocities.

**Linear Algebra** - Mathematical framework for vectors, matrices, and transformations essential to robotics.

**Matrix** - Rectangular array of numbers used to represent transforms, system dynamics, etc.

**Optimization** - Finding parameter values that minimize or maximize objective function, used in motion planning and control.

**Particle Filter** - Monte Carlo method for state estimation using weighted samples.

**Probability Distribution** - Function describing likelihood of different outcomes.

**Rotation Matrix** - 3×3 orthogonal matrix representing rotation in 3D space.

**Vector** - Ordered array of numbers representing direction and magnitude in space.

---

## Physics Terms

**Angular Velocity** - Rate of rotation around an axis, measured in radians/second.

**Centripetal Force** - Force directed toward center of circular motion.

**Damping** - Dissipation of energy in mechanical systems, reducing oscillations.

**Friction** - Resistance to relative motion between surfaces in contact. Types: static, kinetic.

**Gravity** - Acceleration due to Earth's gravitational field (9.81 m/s² downward).

**Inertial Frame** - Non-accelerating reference frame where Newton's laws hold.

**Mass** - Measure of inertia; resistance to acceleration.

**Momentum** - Product of mass and velocity; conserved in closed systems.

**Stiffness** - Resistance to deformation under applied force.

**Torque** - Rotational force; product of force and moment arm.

**Velocity** - Rate of change of position; vector with magnitude and direction.

---

## Control Theory Terms

**Closed-Loop Control** - Control using feedback from sensors to adjust commands.

**Disturbance** - External input affecting system behavior (wind, collisions, etc.).

**Feedback** - Using system output to adjust input for desired behavior.

**Feedforward** - Using knowledge of desired trajectory to precompute control commands.

**Gain** - Scaling factor in controller determining response strength.

**Open-Loop Control** - Control without feedback, executing predetermined commands.

**Setpoint** - Desired value that controller attempts to maintain.

**Stability** - Property where system returns to equilibrium after disturbances.

**Steady-State Error** - Persistent difference between desired and actual values after transients decay.

**Transfer Function** - Mathematical relationship between input and output in frequency domain.

---

## Machine Learning Terms

**Batch Size** - Number of samples processed together during training.

**CNN (Convolutional Neural Network)** - Neural network architecture particularly effective for image processing.

**Dataset** - Collection of labeled examples used for training and evaluation.

**Inference** - Using trained model to make predictions on new data.

**Overfitting** - Model performs well on training data but poorly on new data.

**Reinforcement Learning** - Training agents through trial-and-error using rewards.

**Supervised Learning** - Training with labeled input-output pairs.

**Training** - Process of adjusting model parameters to minimize loss function.

**Transfer Learning** - Leveraging model trained on one task for different but related task.

---

## Quick Reference: Common Units

- **Distance**: meters (m), millimeters (mm)
- **Angle**: radians (rad), degrees (°)
- **Velocity**: meters/second (m/s), radians/second (rad/s)
- **Acceleration**: m/s², rad/s²
- **Force**: Newtons (N)
- **Torque**: Newton-meters (N⋅m)
- **Mass**: kilograms (kg)
- **Time**: seconds (s), milliseconds (ms)
- **Frequency**: Hertz (Hz)
- **Power**: Watts (W)
- **Compute**: TOPS (Tera Operations Per Second), FLOPS (Floating Point Operations Per Second)

---

**Note**: This glossary covers core terminology from the Physical AI & Humanoid Robotics course. For specialized topics, consult official documentation for ROS 2, Isaac Sim, and related technologies.

**Last Updated**: December 2024
