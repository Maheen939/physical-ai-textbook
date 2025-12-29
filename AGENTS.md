# Claude Code Agents & Skills

This document describes the reusable intelligence features implemented for the Physical AI & Humanoid Robotics textbook project.

## Overview

The project includes custom Claude Code agents and skills for automating common tasks related to robotics education and development. These are worth **50 bonus points** in the hackathon.

---

## Claude Code Subagents

Custom agents are defined in `.claude/agents/` and provide specialized capabilities for robotics development.

### 1. ROS 2 Code Generator

**File**: `.claude/agents/ros2-code-generator.json`

Generates production-ready ROS 2 nodes, packages, and launch files from specifications.

**Usage**:
```bash
# Generate a publisher node
/sp.generate_ros2_node --node_type publisher --package_name my_package --node_name my_publisher --topic_name /chatter --message_type std_msgs/msg/String
```

**Generates**:
- `package.xml` and `setup.py`
- Python node with rclpy
- Launch files
- Custom message definitions

### 2. URDF Validator

**File**: `.claude/agents/urdf-validator.json`

Validates and generates URDF/XACRO robot description files for humanoid robots.

**Usage**:
```bash
# Validate existing URDF
/sp.validate_urdf --urdf_content "..." --robot_type humanoid

# Generate new humanoid URDF
/sp.generate_urdf --robot_type humanoid --sensors ["camera", "lidar"]
```

**Checks**:
- XML syntax validity
- Joint type correctness
- Inertial properties
- TF tree connectivity

### 3. Gazebo World Builder

**File**: `.claude/agents/gazebo-world-builder.json`

Generates SDF world files for Gazebo simulation environments.

**Usage**:
```bash
# Create an obstacle course world
/sp.build_world --world_type obstacle_course --physics_engine ode --include_robot true
```

**World Types**:
- `empty` - Basic empty world
- `indoor` - Indoor environment with walls
- `outdoor` - Outdoor terrain
- `obstacle_course` - Testing environment with obstacles
- `lab` - Robotics lab environment

---

## Claude Code Skills

Reusable skills are defined in `.claude/skills/` for common operations.

### Skill 1: Generate ROS 2 Launch File

**File**: `.claude/skills/generate-ros2-launch.json`

Creates comprehensive `launch.py` files for ROS 2 packages.

**Usage**:
```bash
/sp.generate_ros2_launch --package_name my_robot --nodes '[{"name": "publisher", "executable": "talker"}, {"name": "subscriber", "executable": "listener"}]'
```

### Skill 2: Create Docusaurus Chapter

**File**: `.claude/skills/create-docusaurus-chapter.json`

Generates complete textbook chapters in Docusaurus markdown format.

**Usage**:
```bash
/sp.create_chapter --chapter_title "Introduction to ROS 2" --module 1 --chapter_number 1 --learning_objectives '["Understand ROS 2 concepts", "Create first node"]'
```

### Skill 3: Test Code Example

**File**: `.claude/skills/test-code-example.json`

Validates and tests code examples from the textbook.

**Usage**:
```bash
/sp.test_code --code "..." --language python --test_type syntax
```

---

## Integration with Spec-Kit Plus

These agents work seamlessly with Spec-Kit Plus commands:

```bash
/sp.constitution    # View content quality principles
/sp.specify         # Create chapter specifications
/sp.plan            # Plan implementation details
/sp.implement       # Generate content with agents
```

---

## Directory Structure

```
.claude/
├── agents/
│   ├── ros2-code-generator.json
│   ├── urdf-validator.json
│   └── gazebo-world-builder.json
├── skills/
│   ├── generate-ros2-launch.json
│   ├── create-docusaurus-chapter.json
│   └── test-code-example.json
└── skills.json
```

---

## Scoring

This feature contributes **50 bonus points** to the hackathon score:

- 3 custom agents implemented (ROS 2, URDF, Gazebo)
- 3 reusable skills for common tasks
- Full documentation in AGENTS.md
- Integration with Spec-Kit Plus workflow

---

## Examples

### Example 1: Generate a Complete ROS 2 Package

```bash
/sp.generate_ros2_node \
  --node_type full_node \
  --package_name humanoid_control \
  --node_name joint_controller \
  --topic_name /joint_commands \
  --message_type geometry_msgs/msg/Twist
```

### Example 2: Create Humanoid URDF with Sensors

```bash
/sp.generate_urdf \
  --robot_type humanoid \
  --sensors '["camera", "lidar", "imu"]' \
  --joint_config '[{"name": "head_yaw", "type": "revolute"}, {"name": "arm_lift", "type": "prismatic"}]'
```

### Example 3: Build Simulation World

```bash
/sp.build_world \
  --world_type obstacle_course \
  --physics_engine ode \
  --include_robot true \
  --obstacles '["box", "cylinder", "wall"]'
```

---

## Best Practices

1. **Use Type Hints**: All generated code includes Python type hints
2. **Error Handling**: Comprehensive error handling with logging
3. **Documentation**: Inline comments and README files for each generated component
4. **Testing**: Each agent generates testable, working code

---

## Resources

- [Claude Code Documentation](https://docs.claude.com/)
- [ROS 2 Documentation](https://docs.ros.org/)
- [Gazebo Simulation](https://gazebosim.org/)
- [URDF Documentation](https://wiki.ros.org/urdf)
