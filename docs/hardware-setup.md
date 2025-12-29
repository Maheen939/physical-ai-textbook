# Hardware Setup

This page outlines the hardware requirements for the Physical AI & Humanoid Robotics course and provides alternatives for different budgets.

## Minimum Requirements

### Option 1: Local Workstation (Recommended)

**"Digital Twin" Workstation** - Required for simulation

| Component | Specification | Purpose |
|-----------|---------------|---------|
| **GPU** | NVIDIA RTX 4070 Ti (12GB VRAM) minimum | Isaac Sim, Gazebo rendering, AI inference |
| **CPU** | Intel Core i7 (13th Gen+) or AMD Ryzen 9 | Physics simulation |
| **RAM** | 64 GB DDR5 (32 GB absolute minimum) | Complex scene rendering |
| **Storage** | 512 GB NVMe SSD | Fast I/O for simulation assets |
| **OS** | Ubuntu 22.04 LTS | Native ROS 2 support |

**Why These Requirements?**
- NVIDIA Isaac Sim requires RTX GPUs for ray tracing
- Gazebo physics simulation is CPU-intensive
- Large scene files and robot models need significant RAM

### Option 2: Cloud Workstation (Budget Alternative)

If you don't have access to an RTX-capable workstation:

**AWS EC2 Instance**
- **Instance Type**: g5.2xlarge (A10G GPU, 24GB VRAM)
- **Cost**: ~$1.50/hour spot pricing
- **Usage**: 10 hours/week × 13 weeks = ~$195 total
- **Setup**: Use NVIDIA Isaac Sim AMI

**Other Cloud Providers**
- Azure: NC-series VMs
- Google Cloud: N1 with NVIDIA T4/A10G
- Paperspace: GPU instances with Ubuntu

## Optional: Edge AI Kit

For Module 3 (NVIDIA Isaac) and real robot deployment:

| Component | Model | Price | Purpose |
|-----------|-------|-------|---------|
| **Edge Computer** | NVIDIA Jetson Orin Nano Super (8GB) | $249 | Edge AI inference |
| **Depth Camera** | Intel RealSense D435i | $349 | RGB-D perception |
| **Microphone** | ReSpeaker USB Mic Array v2.0 | $69 | Voice commands |
| **Storage** | 128GB microSD (high-endurance) | $30 | OS and data |
| **Total** | | **~$700** | Complete edge kit |

**Note**: This is optional. You can complete the course using simulation only.

## Optional: Physical Robot

For hands-on physical AI experimentation:

### Budget Option
- **Hiwonder TonyPi Pro**: ~$600 (tabletop humanoid)
- Limited capabilities, good for learning kinematics

### Professional Option
- **Unitree Go2**: $1,800-$3,000 (quadruped robot)
- Excellent ROS 2 support, durable, industry-standard

### Premium Option
- **Unitree G1 Humanoid**: ~$16,000
- True bipedal humanoid, dynamic walking, full SDK

**Note**: Physical robots are NOT required for this course. All exercises can be completed in simulation.

## Cloud-Native Architecture

If using cloud workstations:

```
Local Machine (Laptop)
    ↓ SSH/VNC
Cloud Instance (g5.2xlarge)
    ├─ NVIDIA Isaac Sim
    ├─ Gazebo
    ├─ ROS 2 Environment
    └─ Code Development

Optional: Jetson Orin Nano (for final deployment)
```

**Workflow**:
1. Develop and simulate in cloud
2. Download trained models
3. Deploy to local Jetson (if available)

## Hybrid Approach (Recommended for Most Students)

- **Simulation**: Cloud workstation (AWS g5.2xlarge)
- **Edge Deployment**: NVIDIA Jetson Orin Nano kit ($700)
- **Physical Robot**: Skip or use shared lab resources

**Total Cost**: ~$900 (Jetson kit + ~$200 cloud credits)

## Software Requirements

See [Software Setup](./software-setup.md) for detailed installation instructions for:
- Ubuntu 22.04 LTS
- ROS 2 (Humble/Iron)
- Gazebo Sim
- NVIDIA Isaac Sim
- Python development environment

## Verification Script

After hardware setup, run this verification script:

```bash
./code_examples/module-1/chapter-01/check_environment.sh
```

This will verify:
- GPU capabilities (RTX support)
- CUDA installation
- RAM and storage
- Ubuntu version

## Need Help?

- **Hardware Issues**: See [Troubleshooting](./appendix/troubleshooting.md)
- **Budget Constraints**: Contact course instructors for lab access options
- **Cloud Setup**: Detailed guides in each module's setup section

## Next Steps

Proceed to [Software Setup](./software-setup.md) to install required software.
