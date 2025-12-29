"""
Demo Chatbot Service - Works without API keys for testing
Simulates RAG chatbot responses for the Physical AI textbook
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import json

router = APIRouter()

# Knowledge base for demo responses
KNOWLEDGE_BASE = {
    "ros2": """ROS 2 (Robot Operating System 2) is the next generation of ROS.
It provides middleware for robot control with nodes, topics, services, and actions.
Key concepts:
- Nodes: Individual computational units that communicate via messages
- Topics: Asynchronous pub/sub communication for continuous data streams
- Services: Synchronous request/response for one-time operations
- Actions: Long-running tasks with feedback and goal tracking
- QoS: Quality of Service policies for reliable communication""",

    "urdf": """URDF (Unified Robot Description Format) is an XML format for
describing robots in ROS. Key elements:
- <link>: Describes a physical part of the robot (visual, collision, inertial)
- <joint>: Describes connection between links (revolute, prismatic, continuous)
- <robot>: Root element containing all links and joints
- Xacro: Preprocessor for parameterized, modular URDF files

For humanoids, typical structure includes:
- Base/waist link
- Torso with hip joints
- Upper/lower legs with knee joints
- Arms with shoulder, elbow, wrist joints
- Head with neck joints""",

    "gazebo": """Gazebo is a physics simulation environment for robotics.
Key features:
- SDF (Simulation Description Format) for world and model definitions
- Physics engines: ODE, Bullet, Simbody, DART
- Sensor simulation: cameras, LIDAR, IMU, force sensors
- ROS 2 integration via gazebo_ros packages
- Plugins for custom sensor and actuator simulation

Common workflows:
1. Create world SDF file with ground, lighting, obstacles
2. Spawn robot from URDF using gazebo_ros spawn_entity
3. Configure ros2_control for joint control
4. Run navigation, manipulation, or other ROS 2 nodes""",

    "navigation": """Nav2 is the ROS 2 navigation stack for autonomous robot navigation.
Key components:
- Behavior Trees: Define navigation logic and decision making
- Planners: Global path planning (A*, RRT, Dijkstra)
- Controllers: Local execution (DWA, TEB, MPC)
- Costmaps: 2D/3D representation of obstacles
- Recovery behaviors: Rotate, backup, clear obstacles

For humanoids, bipedal navigation requires:
- Footstep planning with valid contact points
- Balance maintenance during walking
- Terrain adaptation for uneven surfaces
- Fall detection and recovery strategies""",

    "isaac": """NVIDIA Isaac is a powerful platform for AI robotics:
- Isaac Sim: Photorealistic simulation with RTX rendering
- Isaac ROS: Hardware-accelerated perception packages
- Isaac SDK: Robotics algorithms and utilities

Key Isaac ROS packages:
- isaac_ros_visual_slam: VSLAM with GPU acceleration
- isaac_ros_dnn_inference: TensorRT-based object detection
- isaac_ros_depth_segmentation: Semantic segmentation
- isaac_ros_nvengine: Performance optimization

Isaac Sim advantages:
- Synthetic data generation for training
- Domain randomization for sim-to-real transfer
- Photorealistic rendering for computer vision""",

    "humanoid": """Humanoid robots have human-like physical structure:
- Bipedal locomotion: Walking on two legs with balance control
- Manipulation: Arms and hands for object interaction
- Perception: Vision, touch, and proprioception sensors
- Cognition: AI for decision making and learning

Key challenges:
- Balance: Center of Mass (COM) and Zero Moment Point (ZMP) control
- Grasping: Multi-finger hands with force control
- Locomotion: Dynamic walking over varied terrain
- Safety: Human-robot interaction without injury""",

    "vla": """Vision-Language-Action (VLA) models combine perception and planning:
- Vision: Process camera images to understand environment
- Language: Interpret natural language commands
- Action: Generate robot control commands

Integration pipeline:
1. Speech-to-text (Whisper) for voice commands
2. LLM (GPT-4) for command parsing and planning
3. Object detection (YOLO) for scene understanding
4. Motion planning (MoveIt2) for trajectory generation
5. Execution with real-time feedback and correction"""
}


class ChatQuery(BaseModel):
    message: str
    chapter_context: Optional[str] = None
    selected_text: Optional[str] = None


def find_relevant_content(query: str) -> tuple[str, List[str]]:
    """Find relevant content from knowledge base."""
    query_lower = query.lower()
    sources = []
    relevant = ""

    # Find matching topics
    for topic, content in KNOWLEDGE_BASE.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            relevant += content + "\n\n"
            sources.append(topic.title())

    if not relevant:
        # Default response
        relevant = """I'm your Physical AI & Humanoid Robotics learning assistant!

I can help you understand:
- ROS 2 concepts (nodes, topics, services, actions)
- URDF robot descriptions
- Gazebo simulation
- Nav2 navigation
- NVIDIA Isaac platform
- Humanoid robot fundamentals
- Vision-Language-Action models

Ask me any question about the textbook content, and I'll explain concepts from the course materials."""
        sources = ["Course Introduction"]

    return relevant.strip(), sources


@router.post("/api/chat/query")
async def demo_chat_query(query: ChatQuery):
    """Demo chat endpoint that simulates RAG without API keys."""
    import time
    time.sleep(0.5)  # Simulate processing time

    relevant_content, sources = find_relevant_content(query.message)

    # Generate contextual response
    response = f"""Based on the textbook content, here's what I can tell you about your question:

{relevant_content}

---

Is there a specific aspect you'd like me to explain further? You can also:

1. Highlight text in the chapter and ask about it
2. Ask for code examples
3. Request clarification on any concept"""

    return {
        "response": response,
        "sources": sources,
        "conversation_id": 1,
        "demo_mode": True
    }


@router.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "demo_mode": True,
        "services": {
            "openai": "demo",
            "qdrant": "demo",
            "neon": "demo"
        }
    }
