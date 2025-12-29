/**
 * Custom hook for chatbot API communication
 * Falls back to demo responses if API is unavailable
 */

import { useState, useCallback, useEffect } from 'react';

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  sources?: string[];
}

export interface ChatbotState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
  conversationId: number | null;
  isDemo: boolean;
}

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Demo responses for offline testing
const DEMO_RESPONSES: Record<string, string> = {
  default: `I'm your Physical AI & Humanoid Robotics learning assistant!

I can help you understand topics from the textbook:

**ROS 2 (Robot Operating System 2)** provides middleware for robot control with:
- Nodes: Individual computational units that communicate via messages
- Topics: Asynchronous pub/sub communication for continuous data
- Services: Synchronous request/response for one-time operations
- Actions: Long-running tasks with feedback

**URDF (Unified Robot Description Format)** describes robots with:
- Links: Physical parts (visual, collision, inertial properties)
- Joints: Connections between links (revolute, prismatic, continuous)

**Gazebo Simulation** offers:
- Physics simulation with multiple engines (ODE, Bullet, Simbody, DART)
- Sensor simulation (cameras, LIDAR, IMU)
- ROS 2 integration via gazebo_ros packages

**NVIDIA Isaac Platform** includes:
- Isaac Sim: Photorealistic simulation with RTX rendering
- Isaac ROS: Hardware-accelerated perception packages
- VSLAM and object detection capabilities

Ask me any specific question about these topics!`,

  ros2: `**ROS 2 (Robot Operating System 2)** is the next generation of ROS for robot control.

Key concepts:
- **Nodes**: Individual processes that perform computations
- **Topics**: Publish/subscribe messaging for continuous data streams
- **Services**: Request/response pattern for one-time operations
- **Actions**: Long-running tasks with goals, feedback, and results
- **QoS**: Quality of Service policies for reliable communication

Example publisher in Python:
\`\`\`python
import rclpy
from std_msgs.msg import String

class TalkerNode(rclpy.Node):
    def __init__(self):
        super().__init__('talker')
        self.publisher = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        msg = String()
        msg.data = 'Hello ROS 2!'
        self.publisher.publish(msg)
\`\`\``,

  urdf: `**URDF (Unified Robot Description Format)** is an XML format for describing robots.

Key elements:
- **<robot>**: Root element containing all links and joints
- **<link>**: Describes a physical part (visual, collision, inertial)
- **<joint>**: Describes connection between links

Joint types:
- **revolute**: Rotates within limits (like a knee)
- **continuous**: Rotates without limits (like a neck)
- **prismatic**: Linear sliding (like a telescope)
- **fixed**: Rigid connection (no movement)

For humanoids, you typically need:
- Base/waist, torso, hip joints
- Upper/lower legs with knee joints
- Arms with shoulder, elbow, wrist joints
- Head with neck joints`,

  gazebo: `**Gazebo** is a physics simulation environment for robotics.

Key features:
- **SDF Format**: Simulation Description Format for worlds and models
- **Physics Engines**: ODE, Bullet, Simbody, DART
- **Sensor Simulation**: Cameras, LIDAR, IMU, force sensors
- **ROS 2 Integration**: Via gazebo_ros packages

Workflow:
1. Create world SDF with ground, lighting, obstacles
2. Spawn robot from URDF using gazebo_ros spawn_entity
3. Configure ros2_control for joint control
4. Run navigation, manipulation, or other ROS 2 nodes`,

  navigation: `**Nav2** is the ROS 2 navigation stack for autonomous navigation.

Key components:
- **Behavior Trees**: Define navigation logic and decisions
- **Planners**: Global path planning (A*, RRT, Dijkstra)
- **Controllers**: Local execution (DWA, TEB, MPC)
- **Costmaps**: 2D/3D obstacle representation
- **Recovery**: Rotate, backup, clear obstacles

For humanoids, bipedal navigation requires:
- Footstep planning with valid contact points
- Balance maintenance during walking
- Terrain adaptation for uneven surfaces`,

  isaac: `**NVIDIA Isaac** platform for AI robotics:

**Isaac Sim**: Photorealistic simulation
- RTX rendering for photorealistic scenes
- Synthetic data generation for training
- Domain randomization for sim-to-real transfer

**Isaac ROS**: Hardware-accelerated perception
- isaac_ros_visual_slam: GPU-accelerated VSLAM
- isaac_ros_dnn_inference: TensorRT object detection
- isaac_ros_depth_segmentation: Semantic segmentation

**Isaac Gym**: Reinforcement learning for robot control`,

  humanoid: `**Humanoid robots** have human-like physical structure:

**Key challenges:**
- **Balance**: COM (Center of Mass) and ZMP (Zero Moment Point) control
- **Grasping**: Multi-finger hands with force control
- **Locomotion**: Dynamic walking over varied terrain
- **Safety**: Human-robot interaction without injury

**Common platforms:**
- Unitree G1, H1 (commercial humanoids)
- Boston Dynamics Atlas (research)
- Tesla Optimus (emerging)`,

  vla: `**Vision-Language-Action (VLA)** models combine perception and planning:

Pipeline:
1. **Speech-to-Text**: Whisper for voice commands
2. **Language Understanding**: GPT-4 for command parsing
3. **Visual Perception**: YOLO for object detection
4. **Motion Planning**: MoveIt2 for trajectory generation
5. **Execution**: Real-time feedback and correction

This enables commands like "Bring me the water bottle" to be converted into robot actions!`,

  default_lower: `I'm your Physical AI & Humanoid Robotics learning assistant!

I can help you understand:
- ROS 2 concepts (nodes, topics, services, actions)
- URDF robot descriptions
- Gazebo simulation
- Nav2 navigation
- NVIDIA Isaac platform
- Humanoid robot fundamentals
- Vision-Language-Action models

Ask me a specific question about any of these topics!`
};

function getDemoResponse(query: string): { response: string; sources: string[] } {
  const queryLower = query.toLowerCase();

  if (queryLower.includes('ros2') || queryLower.includes('ros 2') || queryLower.includes('node') || queryLower.includes('topic')) {
    return { response: DEMO_RESPONSES.ros2, sources: ['Module 1: ROS 2'] };
  }
  if (queryLower.includes('urdf') || queryLower.includes('robot description') || queryLower.includes('link') || queryLower.includes('joint')) {
    return { response: DEMO_RESPONSES.urdf, sources: ['Module 1: URDF'] };
  }
  if (queryLower.includes('gazebo') || queryLower.includes('simulation') || queryLower.includes('sdf')) {
    return { response: DEMO_RESPONSES.gazebo, sources: ['Module 2: Simulation'] };
  }
  if (queryLower.includes('navigation') || queryLower.includes('nav2') || queryLower.includes('path') || queryLower.includes('planner')) {
    return { response: DEMO_RESPONSES.navigation, sources: ['Module 3: Nav2'] };
  }
  if (queryLower.includes('isaac') || queryLower.includes('nvidia') || queryLower.includes('vslam') || queryLower.includes('tensorrt')) {
    return { response: DEMO_RESPONSES.isaac, sources: ['Module 3: NVIDIA Isaac'] };
  }
  if (queryLower.includes('humanoid') || queryLower.includes('walking') || queryLower.includes('balance') || queryLower.includes('locomotion')) {
    return { response: DEMO_RESPONSES.humanoid, sources: ['Module 4: Humanoid'] };
  }
  if (queryLower.includes('vla') || queryLower.includes('vision') || queryLower.includes('language') || queryLower.includes('action') || queryLower.includes('voice')) {
    return { response: DEMO_RESPONSES.vla, sources: ['Module 4: VLA'] };
  }

  return { response: DEMO_RESPONSES.default_lower, sources: ['Course Introduction'] };
}

function getUserId(): string {
  if (typeof window === 'undefined') {
    return 'anonymous';
  }
  let userId = localStorage.getItem('chatbot_user_id');
  if (!userId) {
    userId = `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem('chatbot_user_id', userId);
  }
  return userId;
}

export const useChatbot = () => {
  const [state, setState] = useState<ChatbotState>({
    messages: [],
    isLoading: false,
    error: null,
    conversationId: null,
    isDemo: false,
  });

  // Check API availability on mount
  useEffect(() => {
    const checkApi = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`, {
          signal: AbortSignal.timeout(2000)
        });
        if (!response.ok) {
          throw new Error('API not available');
        }
      } catch {
        console.log('Chatbot running in demo mode (no API)');
        setState(prev => ({ ...prev, isDemo: true }));
      }
    };
    checkApi();
  }, []);

  const sendMessage = useCallback(
    async (message: string, chapterContext?: string, selectedText?: string) => {
      const userMessage: Message = {
        role: 'user',
        content: message,
        timestamp: new Date(),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        isLoading: true,
        error: null,
      }));

      try {
        const response = await fetch(`${API_BASE_URL}/api/chat/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            user_id: getUserId(),
            message,
            chapter_context: chapterContext,
            selected_text: selectedText,
          }),
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const data = await response.json();

        const assistantMessage: Message = {
          role: 'assistant',
          content: data.response,
          timestamp: new Date(),
          sources: data.sources || [],
        };

        setState((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
          isLoading: false,
        }));
      } catch (error) {
        console.log('API unavailable, using demo mode');

        // Use demo response
        const demo = getDemoResponse(message);
        const assistantMessage: Message = {
          role: 'assistant',
          content: demo.response,
          timestamp: new Date(),
          sources: demo.sources,
        };

        setState((prev) => ({
          ...prev,
          messages: [...prev.messages, assistantMessage],
          isLoading: false,
          isDemo: true,
        }));
      }
    },
    []
  );

  const clearChat = useCallback(() => {
    setState({
      messages: [],
      isLoading: false,
      error: null,
      conversationId: null,
      isDemo: state.isDemo,
    });
  }, [state.isDemo]);

  return {
    messages: state.messages,
    isLoading: state.isLoading,
    error: state.error,
    sendMessage,
    clearChat,
    isDemo: state.isDemo,
  };
};
