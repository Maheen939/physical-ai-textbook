import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  docs: [
    'intro',
    'prerequisites',
    'hardware-setup',
    'software-setup',
    {
      type: 'category',
      label: 'Module 1: The Robotic Nervous System (ROS 2)',
      collapsed: false,
      items: [
        'module-1/ros2-fundamentals',
        'module-1/ros2-communication',
        'module-1/urdf-robot-description',
      ],
    },
    {
      type: 'category',
      label: 'Module 2: The Digital Twin (Gazebo & Unity)',
      collapsed: false,
      items: [
        'module-2/gazebo-simulation',
        'module-2/sensor-simulation',
        'module-2/unity-integration',
      ],
    },
    {
      type: 'category',
      label: 'Module 3: The AI-Robot Brain (NVIDIA Isaac)',
      collapsed: false,
      items: [
        'module-3/isaac-sim-intro',
        'module-3/isaac-ros-perception',
        'module-3/nav2-navigation',
      ],
    },
    {
      type: 'category',
      label: 'Module 4: Vision-Language-Action (VLA)',
      collapsed: false,
      items: [
        'module-4/humanoid-locomotion',
        'module-4/manipulation-grasping',
        'module-4/conversational-robotics',
      ],
    },
    {
      type: 'category',
      label: 'Appendix',
      collapsed: true,
      items: [
        'appendix/troubleshooting',
        'appendix/glossary',
        'appendix/resources',
        'appendix/references',
      ],
    },
    {
      type: 'category',
      label: 'Bonus Features',
      collapsed: true,
      items: [
        'bonus/authentication',
        'bonus/personalization',
        'bonus/translation',
      ],
    },
  ],
};

export default sidebars;
