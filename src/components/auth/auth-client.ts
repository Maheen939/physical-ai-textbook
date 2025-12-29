/**
 * Physical AI Textbook - Better-Auth Client Configuration
 *
 * This module configures Better-Auth for user authentication
 * with profiling for personalized learning experiences.
 */

import { createAuth } from 'better-auth';
import { betterFetch } from '@better-fetch/fetch';
import type { Session, User } from 'better-auth';

export const auth = createAuth({
  baseURL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  fetch: betterFetch,
  plugins: [],
});

export type { Session, User };

// Auth configuration
export const authConfig = {
  // Enable these plugins in production
  plugins: {
    // emailAndPassword: true, // Enable email/password auth
    // oAuth: { // Enable OAuth providers
    //   google: {
    //     clientId: process.env.GOOGLE_CLIENT_ID,
    //     clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    //   },
    //   github: {
    //     clientId: process.env.GITHUB_CLIENT_ID,
    //     clientSecret: process.env.GITHUB_CLIENT_SECRET,
    //   },
    // },
  },
};

// User profile types for personalization
export interface UserProfile {
  programmingLevel: 'beginner' | 'intermediate' | 'advanced';
  roboticsExperience: 'none' | 'academic' | 'professional';
  ros2Familiarity: 'none' | 'basic' | 'intermediate' | 'expert';
  learningGoal: 'understanding' | 'hands-on' | 'career';
  hardwareAccess: 'simulation' | 'jetson' | 'full-lab';
}

export interface UserWithProfile extends User {
  profile?: UserProfile;
}

// Profiling questions for signup
export const profilingQuestions = [
  {
    key: 'programmingLevel',
    question: 'What is your programming experience level?',
    options: [
      { value: 'beginner', label: 'Beginner (0-1 years)' },
      { value: 'intermediate', label: 'Intermediate (1-3 years)' },
      { value: 'advanced', label: 'Advanced (3+ years)' },
    ],
  },
  {
    key: 'roboticsExperience',
    question: 'What is your prior robotics experience?',
    options: [
      { value: 'none', label: 'No experience' },
      { value: 'academic', label: 'Academic (courses, research)' },
      { value: 'professional', label: 'Professional (work experience)' },
    ],
  },
  {
    key: 'ros2Familiarity',
    question: 'How familiar are you with ROS 2?',
    options: [
      { value: 'none', label: 'Never used it' },
      { value: 'basic', label: 'Basic (tried some tutorials)' },
      { value: 'intermediate', label: 'Intermediate (built some nodes)' },
      { value: 'expert', label: 'Expert (production experience)' },
    ],
  },
  {
    key: 'learningGoal',
    question: 'What is your primary learning goal?',
    options: [
      { value: 'understanding', label: 'Understanding concepts' },
      { value: 'hands-on', label: 'Building projects' },
      { value: 'career', label: 'Career transition' },
    ],
  },
  {
    key: 'hardwareAccess',
    question: 'What hardware do you have access to?',
    options: [
      { value: 'simulation', label: 'Simulation only (no hardware)' },
      { value: 'jetson', label: 'NVIDIA Jetson Orin' },
      { value: 'full-lab', label: 'Full robotics lab' },
    ],
  },
];
