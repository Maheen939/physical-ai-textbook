/**
 * Authentication Context for Physical AI Textbook
 *
 * Provides authentication state management using Better-Auth pattern
 * with local storage for session persistence.
 */

import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

// Types
interface User {
  id: string;
  email: string;
  name: string;
  profile?: UserProfile;
}

interface UserProfile {
  programmingLevel: 'beginner' | 'intermediate' | 'advanced';
  roboticsExperience: 'none' | 'academic' | 'professional';
  ros2Familiarity: 'none' | 'basic' | 'intermediate' | 'expert';
  learningGoal: 'understanding' | 'hands-on' | 'career';
  hardwareAccess: 'simulation' | 'jetson' | 'full-lab';
}

interface AuthContextType {
  user: User | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<void>;
  signUp: (email: string, password: string, profile: UserProfile) => Promise<void>;
  signOut: () => Promise<void>;
  updateProfile: (profile: Partial<UserProfile>) => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  // Check for existing session on mount
  useEffect(() => {
    const checkSession = async () => {
      const sessionToken = localStorage.getItem('session_token');
      if (sessionToken) {
        try {
          const response = await fetch(`${API_URL}/api/auth/me?session_token=${sessionToken}`);
          if (response.ok) {
            const data = await response.json();
            setUser(data.user);
          } else {
            localStorage.removeItem('session_token');
          }
        } catch (error) {
          console.error('Session check failed:', error);
          localStorage.removeItem('session_token');
        }
      }
      setLoading(false);
    };

    checkSession();
  }, []);

  const signIn = async (email: string, password: string) => {
    const response = await fetch(`${API_URL}/api/auth/signin`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Sign in failed');
    }

    const data = await response.json();
    localStorage.setItem('session_token', data.session_token);
    setUser(data.user);
  };

  const signUp = async (email: string, password: string, profile: UserProfile) => {
    const response = await fetch(`${API_URL}/api/auth/signup`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password, profile }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Sign up failed');
    }

    const data = await response.json();
    localStorage.setItem('session_token', data.session_token);
    setUser(data.user);
  };

  const signOut = async () => {
    const sessionToken = localStorage.getItem('session_token');
    if (sessionToken) {
      try {
        await fetch(`${API_URL}/api/auth/signout`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_token: sessionToken }),
        });
      } catch (error) {
        console.error('Sign out error:', error);
      }
    }
    localStorage.removeItem('session_token');
    setUser(null);
  };

  const updateProfile = async (profileUpdates: Partial<UserProfile>) => {
    const sessionToken = localStorage.getItem('session_token');
    if (!sessionToken || !user) {
      throw new Error('Not authenticated');
    }

    const response = await fetch(`${API_URL}/api/auth/profile`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_token: sessionToken, ...profileUpdates }),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Profile update failed');
    }

    // Refresh user data
    const sessionResponse = await fetch(`${API_URL}/api/auth/me?session_token=${sessionToken}`);
    if (sessionResponse.ok) {
      const data = await sessionResponse.json();
      setUser(data.user);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        signIn,
        signUp,
        signOut,
        updateProfile,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
}

export type { User, UserProfile };
