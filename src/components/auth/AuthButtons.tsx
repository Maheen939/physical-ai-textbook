/**
 * Authentication UI Components for Physical AI Textbook
 *
 * Sign-in and Sign-up forms with profiling questions.
 */

import React, { useState } from 'react';
import { useAuth, UserProfile } from './AuthProvider';
import styles from './AuthButtons.module.css';

export function AuthButtons() {
  const { user, signOut, isAuthenticated, loading } = useAuth();

  if (loading) {
    return <div className={styles.loading}>Loading...</div>;
  }

  if (isAuthenticated) {
    return (
      <div className={styles.userMenu}>
        <span className={styles.userName}>{user?.name}</span>
        <button onClick={signOut} className={styles.signOutButton}>
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <div className={styles.authButtons}>
      <button
        onClick={() => document.getElementById('auth-modal')?.showModal()}
        className={styles.signInButton}
      >
        Sign In
      </button>
      <AuthModal />
    </div>
  );
}

export function AuthModal() {
  const [mode, setMode] = useState<'signin' | 'signup' | 'profiling'>('signin');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [profile, setProfile] = useState<UserProfile>({
    programmingLevel: 'beginner',
    roboticsExperience: 'none',
    ros2Familiarity: 'none',
    learningGoal: 'understanding',
    hardwareAccess: 'simulation',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { signIn, signUp } = useAuth();

  const handleSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signIn(email, password);
      closeModal();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign in failed');
    } finally {
      setLoading(false);
    }
  };

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signUp(email, password, profile);
      closeModal();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Sign up failed');
    } finally {
      setLoading(false);
    }
  };

  const closeModal = () => {
    const modal = document.getElementById('auth-modal') as HTMLDialogElement;
    modal?.close();
    // Reset form
    setMode('signin');
    setEmail('');
    setPassword('');
    setError('');
  };

  return (
    <dialog id="auth-modal" className={styles.modal}>
      <div className={styles.modalContent}>
        <button onClick={closeModal} className={styles.closeButton}>
          &times;
        </button>

        {mode === 'signin' && (
          <>
            <h2>Welcome Back</h2>
            <p className={styles.subtitle}>Sign in to continue learning</p>

            <form onSubmit={handleSignIn} className={styles.form}>
              <label className={styles.label}>
                Email
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className={styles.input}
                />
              </label>

              <label className={styles.label}>
                Password
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  className={styles.input}
                />
              </label>

              {error && <p className={styles.error}>{error}</p>}

              <button type="submit" disabled={loading} className={styles.submitButton}>
                {loading ? 'Signing in...' : 'Sign In'}
              </button>
            </form>

            <p className={styles.switchMode}>
              New here?{' '}
              <button onClick={() => setMode('signup')} className={styles.linkButton}>
                Create account
              </button>
            </p>
          </>
        )}

        {mode === 'signup' && (
          <>
            <h2>Create Account</h2>
            <p className={styles.subtitle}>Help us personalize your learning experience</p>

            <form onSubmit={handleSignUp} className={styles.form}>
              <label className={styles.label}>
                Email
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  required
                  className={styles.input}
                />
              </label>

              <label className={styles.label}>
                Password
                <input
                  type="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  minLength={8}
                  className={styles.input}
                />
              </label>

              <button type="submit" disabled={loading} className={styles.submitButton}>
                {loading ? 'Creating account...' : 'Continue to Profile'}
              </button>
            </form>

            <p className={styles.switchMode}>
              Already have an account?{' '}
              <button onClick={() => setMode('signin')} className={styles.linkButton}>
                Sign in
              </button>
            </p>
          </>
        )}

        {mode === 'profiling' && (
          <>
            <h2>Tell Us About Yourself</h2>
            <p className={styles.subtitle}>
              This helps us personalize content for your experience level
            </p>

            <form onSubmit={handleSignUp} className={styles.form}>
              <ProfilingForm profile={profile} setProfile={setProfile} />

              {error && <p className={styles.error}>{error}</p>}

              <button type="submit" disabled={loading} className={styles.submitButton}>
                {loading ? 'Creating account...' : 'Start Learning'}
              </button>
            </form>

            <p className={styles.switchMode}>
              <button onClick={() => setMode('signup')} className={styles.linkButton}>
                Back to account details
              </button>
            </p>
          </>
        )}
      </div>
    </dialog>
  );
}

interface ProfilingFormProps {
  profile: UserProfile;
  setProfile: (profile: UserProfile) => void;
}

function ProfilingForm({ profile, setProfile }: ProfilingFormProps) {
  const updateProfile = (key: keyof UserProfile, value: string) => {
    setProfile({ ...profile, [key]: value });
  };

  return (
    <div className={styles.profilingForm}>
      <div className={styles.question}>
        <label className={styles.questionLabel}>
          What is your programming experience level?
        </label>
        <select
          value={profile.programmingLevel}
          onChange={(e) => updateProfile('programmingLevel', e.target.value)}
          className={styles.select}
        >
          <option value="beginner">Beginner (0-1 years)</option>
          <option value="intermediate">Intermediate (1-3 years)</option>
          <option value="advanced">Advanced (3+ years)</option>
        </select>
      </div>

      <div className={styles.question}>
        <label className={styles.questionLabel}>
          What is your prior robotics experience?
        </label>
        <select
          value={profile.roboticsExperience}
          onChange={(e) => updateProfile('roboticsExperience', e.target.value)}
          className={styles.select}
        >
          <option value="none">No experience</option>
          <option value="academic">Academic (courses, research)</option>
          <option value="professional">Professional (work experience)</option>
        </select>
      </div>

      <div className={styles.question}>
        <label className={styles.questionLabel}>
          How familiar are you with ROS 2?
        </label>
        <select
          value={profile.ros2Familiarity}
          onChange={(e) => updateProfile('ros2Familiarity', e.target.value)}
          className={styles.select}
        >
          <option value="none">Never used it</option>
          <option value="basic">Basic (tried some tutorials)</option>
          <option value="intermediate">Intermediate (built some nodes)</option>
          <option value="expert">Expert (production experience)</option>
        </select>
      </div>

      <div className={styles.question}>
        <label className={styles.questionLabel}>
          What is your primary learning goal?
        </label>
        <select
          value={profile.learningGoal}
          onChange={(e) => updateProfile('learningGoal', e.target.value)}
          className={styles.select}
        >
          <option value="understanding">Understanding concepts</option>
          <option value="hands-on">Building projects</option>
          <option value="career">Career transition</option>
        </select>
      </div>

      <div className={styles.question}>
        <label className={styles.questionLabel}>
          What hardware do you have access to?
        </label>
        <select
          value={profile.hardwareAccess}
          onChange={(e) => updateProfile('hardwareAccess', e.target.value)}
          className={styles.select}
        >
          <option value="simulation">Simulation only (no hardware)</option>
          <option value="jetson">NVIDIA Jetson Orin</option>
          <option value="full-lab">Full robotics lab</option>
        </select>
      </div>
    </div>
  );
}
