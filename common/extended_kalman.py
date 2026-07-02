
from __future__ import annotations

import numpy as np


class KalmanFilter:
    """Standard linear Kalman filter.

    x_{k+1} = F x_k + B u_k + w,   w ~ N(0, Q)
    z_k     = H x_k + v,           v ~ N(0, R)
    """

    def __init__(self, F, H, Q, R, x0, P0, B=None):
        self.F = np.asarray(F, float)
        self.H = np.asarray(H, float)
        self.Q = np.asarray(Q, float)
        self.R = np.asarray(R, float)
        self.B = None if B is None else np.asarray(B, float)
        self.x = np.asarray(x0, float).reshape(-1, 1)
        self.P = np.asarray(P0, float)

    def predict(self, u=None):
        self.x = self.F @ self.x
        if self.B is not None and u is not None:
            self.x = self.x + self.B @ np.asarray(u, float).reshape(-1, 1)
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.ravel()

    def update(self, z):
        z = np.asarray(z, float).reshape(-1, 1)
        y = z - self.H @ self.x                      # innovation
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)     # Kalman gain
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P
        return self.x.ravel()

    def step(self, z, u=None):
        self.predict(u)
        return self.update(z)


def constant_velocity_filter(dt: float, q: float = 1.0, r: float = 4.0):
    """Convenience: 1-D position/velocity CV model (state = [pos, vel])."""
    F = np.array([[1, dt], [0, 1]])
    H = np.array([[1, 0]])
    Q = q * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    R = np.array([[r]])
    x0 = np.array([0.0, 0.0])
    P0 = np.eye(2) * 10.0
    return KalmanFilter(F, H, Q, R, x0, P0)
class ExtendedKalmanFilter:
    """EKF for 2D vehicle localization (Bicycle Model)."""
    def __init__(self, dt, q_std, r_std):
        self.dt = dt
        self.x = np.array([[0.0], [0.0], [0.0], [0.0]]) # [x, y, v, theta]
        self.P = np.eye(4) * 10.0
        self.Q = np.diag([q_std**2, q_std**2, q_std**2, q_std**2])
        self.R = np.diag([r_std**2, r_std**2]) # GPS noise

    def predict(self, a, omega):
        """Non-linear motion model (Bicycle)."""
        v = self.x[2, 0]
        theta = self.x[3, 0]
        dt = self.dt

        # State transition (non-linear)
        self.x[0, 0] += v * np.cos(theta) * dt
        self.x[1, 0] += v * np.sin(theta) * dt
        self.x[2, 0] += a * dt
        self.x[3, 0] += omega * dt

        # Calculate Jacobian (F matrix)
        F = np.eye(4)
        F[0, 2] = np.cos(theta) * dt
        F[0, 3] = -v * np.sin(theta) * dt
        F[1, 2] = np.sin(theta) * dt
        F[1, 3] = v * np.cos(theta) * dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        """z is GPS measurement [x, y]"""
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        y = z.reshape(-1, 1) - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P