"""Chaotic system definitions and data generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
from scipy.integrate import odeint


def duffing_oscillator(state, t, delta, alpha, beta, gamma, omega):
    x, x_dot = state
    x_dot_dot = -delta * x_dot - beta * x - alpha * x**3 + gamma * np.cos(omega * t)
    return [x_dot, x_dot_dot]


def lorenz63(state, t, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def lorenz96(state, t, F):
    n = len(state)
    dstate = np.zeros(n)
    for i in range(n):
        dstate[i] = (state[(i + 1) % n] - state[i - 2]) * state[i - 1] - state[i] + F
    return dstate


@dataclass
class SystemSpec:
    name: str
    dim: int
    func: Callable
    default_params: Dict[str, float]


SYSTEMS: Dict[str, SystemSpec] = {
    "duffing": SystemSpec(
        name="duffing",
        dim=2,
        func=duffing_oscillator,
        default_params={"delta": 0.02, "alpha": 5.0, "beta": 1.0, "gamma": 7.0, "omega": 0.5},
    ),
    "lorenz63": SystemSpec(
        name="lorenz63",
        dim=3,
        func=lorenz63,
        default_params={"sigma": 10.0, "rho": 28.0, "beta": 8 / 3},
    ),
    "lorenz96": SystemSpec(
        name="lorenz96",
        dim=25,
        func=lorenz96,
        default_params={"F": 8.0},
    ),
}


def generate_timeseries(
    system: str,
    t_end: float,
    t_points: int,
    lyapunov_exponent: float,
    initial_conditions: np.ndarray | None = None,
    params: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if system not in SYSTEMS:
        raise ValueError(f"Unknown system '{system}'. Options: {list(SYSTEMS.keys())}")

    spec = SYSTEMS[system]
    params = params or {}
    all_params = {**spec.default_params, **params}

    if initial_conditions is None:
        if system == "duffing":
            initial_conditions = np.array([0.0, 0.0], dtype=float)
        elif system == "lorenz63":
            initial_conditions = np.array([-5.0, -10.0, 20.0], dtype=float)
        elif system == "lorenz96":
            rng = np.random.default_rng(1)
            initial_conditions = rng.normal(size=spec.dim)
        else:
            raise ValueError("Missing initial conditions.")

    t = np.linspace(0, t_end, t_points)
    lyapunov_time = 1 / np.abs(lyapunov_exponent) if lyapunov_exponent != 0 else 1.0
    t_lyapunov = t / lyapunov_time

    series = odeint(spec.func, initial_conditions, t_lyapunov, args=tuple(all_params.values()))
    series = np.asarray(series, dtype=float)

    return t_lyapunov, series


def select_observable(series: np.ndarray, system: str, observable: str) -> np.ndarray:
    if system == "duffing":
        if observable == "x":
            return series[:, 0:1]
        if observable == "x_dot":
            return series[:, 1:2]
        if observable == "full":
            return series
        raise ValueError("Duffing observable must be one of: x, x_dot, full.")

    if system in {"lorenz63", "lorenz96"}:
        return series

    raise ValueError(f"Unknown system '{system}'.")
