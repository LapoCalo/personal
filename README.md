# Personal
Here I put all the projects done during university and during my free-time. The work is in progress...

## Estimation and Filtering (MSc subject)
The goal was to design and analyse a Kalman filter for satellite attitude estimation, fusing gyroscope and star tracker measurements. Using MATLAB, the work covered both continuous-time and discrete-time formulations, deriving the state-space representation and optimal filter gains for first- and second-order models. A parametric study was then carried out to analyse the effect of noise variance on transient and steady-state behaviour, and to evaluate the impact of temporary star tracker loss on estimation accuracy.
## Launchers Guidance and Control (MSc subject)
The objective was to design, analyse, and validate an H-infinity controller for the rigid motion control of a heavy launcher (Ariane 5 class) during its atmospheric ascent phase. Using MATLAB, the launcher was modelled as a rigid body in both time and frequency domains. An augmented system with appropriately tuned weighting functions was built to synthesise the controller. A sensitivity study was conducted on the weight parameters, followed by frequency-domain assessment via Bode and Nichols plots, and time-domain simulations under realistic disturbances such as wind gusts and thrust deflection offsets to verify compliance with angle-of-attack and deflection constraints.
## Launchers Architecture (MSc subject)
The objective was to design a two-stage rocket launcher using the Falcon 9 as a reference architecture. Using MATLAB, the report covered the computation of delta-V requirements, sizing of the two stages in terms of masses and propellants, propulsion system design, geometry definition, and trajectory optimisation accounting for gravity and drag losses. A bonus section computed the orbital injection state vector in the ECI frame and verified the final orbit.
## Orbital Mechanics (MSc subject)
The aim was to apply orbital mechanics theory to practical problems through numerical simulations in MATLAB. Three main problems were addressed: solving Kepler's equation via a Newton-Raphson method to propagate satellite orbits and analyse eclipse times, numerically integrating the equations of motion using ode45 to study 2-body dynamics and conservation of energy and angular momentum, and performing orbit phasing and rendezvous manoeuvres, including ISS interception, delta-V computation, and de-orbiting of the chaser.
## Satellite Architecture Project (MSc subject)
A comprehensive multi-subsystem satellite design project developed using MATLAB and SatOrb. Starting from mission requirements, the report covered orbitography analysis (Sun-synchronous and inclined orbits, ground coverage, visibility windows and perturbations), communications link budget sizing, thermal analysis across different orbit and coating configurations, AOCS subsystem design including pointing requirements, disturbing torques, and comparison of attitude control architectures, and power system sizing including battery capacity and solar array requirements.

## Lunar Landing (self-study)
Personal simulation of a Moon landing written in Python. Models the full mission from a 100 km circular parking orbit through a three-phase PID-controlled powered descent to touchdown, with a 3-D textured Moon globe and a close-up terminal descent animation built with PyVista (work ongoing).

## AOCS (self-study) (work is ongoing)

### 1. Single-axis PID — Attitude Control Basics

### 2. 3-Axis Nadir Pointing with Quaternion PD Controller
Simulates a satellite at 500 km LEO maintaining nadir 
pointing with a quaternion-based PD controller.
- Settling time: ~40s
- Pointing accuracy: <0.1° (3σ)
- Languages: MATLAB
(work ongoing)
