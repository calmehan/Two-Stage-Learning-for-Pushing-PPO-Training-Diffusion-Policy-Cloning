# Two-Stage Learning for Pushing: PPO Training + Diffusion Policy Cloning

This project implements a two-stage pipeline for planar object pushing in simulation: (1) train an expert PPO policy in a shaped-reward environment, then (2) distill that expert into a conditional diffusion policy via behavioral cloning.

This project:
- Trains a **PPO expert** (Stable-Baselines3) for **planar block pushing** in **PyBullet**, using a shaped reward over the **object pose state space**.
- Collects PPO **rollout trajectories** as demonstrations.
- Performs **policy distillation** by cloning the PPO expert into a **conditional diffusion policy** (**DDPM + 1D UNet**), optimizing noise-prediction MSE on action sequences conditioned on recent observations, and using **iterative denoising** at inference time to generate actions.

---

## 🎬 Demo

(2x speed)  

https://github.com/user-attachments/assets/ce9b0704-4d30-4345-b620-9182d42d82e0


---

## 🚀 Pipeline Overview

### Stage 1 — PPO Expert (RL)
1. **Initialize PyBullet pushing environment**
2. **Define shaped reward** over object pose state
3. **Train PPO** using Stable-Baselines3
4. **Evaluate** trained policy in the simulator
5. **Collect rollouts** to form an imitation dataset

### Stage 2 — Diffusion Policy (Behavior Cloning)
1. Build a **DDPM noise scheduler** (forward process: add noise to actions)
2. Implement **reverse diffusion step** (denoising)
3. Train a **conditional 1D UNet** to predict noise given:
   - noisy action sequence
   - diffusion timestep
   - recent observation history (conditioning)
4. **Sample actions** by starting from Gaussian noise and running iterative denoising
5. Execute the first few predicted actions (receding-horizon style)

---

## 🧰 Dependencies

Core:
- Python 3.x
- `numpy`, `matplotlib`, `tqdm`
- `pybullet`
- `torch`
- `stable-baselines3[extra]`
- `shimmy>=2.0`
- `numpngw`
