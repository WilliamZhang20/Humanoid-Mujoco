# Humanoid Mujoco

Experimenting with controllers on a humanoid figure in the [Mujoco](https://github.com/google-deepmind/mujoco) simulation environment.

The specifications on joints and parts of the model were imported directly from DeepMind's repository to put the focus on control laws.

So far, I have successfully set up a Linear Quadratic Regulator controller for the humanoid to balance on its left foot. This control strategy minimizes a quadratic cost function by computing a linear feedback gain matrix. [More info](https://www.youtube.com/watch?v=E_RDCFOlJx4) from Brian Douglas.

A demonstration:

https://github.com/user-attachments/assets/26d9ca69-8f12-499a-814d-7effc0f0bc5f
