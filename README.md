# Data-Augmented Control Synthesis and CLF-CBF Verification for CSTR

A Python-based framework for extracting nominal linear models, evaluating Control Lyapunov Functions (CLF), and synthesizing Control Barrier Functions (CBF) for an open-loop unstable Continuous Stirred-Tank Reactor (CSTR).

---

## Technical Overview

The framework provides an end-to-end control and system identification pipeline:

* **Dynamic Simulation:** Simulates the non-linear coupled mass and energy dynamics of a CSTR.
* **Ground Truth Matrix Extraction:** Stacks state X and control U snapshots into an augmented matrix representation to extract local linear state-space ground truth matrices of the linear state-space model Ax + Bu using by augmenting the state and control matrices, appending them with the matrix representing the vector field of the system manifold xdot, and solving for the Moore-Penrose pseudoinverse.
* **Control Lyapunov Function (CLF) Synthesis:** Models energy decay and tracks performance stability using the Continuous-Time Algebraic Lyapunov Equation (CALE).
* **Control Barrier Function (CBF) & Lie Derivative Auditing:** Constructs equilibrium-centered safety cages using Nagumo's theorem and evaluates scalar drift ($L_f h$), actuator leverage ($L_g h$), boundary variance projections, and gradient stability.
* **Pure CLF-CBF QP Controller Integration (Active Work):** Synthesizes a unified Quadratic Program (QP) to act as the primary real-time controller without relying on a nominal baseline control law.

---

## Governing System Equations & Control Formulations

### Non-Linear CSTR Dynamics
The internal state of the CSTR is defined by the state vector $\mathbf{x}(t) = [C_A(t), T(t)]^T$:
* $C_A(t)$: Concentration of reactant $A$ ($\text{kmol/m}^3$)
* $T(t)$: Reactor temperature ($\text{K}$)

The coupled differential equations governing mass and energy balances are:

$$\frac{dC_A}{dt} = \frac{q}{V} (C_{Af} - C_A) - k_0 \exp\left(-\frac{E}{R T}\right) C_A$$

$$\frac{dT}{dt} = \frac{q}{V} (T_f - T) + \Delta H_{\text{term}} \cdot k_0 \exp\left(-\frac{E}{R T}\right) C_A - U_A (T - T_c)$$

Where $T_c$ acts as the control input $u(t)$ modulating system dynamics through the cooling jacket.

### Data-Augmented Ground Truth Matrix Extraction
To extract the nominal linear ground truth matrices without relying on open-loop dither/excitation, state vector snapshots $\mathbf{X}$ and control inputs $\mathbf{U}$ are stacked into an augmented workspace matrix $\mathbf{X}_U$:

$$\mathbf{X}_U = \begin{bmatrix} \mathbf{X} \\ \mathbf{U} \end{bmatrix}$$

Using the exact time derivative matrix $\dot{\mathbf{X}}$, the ground-truth system matrices $\mathbf{A}_{\text{nominal}}$ and $\mathbf{B}_{\text{nominal}}$ are solved simultaneously via the Moore-Penrose pseudoinverse:

$$\begin{bmatrix} \mathbf{A}_{\text{nominal}} & \mathbf{B}_{\text{nominal}} \end{bmatrix} = \dot{\mathbf{X}} \, \mathbf{X}_U^\dagger$$

---

## CLF & Safety Barrier Synthesis

### Continuous Algebraic Lyapunov Equation (CALE)
Given closed-loop performance intent $\mathbf{A}_{\text{cl}} = \mathbf{A} - \mathbf{B}\mathbf{K}$ and state penalty matrix $\mathbf{Q}$, the positive-definite shape matrix $\mathbf{P}$ is computed numerically using `scipy.linalg.solve_continuous_lyapunov`:

$$\mathbf{A}_{\text{cl}}^T \mathbf{P} + \mathbf{P} \mathbf{A}_{\text{cl}} = -\mathbf{Q}$$

The performance potential $V(\mathbf{x})$ and temporal decay rate $\dot{V}(\mathbf{x})$ are evaluated as:

$$V(\mathbf{x}) = \mathbf{x}^T \mathbf{P} \mathbf{x}$$

$$\dot{V}(\mathbf{x}) = 2 \mathbf{x}^T \mathbf{P} (\mathbf{A}\mathbf{x} + \mathbf{B} u)$$

### Control Lyapunov Function (CLF) Negative Definiteness
To ensure asymptotic stability, the temporal derivative of the Control Lyapunov Function must remain strictly negative along the system trajectories:

$$\dot{V}(\mathbf{x}) = 2 \mathbf{x}^T \mathbf{P} (\mathbf{A}\mathbf{x} + \mathbf{B} u) < 0 \quad \forall \, \mathbf{x} \neq \mathbf{0}$$

If $\dot{V}(\mathbf{x}) \ge 0$, the controller fails to dissipate system energy rapidly enough, allowing the unmodeled non-linear kinetics and open-loop thermal runaway modes to override closed-loop dynamics and destabilize the reactor.

### Equilibrium-Ellipsoid Safety Set (CBF) & Nagumo's Theorem
Following Nagumo's theorem, invariance of the safe set $\mathcal{C}$ is guaranteed if the tangent vector field points inward or along the boundary at all points where the barrier function equals zero. For a Control Barrier Function $h(\mathbf{x})$ defining the safe set $\mathcal{C} = \{ \mathbf{x} \in \mathbb{R}^n : h(\mathbf{x}) \ge 0 \}$, Nagumo's theorem requires that there exists an extended class-$\mathcal{K}$ function $\alpha$ for all states of x in the safe set such that:

$$\dot{h}(\mathbf{x}) \ge -\alpha(h(\mathbf{x}))$$

Applying this to the equilibrium-centered safety cage built around the origin ($\mathbf{x}_e = \mathbf{0}$):

$$\mathcal{C} = \{ \mathbf{x} \in \mathbb{R}^n : \mathbf{x}^T \mathbf{P} \mathbf{x} \le \delta^2 \}$$

$$h(\mathbf{x}) = \delta^2 - \mathbf{x}^T \mathbf{P} \mathbf{x} \ge 0$$

### Lie Derivative Decomposition
$$\nabla h(\mathbf{x}) = \frac{\partial h}{\partial \mathbf{x}} = -2 \mathbf{x}^T \mathbf{P}$$

$$L_f h = \nabla h(\mathbf{x}) f(\mathbf{x}) = -2 \mathbf{x}^T \mathbf{P} (\mathbf{A}\mathbf{x})$$

$$L_g h = \nabla h(\mathbf{x}) g(\mathbf{x}) = -2 \mathbf{x}^T \mathbf{P} \mathbf{B}$$

---

## Structural Analysis of Lie Derivatives

### 1. Independent Lie Derivative Component Audits ($L_f h$ vs. $L_g h$)
The codebase explicitly decouples the Lie derivative calculation into $f(x)$ (unforced drift) and $g(x)$ (actuator mapping) components to perform independent physical audits:
* **Drift Flux Audit ($L_f h$):** Evaluates system observability and natural trajectory propagation. When $L_f h < 0$, unforced thermodynamics naturally pull the state toward safety. When $L_f h > 0$, thermal drift accelerates toward a boundary violation.
* **Actuator Leverage Audit ($L_g h$):** Measures real-time actuator authority over the safety boundary normal.

### 2. Higher-Order Lie Derivatives, Actuator Paralysis, and Chattering
If $L_g h \approx 0$, the actuator is physically vector-aligned parallel to the boundary wall, leaving it momentarily powerless to alter boundary distance.

In higher-relative-degree architectures where $L_g h = 0$, safety synthesis requires computing higher-order Lie derivatives (e.g., $L_f^2 h$, $L_g L_f h$). However, higher-order derivatives drastically amplify noise present in incoming sensor streams. This sensor noise propagation destabilizes the QP decision boundary, causing rapid high-frequency control input switching—known as actuator chattering—which accelerates physical mechanical wear on valves and pumps.

### 3. SVD-Based Condition Number Monitoring ($\kappa$)
Actuator authority is audited by performing a Singular Value Decomposition (SVD) on $L_g h$ to extract maximum ($\sigma_{\max}$) and minimum ($\sigma_{\min}$) singular values:

$$\text{SVD}(L_g h) = U \Sigma V^T \implies \kappa = \frac{\sigma_{\max}}{\sigma_{\min} + \epsilon}$$

If $\kappa > 100$, the system approaches ill-conditioned control mapping. Rather than applying hard guardrails—which can alter control dynamics unpredictably and create severe operational hazards—the architecture generates real-time warnings to soften optimization constraints downstream.

### 4. Boundary Gradient Stress Testing via $\epsilon$-Nudge Perturbation
To verify the structural stability of the gradient matrix $\nabla h(\mathbf{x}) = \frac{\partial h}{\partial \mathbf{x}}$ independently before coupling it with system vector fields $f(\mathbf{x})$ and $g(\mathbf{x})$, the code executes an $\epsilon$-nudge perturbation test:
1. A small virtual displacement vector ($\epsilon \approx 10^{-5}$) is applied along the gradient normal: $\mathbf{x}_{\text{perturbed}} = \mathbf{x} + \epsilon \cdot \frac{\nabla h^T}{\Vert{}\nabla h\Vert{}}$.
2. The non-linear barrier function at a perturbed state $\mathbf{x}_{\text{pert}}$ is compared against its first-order linear Taylor series expansion around $\mathbf{x}$:

$$h(\mathbf{x}_{\text{pert}}) \approx h(\mathbf{x}) + \nabla h(\mathbf{x})^T (\mathbf{x}_{\text{pert}} - \mathbf{x})$$

Residual errors above $10^{-8}$ trigger non-linearity warnings, verifying local geometric smoothness before online optimization.

### 5. Covariance Mapping vs. Collinearity for Observability Auditing
To assess internal state observability vulnerabilities and drift sensitivity, this framework utilizes Boundary Variance Projection (Covariance Mapping) over collinearity audits:
* **Limitations of Collinearity:** Collinearity metrics zoom in locally on isolated data points or specific vector alignments, failing to capture holistic systemic behavior.
* **Advantages of Covariance Mapping:** Projecting state uncertainty covariance $\mathbf{\Sigma}_x$ directly onto the safety boundary normal ($\sigma_h^2 = \nabla h \cdot \mathbf{\Sigma}_x \cdot \nabla h^T$) delivers a comprehensive, system-wide view of state estimation risk and observability degradation across the entire operating region.

---

## Architectural Assumptions: Pure CLF-CBF (No Nominal Controller) in CSTR Applications

In a low-dimensional Continuous Stirred-Tank Reactor (CSTR) with simple kinetics, a pure CLF-CBF controller is highly feasible without requiring a nominal baseline controller. Unlike robotics (which suffers from non-convex physical obstacles and multi-stage path planning), this simplified case CSTR benefits from:

* **Trivial Safety Geometry:** Safety boundaries are static, decoupled box limits ($T_{\min}, T_{\max}, C_{A,\min}, C_{A,\max}$).
* **Thermodynamic Alignment:** Kinetic coupling means driving the system toward steady state (CLF) naturally aligns with keeping temperature bounded (CBF).
* **Ultra-Low Solver Overhead:** A 2-state, 1-input system leads to a minimal QP that can be solved analytically or via microsecond-level explicit calculations.
* **Thin Boundary Layers:** High control authority allows the system to operate close to operational limits without triggering conservative, system-stalling actions.
* **Simplified Reaction Kinetics:** A critical assumption in this formulation is the simplification of reaction kinetics (e.g., first-order, single-reactant Arrhenius steps). In chemical processes, unmodeled complex or multi-stage reaction kinetics represent a substantially greater source of non-linearity than even boundary layer effects or spatial geometry, making kinetic simplification a prerequisite for clean CLF-CBF synthesis.

---

## Current Work: Unified Standalone CLF-CBF QP Controller

**Active Implementation Status:** The current phase of development is focused on formulating and validating the online Quadratic Program (QP) solver that directly unifies the Control Lyapunov Function (CLF) and Control Barrier Function (CBF) constraints.

Because this architecture operates without a nominal baseline controller, the QP formulation itself serves as the sole control generator. The optimization problem minimizes control effort while enforcing strict stability and safety guarantees in real time:

$$\min_{u, \delta_{\text{slack}}} \, \frac{1}{2} u^T R u + p \cdot \delta_{\text{slack}}^2$$

$$\text{s.t. } \quad L_f V(\mathbf{x}) + L_g V(\mathbf{x})u + c_1 V(\mathbf{x}) \le \delta_{\text{slack}} \quad \text{(CLF Stability Constraint)}$$

$$L_f h(\mathbf{x}) + L_g h(\mathbf{x})u + \alpha(h(\mathbf{x})) \ge 0 \quad \text{(CBF Hard Safety Constraint)}$$

$$u_{\min} \le u \le u_{\max} \quad \text{(Actuator Saturation Limits)}$$

---

## Future Work

* **Lipschitz Continuity Bounds.** Given the maximum residual error between true non-linear CSTR equations and $Ax + Bu$ model over a bounded region, a formal Region of Attraction (RoA) can be computed. Lyapunov theory $V(x)$ is then used to define an invariant sublevel set where the controller guarantees the states will never cross into the "danger zone" where the linear approximation fails.
* **Input-to-State Stability (ISS) Lyapunov functions:** For an unstable linear system, an open-loop pole with $\text{Re}(\lambda) > 0$ means errors amplify. An ISS proof mathematically guarantees that stabilizing feedback gain $K$ (where $u = Kx$) provides enough damping to overcome the positive eigenvalues, ensuring that bounded data/sensor errors result in bounded, predictable state tracking errors rather than infinite divergence.

* Doing the proofs first will provide the theoretical guarantees, bounds, or structural insights needed to determine how warm starting should be applied (e.g., how aggressive the initialization should be, whether it guarantees speedups without hurting convergence, or if it requires specific safeguards).
