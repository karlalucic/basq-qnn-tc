# Challenge Proposal Template for University Partners

_(Please use this template as a guideline. Answer as accurately as possible, and feel free to add or modify sections where necessary.)_

---

## General Information

1. **Institution** Basque Quantum (BasQ)  

2. **Research Group Name**: Benjamin Tirado, Unai Aseguinolaza,  

3. **Contact Email(s)**: redacted in this public mirror  

4. **Research Field/Expertise**: Quantum optimization and Quantum simulation using IBM Quantum Computers.  

---

## Challenge Overview

5. **Challenge idea Title**: Predicting critical temperatures in superconductors using Quantum Neural Networks (QNNs)  

6. **Brief Description of the Challenge**:  

- Participants will design, train, and evaluate a Quantum Neural Network (QNN) to predict the superconducting critical temperature *Tc* of materials from their physical/material descriptors. In addition to designing QNNs for Tc regression in simulation, participants will explore how circuit depth, qubit count, gate noise, hardware connectivity, shot budget, and readout errors affect model performance. Since QPU usage time will be provided, teams are encouraged to perform hardware-aware experiments comparing ideal simulation, noisy simulation, and real-device execution. The goal is not only to minimize prediction error, but also to understand what makes a QNN practical, robust, and hardware-efficient for a real scientific regression task.

7. **Problem Statement**:  

- Superconducting materials can carry electrical current without resistance below a material-specific critical temperature, *Tc*. Discovering materials with higher *Tc* is a major scientific and technological goal, but experimental discovery and theoretical screening remain expensive and time-consuming.  

- This challenge addresses the regression problem of mapping material properties to a continuous *Tc* value. It relates directly to quantum materials because superconductivity is a quantum phase of matter, and because many quantum technologies, including superconducting qubits, depend on the properties of superconducting materials. Better computational tools for *Tc* prediction could help accelerate the search for improved superconductors.  

8. **Relevant Sustainable Development Goals (SDGs)**:  

- **SDG 7: Affordable and Clean Energy – Ensure access to affordable, reliable, sustainable, and modern energy**. Higher-temperature superconductors could reduce or eliminate resistive losses in power transmission and enable more efficient energy infrastructure.  

- **SDG 13: Climate Action – Take urgent action to combat climate change and its impacts**. Improved energy efficiency and reduced grid losses can contribute to lower emissions and more sustainable energy systems.


---

- SDG 9 - Industry, Innovation and Infrastructure – Build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation. Superconductors are relevant to advanced infrastructure, quantum technologies, high-field magnets, sensing, and next-generation computing.

- SDG 3 - Good Health and Well-being – Ensure healthy lives and promote well-being for all at all ages. Superconductors are key components in MRI systems; materials with lower cooling requirements could make advanced medical diagnostics more accessible.

9. **Role of Quantum Computing in Solving the Challenge**:  

- Quantum computing is central to this challenge not only as a modeling tool, but also as the object of study. Participants will explore whether QNNs can provide useful regression models for materials data under realistic hardware constraints.  

- The challenge is especially relevant because superconductors are themselves a key enabling technology for many quantum processors. By using QPUs to help model superconducting materials, participants are exploring a feedback loop in which quantum computers may contribute to the discovery of materials that improve future quantum hardware.  

- The focus is not to assume immediate quantum advantage, but to study practical questions such as:  

  ○ Which QNN architectures are most suitable for current QPUs?  

  ○ How much circuit depth can be tolerated before noise dominates?  

  ○ Can hardware-efficient ansätze provide a useful compromise between trainability and expressivity?  

  ○ Do error-mitigation techniques improve Tc prediction on real quantum devices?  

  ○ How close is real-hardware performance to ideal and noisy simulation?

10. **Preliminary Resources for Participants**:  

- Superconductivity data: https://archive.ics.uci.edu/dataset/464/superconductivty+data  

- Participants are encouraged to consult the Qiskit documentation, available in the IBM Quantum platform (https://eu-de.quantum.cloud.ibm.com/docs/en/guides), as well as the Qiskit Machine Learning module (https://qiskit-community.github.io/qiskit-machine-learning/). Particularly useful documentation pages are:  

  ○ Tutorials about different Machine Learning models: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/index.html  

  ○ EstimatorQNN: https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.EstimatorQNN.html  

  ○ NeuralNetworkRegressor: https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.NeuralNetworkRegressor.html  

  ○ Variational Quantum Algorithms (VQAs): https://quantum.cloud.ibm.com/learning/en/courses/utility-scale-quantum-computing/variational-quantum-algorithms  

- These references and some more will be cited along the Jupyter Notebook tutorial (see next point)

11. **Pre-existing Code or Templates**:


---

- For this purpose, we are currently developing a tutorial Jupyter Notebook with a step by step guide of the typical structure of a QNN: data encoding, variational layers and measurement. For this, we will provide the participants a very simple QNN model that they could potentially use as a starting point.

---

# Technical Feasibility & Requirements

12. **Quantum Algorithms or Approaches**:  

- Variational Quantum Circuits / Quantum Neural Networks: QNNs with trainable notation and entangling gates.  

- Hardware-Efficient Ansätze: Low-depth circuits using native or hardware-friendly gate sets.  

- Hybrid Quantum-Classical Training: Classical optimizers updating quantum circuit parameters based on regression loss.  

- Parameter-Shift Gradients: Gradient estimation for trainable quantum gates.  

- Shot-Based Training and Inference: Studying the effect of finite measurement shots on Tc prediction.  

- Noisy Simulation: Testing models under realistic noise models before QPU execution.  

- Error Mitigation: Applying techniques such as measurement-error mitigation, zero-noise extrapolation, dynamical decoupling, or resilience options available in the chosen quantum platform.  

- Hardware-Aware Circuit Compilation: Adapting circuits to QPU topology, native gates, and transpilation constraints.  

- Classical Baselines: Comparing against a classical MLP with a similar number of trainable parameters, as suggested in the original challenge notes.  

- Model regularization : Demonstrate the robustness of the model through selected metrics.

13. **Quantum Hardware or Software Requirements**:  

- This challenge is recommended to be executed on both quantum simulators and real quantum hardware. The recommended platforms are:  

  - Qiskit / Qiskit Machine Learning  

    - EstimatorQNN or SamplerQNN-based models.  

    - Qiskit Runtime primitives for hardware execution.  

    - Transpilation tools for hardware-aware circuit optimization.  

  - PennyLane  

    - Variational circuit construction.  

    - Parameter-shift gradients.  

    - Interfaces with simulators and available hardware backends.

14. **Classical Computing Power and Software**:  

- The challenge should be completable with standard laptops and the QPU. Required software:  

  - Python, NumPy, SciPy, pandas, scikit-learn.  

  - PyTorch or TensorFlow, if used for classical baselines.  

  - Matplotlib or similar tools for plotting training curves, RMSE, and hardware/simulator comparisons.

15. **Prior Knowledge Requirements**:  

- Beginner – Basic knowledge of quantum computing. We believe that the starter notebook and in-person mentorship could help even beginners develop a simple


---

QNN model. Nonetheless, prior knowledge on the following topics could be useful:

- Single- and two-qubit gates

- Gradient-based optimization

- Regression metrics

- Basic Python

---

## Expected Outcomes & Impact

16. **Desired Outcomes**:  

- Participants should aim to produce a complete hardware-aware QNN workflow for \( T_c \) regression. Successful submissions should include:  

  - A QNN model for predicting superconducting critical temperature Tc.  

  - Feature reduction to a QPU-compatible number of input dimensions.  

  - A comparison against at least one classical baseline, such as a small MLP.  

  - Results from ideal simulation.  

  - Results from noisy simulation or real QPU execution.  

  - At least one hardware-aware design choice, such as shallow ansatz design, topology-aware transpilation, reduced entanglement, shot optimization, or hardware-native gate selection.  

  - At least one error-mitigation or noise-management strategy.  

  - A discussion of the trade-off between model accuracy, circuit complexity, and hardware robustness.  

- Outstanding submissions would demonstrate not only low RMSE, but also a thoughtful analysis of how the QNN behaves under realistic QPU conditions.

17. **Evaluation Criteria**:

| Weight | Description |
| --- | --- |
| **Regression performance** | 25% | Final RMSE or MAE on the test set. Teams should report performance for simulator and, where possible, hardware runs. |
| **Hardware efficiency** | 25% | Quality of the circuit design for real QPUs: low depth, limited two-qubit gates, topology awareness, efficient qubit usage, and sensible shot budget. |
| **Noise robustness and error mitigation** | 20% | Use and analysis of techniques such as measurement mitigation, zero-noise extrapolation, dynamical decoupling, transpilation optimization, or shot-noise analysis. |
| **Model design and expressivity** | 15% | Appropriateness of the feature map, ansatz, entanglement pattern, and number of trainable parameters. |


---

| **Scientific insight and comparison** | 15% | Quality of comparison between classical baseline, ideal simulation, noisy simulation, and QPU execution. Includes clarity of conclusions and limitations. |

18. **Real-World Impact**:  

- Improved \(T_c\) prediction models could help screen candidate superconducting materials before costly synthesis and experimental testing. In the long term, better superconductors could support lossless power transmission, more efficient magnets, cheaper MRI systems, better sensors, and improved superconducting quantum hardware.  

- For the hackathon, the immediate impact is educational and exploratory: participants will learn how quantum machine learning can be applied to a meaningful quantum-materials regression problem and will produce prototypes that may inform future research directions.

19. **Mentorship Availability**:  

- Both of us will be available in person during the hackathon in full capacity e.g., answering questions, providing guidance, reviewing progress before final submissions and evaluation.

---

Final Section

20. **Challenge Presentation**:  

- Would you be available to present this challenge to participants in a short introductory session? Yes, both of us.  

- Would you be interested in giving a keynote during the hackathon? Yes, both of us.

21. **Any additional comments or sections you would like to add ?**  

- No, thanks
