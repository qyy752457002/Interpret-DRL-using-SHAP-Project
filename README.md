```markdown
# Interpret-DRL-using-SHAP: Interpretable Policy Modeling for Deep Reinforcement Learning
This project focuses on improving the interpretability and reliability of Deep Reinforcement Learning (DRL) agents in high-dimensional visual environments.  
We introduce a complete pipeline for **policy interpretation** using **SHAP-based feature attribution**, **boundary surface extraction**, and **surrogate interpretable policy modeling**.

Paper link:  
https://arxiv.org/pdf/2510.19244

```

@article{qian2025interpretdrl,
title={Interpret Policies in Deep Reinforcement Learning using SILVER with RL-Guided Labeling:
A Model-level Approach to High-dimensional and Multi-action Environments},
author={Qian, Yiyu and others},
journal={arXiv preprint arXiv:2510.19244},
year={2025}
}

````

---

## 🔧 Create Virtual Environment
Create a conda environment:
```bash
conda create -n interpret_drl python=3.9
conda activate interpret_drl
````

## 📦 Dependencies Installation

Install required libraries:

```bash
pip install stable-baselines3[extra] gym[atari] shap scikit-learn numpy pandas matplotlib torch
```

## 🛠 Python Path Configuration

Execute in the project root:

```bash
export PYTHONPATH=$(dirname $(pwd)):$PYTHONPATH
```

---

## 🎮 Reproducing Experiment Results

### **MsPacman Environment**

Run the corresponding notebooks:

| Notebook             | Algorithm | Task                                              |
| -------------------- | --------- | ------------------------------------------------- |
| `MsPacman DQN.ipynb` | DQN       | Train agent, run SHAP + interpretability pipeline |
| `MsPacman PPO.ipynb` | PPO       | Same workflow                                     |
| `MsPacman A2C.ipynb` | A2C       | Same workflow                                     |

All trained models, SHAP values, boundary points, and interpretable surrogates are stored in:

```
/MsPacman/
```

### **RoadRunner Environment**

| Notebook               | Algorithm | Task                    |
| ---------------------- | --------- | ----------------------- |
| `RoadRunner DQN.ipynb` | DQN       | Full pipeline execution |
| `RoadRunner PPO.ipynb` | PPO       | Full pipeline execution |
| `RoadRunner A2C.ipynb` | A2C       | Full pipeline execution |

Outputs stored in:

```
/RoadRunner/
```

---

## 📊 Fidelity Score Evaluation

To compare the surrogate interpretable models with the original RL policies:

```bash
Open and run: Fidelity Plot.ipynb
```

This produces Fidelity comparison curves across:

* MsPacman vs RoadRunner
* DQN vs PPO vs A2C
* Decision Tree vs Logistic Regression vs Linear Regression models

---

## 🔁 Fine-tuning / Running Your Own Models

1. Train a new RL agent using any RL algorithm supported by Stable-Baselines3.

2. Collect gameplay rollouts and extract state embeddings (CNN encoder is included in notebooks).

3. Compute SHAP values:

```python
# Example snippet inside notebook
explainer = shap.KernelExplainer(policy_model, background_states)
shap_values = explainer.shap_values(input_states)
```

4. Generate boundary points and train interpretable models:

```bash
Run sections labeled: "Boundary Extraction" & "Train Surrogate Models" in notebooks
```

5. Evaluate surrogate policy fidelity using:

```bash
Run: Fidelity Plot.ipynb
```

---

## 🔍 Exploring the Core Interpretation Module Alone

If you only want to run the interpretability steps without RL training:

```bash
Use the saved feature datasets + SHAP outputs inside /MsPacman/ and /RoadRunner/
```

or plug in your own model’s state encoder.

---

## 🔑 API Keys (Optional for Cloud Execution)

If running SHAP on external GPU/Lab services, set your key:

```bash
export OPENAI_API_KEY="your_key"
```

---

## 📂 Checkpoints & Data Structure

Both `MsPacman/` and `RoadRunner/` folders contain:

* `models/` → Trained RL agent parameters
* `rollouts/` → Collected state/action/reward logs
* `shap/` → Shapley value vectors
* `boundary/` → Extracted boundary point datasets
* `surrogate/` → Trained interpretable policy models

---

## ✨ Summary

This project provides:

* Full SHAP-based interpretability for DRL
* Boundary surface extraction in Shapley-value space
* Surrogate interpretable policy reconstruction
* Fidelity evaluation to quantify interpretability quality

---

## 📫 Contact

Author: [https://github.com/qyy752457002](https://github.com/qyy752457002)
For questions or contributions, please open an Issue or Pull Request.

```



