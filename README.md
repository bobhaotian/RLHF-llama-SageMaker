# Evaluating PPO-based RLHF for Enhanced Pedagogical Quality in AI Tutoring

**Authors:** Robert Bao, Jack Zhao, Benny Wu, Steven Zhu, Catherine Zhang  
**Date:** July 24, 2025

---

## Table of Contents

- [Problem Definition](#problem-definition)
- [Data Construction](#data-construction)
  - [Policy-0 (SFT Model)](#policy-0-sft-model)
  - [Reward Model](#reward-model)
  - [PPO Model](#ppo-model)
- [Algorithm Design](#algorithm-design)
  - [Fine-Tuning (Policy-0)](#fine-tuning-policy-0)
  - [Reward Model Training](#reward-model-training)
  - [PPO-based RLHF](#ppo-based-rlhf)
- [Experiment](#experiment)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [Acknowledgments](#acknowledgments)
- [References](#references)

---

## Problem Definition

The primary objective of this research is to evaluate the effectiveness of Proximal Policy Optimization (PPO) based Reinforcement Learning from Human Feedback (RLHF) in enhancing pedagogical quality within AI tutoring systems. This investigation seeks to determine whether PPO-enhanced models can provide superior educational content compared to traditional supervised fine-tuning approaches, with specific focus on clarity, correctness, and educational value in technical concept explanations.

## Data Construction

Our data construction methodology follows a systematic three-phase approach, designed to create comprehensive datasets for training and evaluation across different model architectures.

### Policy-0 (SFT Model)

We initiated the data construction by selecting ten foundational fields critical for pedagogical evaluation: Algorithms, Calculus, Cryptography, Data Structures, Graph Theory, Linear Algebra, Machine Learning, Operating Systems, Optimization, and Statistics. We systematically generated 100 concise and straightforward concept-level questions per field using OpenAI's API.

The structured prompt explicitly requested minimal complexity and jargon suitable for high-school or first-year university students:

> "Generating questions for topic: {topic}... Generate a list of 100 questions that ask to explain a technical term specifically from the field of {topic}. Use only common jargon typical to {topic}. All questions should be concise and simple. Do not use numbered lists. Example (Data Structures): 'What is a linked list?', 'Describe a stack data structure.', 'Why do we use hash tables?'"

These questions were stored systematically in CSV format. For supervised fine-tuning (SFT) model data (Policy-0), we employed the prompt: *"Given you the question and the answer, summarize the answer in exactly 50 words or less. Keep it factual and clear: Question: {question}"*.

**Example entry:**

```json
{
  "Instruction": "Define a recursive algorithm.",
  "Input": "",
  "Output": "A recursive algorithm solves problems by invoking itself on smaller instances, breaking tasks into simpler subproblems until reaching a base case. Effective for repetitive structures like trees or sequences, they provide elegant and clear solutions but may incur performance issues due to the overhead of multiple function calls."
}
```

### Reward Model

Next, we created a dataset to clearly capture human preferences. Annotators reviewed pairs of responses—one detailed, step-by-step explanation and one concise paragraph—and chose the pedagogically superior response based on clarity, correctness, and educational value. Annotators' choices were meticulously documented.

**An illustrative dataset entry is:**

```json
{
  "prompt": "Define a recursive algorithm.",
  "chosen": "A recursive algorithm is a method of solving problems where the solution to a problem is derived by solving smaller instances of the same problem. This approach involves the algorithm calling itself with modified parameters, allowing it to break down complex tasks into simpler, more manageable subproblems until reaching a base case that directly provides a solution without further recursion. Recursive algorithms are particularly effective for tasks involving repetitive structures, such as traversing trees or computing sequences, and are characterized by their elegance and clarity in expressing solutions, although they can sometimes lead to performance concerns due to increased overhead from multiple function calls.",
  "rejected": "A recursive algorithm is a problem-solving method that breaks down a problem into smaller instances of the same problem. Here's how it works: 1. **Identify the Base Case:** Determine the simplest form of the problem, where a direct solution is possible, defining when recursion stops. 2. **Recursive Step Definition:** Express the problem in terms of smaller, similar problems, bringing it closer to the base case with each recursive call. 3. **Combine the Results:** Solve smaller instances and combine their results to solve the original problem. This process simplifies complex problems through repeated application of the same logic, structuring the solution intuitively."
}
```

### PPO Model

Finally, we constructed data for PPO model training by extracting simplified prompts directly from our structured datasets, creating prompt-only JSON data for reinforcement learning applications.

**An example is:**

```json
{
  "prompt": "Define a recursive algorithm."
}
```

This structured approach ensured accurate representation of our pedagogical objectives, enabling precise comparisons among PPO-enhanced RLHF, Reward Model, and LoRA-tuned methods.

## Algorithm Design

Our algorithmic approach consists of three sequential training phases, each building upon the previous to achieve optimal pedagogical performance.

### Fine-Tuning (Policy-0)

The baseline model is fine-tuned using the LoRA technique on the Llama-3-8B model. Specific parameters include:

- **Rank:** 8
- **LoRA Alpha:** 32
- **Epochs:** 3
- **Target Modules:** ["q_proj", "v_proj"]

LoRA significantly reduces the number of trainable parameters by adapting a subset of weights, efficiently optimizing the pre-trained model without overfitting or high computational costs.

### Reward Model Training

A reward model is trained using the dataset containing preferred (chosen) and less-preferred (rejected) responses. The reward model utilizes LoRA-based adaptation on Llama-3-8B with 4-bit quantization (nf4 quantization type). It employs the Hugging Face Trainer API, with cross-entropy loss designed to rank responses.

**Specific configurations include:**

- **Rank:** 16
- **LoRA Alpha:** 32
- **Epochs:** 3
- **Learning Rate:** 2e-5
- **Batch Size:** 1 (with gradient accumulation over 8 steps)

### PPO-based RLHF

PPO, a policy gradient method, stabilizes reinforcement learning by clipping policy updates and incorporating KL-divergence penalties to maintain policy proximity and prevent overfitting:

```
L(φ) = -E[min(π_φ/π_old * A, clip(π_φ/π_old, 1-ε, 1+ε) * A)] - β*KL(π_φ || π_ref)
```

Here, π_old is the policy before update, π_φ is the new policy, A is the advantage estimate, and β*KL(π_φ || π_ref) prevents policy drift, keeping model outputs aligned with human preferences without overfitting. We use a pre-trained reward model (trained with our human preference dataset) to estimate these advantages.

**In our setup, we:**

1. Load the reward model artifact from SageMaker
2. Initialize policy and reference policy (frozen copy of the original policy)
3. Quantize and apply LoRA adapters to the policy model
4. Train using PPOTrainer from TRL (Transformers Reinforcement Learning library), optimizing policy based on rewards predicted by the reward model

**Configuration includes:**

- **Batch Size:** 8
- **Mini Batch Size:** 2
- **Learning Rate:** 1e-6
- **PPO Epochs:** Defined implicitly by the SageMaker training cycle

This approach leverages the PPO algorithm's robust theoretical underpinnings to enhance pedagogical quality systematically, making incremental policy improvements informed directly by human-aligned reward signals.

## Experiment

We conducted experiments comparing the pedagogical quality of responses generated by two models: the baseline LoRA-tuned model (Policy-0) and the PPO-enhanced RLHF model (Policy-1). Both models were evaluated using MRBench, which assesses responses on clarity, scaffolding, correctness, misconception remediation, and supportive tone.

### MRBench Evaluation Process

For each prompt, we obtained responses from both Policy-0 (baseline LoRA-tuned model) and Policy-1 (PPO-enhanced RLHF model) under identical conditions to ensure fair comparison. The responses were then evaluated by trained annotators who rated each response using a comprehensive 5-point Likert scale across five critical pedagogical dimensions:

- **Clarity:** The extent to which explanations are clear, well-structured, and easily understandable
- **Correctness:** The accuracy of information and appropriateness of the pedagogical approach
- **Scaffolding:** The effectiveness of building understanding through step-by-step guidance
- **Misconception Handling:** The ability to identify and address common student misconceptions
- **Encouraging Tone:** The supportiveness and motivational quality of the response

## Evaluation

The qualitative evaluation indicated modest improvements in the PPO-based RLHF model compared to the baseline LoRA model. Policy-0 responses frequently showed repetitive patterns, lacked coherence, and occasionally lost focus, making explanations less effective pedagogically. While Policy-1 responses generally improved clarity and coherence, particularly through better-structured explanations and more appropriate language, the improvement was not uniformly substantial across all tested prompts. For some complex topics, Policy-1 still struggled to produce thoroughly engaging and misconception-aware explanations. This indicates that PPO-based RLHF has potential but might require additional parameter tuning and more diverse or targeted training datasets to achieve significant pedagogical improvements.

## Conclusion

This research provides preliminary evidence that PPO-enhanced RLHF can moderately improve pedagogical quality over LoRA fine-tuning. However, further experimentation with different hyperparameters, reward model refinements, and richer training datasets is recommended to realize PPO's full potential in enhancing educational AI tutoring capabilities.

## Acknowledgments

We acknowledge the use of ChatGPT (GPT-4o) for grammar improvement and text refinement throughout this document.

## References

Chen, L., Zhang, Y., Wang, X., Li, J., & Brown, M. (2025). MRBench: A comprehensive benchmark for mathematical reasoning evaluation. *arXiv preprint arXiv:2503.09276*.
