# Abstract

MindComposer is an innovative AI-driven music generation system designed to emulate the human "melody-to-harmony" creative process, specifically tailored for Irish folk music. Generating stylistically authentic music under a small-data regime (e.g., the Nottingham dataset) often suffers from structural incoherence and stylistic erosion. To address this, we propose a heterogeneous system architecture that leverages the complementary strengths of different neural networks: a Transformer-based (NanoGPT-style) melody generator to capture long-range structural motifs via self-attention , and a Seq2Seq (BiLSTM + Attention) chord generator to provide a stable, theory-grounded harmonic foundation.

System performance was enhanced through systematic optimizations, including Key Normalization to unify grade relationships and Scheduled Sampling to mitigate exposure bias during inference. Most notably, we implemented a "Strategic Overfitting" approach to transform typical data scarcity into a stylistic advantage, ensuring high fidelity to traditional folk patterns. Quantitative evaluation demonstrates the system's robustness, achieving a 92.10% chord validation accuracy and an 89.8% ABC syntax correctness rate. Statistical analysis reveals a 99.74% cosine similarity in pitch distribution between generated and training data, proving MindComposer’s capability to produce compositions that are both musically coherent and stylistically indistinguishable from human-authored folk melodies.

# Conclusion

## 1 Project Summary and Key Contributions

The MindComposer project successfully establishes a robust framework for automated folk music generation by integrating deep learning architectures with fundamental music theory principles. Our research highlights the efficacy of a layered design : the BiLSTM-based chord generator utilizes sequential inductive bias to maintain harmonic stability , while the Transformer-based melody generator employs self-attention to replicate the repetitive motifs essential to the Irish folk genre. The visualization of our Dual-Attention Strategy confirmed that the model learned to balance global tonal anchoring with local melodic alignment, effectively mimicking human harmonic reasoning.

## 2 Reflections: Turning Constraints into Features

A significant technical insight gained was the management of the "small-data regime." Rather than viewing the limited size of the Nottingham dataset solely as a hurdle, we adopted "Strategic Overfitting". This approach demonstrates that in artistic generation tasks, the goal is often stylistic fidelity rather than broad generalization. By allowing the model to capture the intricate statistical nuances of the training corpus, we achieved a near-perfect pitch distribution similarity (99.74%). Furthermore, transitioning to Song-ID-based data partitioning was a critical engineering decision that eliminated data leakage, ensuring that the evaluation metrics accurately reflected the model’s true generative capacity.

## 3 Future Work

While MindComposer excels in stylistic mimicry, there are clear avenues for further enhancement:

- Harmonic Sophistication: Refining the Seq2Seq decoder to handle more complex transitions between rare or chromatic chord structures. * Dynamic Instrumentation: Developing an emotional-cue-based mechanism to automatically select MIDI instruments according to the calculated melody density.
- Cross-Genre Adaptation: Expanding the architecture to accommodate diverse musical styles, testing the limits of "Strategic Overfitting" across different cultural datasets.
