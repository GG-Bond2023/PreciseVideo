# PreciseVideo: A Dual-Process Framework for Zero-Shot Text-to-Video Generation with Quantitative Content Control
PreciseVideo is a **zero-shot text-to-video generation framework** that enables **precise and quantifiable control** over both background and character content. It is built upon pre-trained text-to-image diffusion models and explicitly separates **background generation** and **character generation** stages to enhance control and visual quality.

## ✨ Key Features

- 🎯 **Dual-Stage Design**: Separates video generation into **background** and **character** stages for enhanced control and flexibility.
- 🌌 **Region-Wise Temporal Control**: Introduces a *latent space noise modulator* and *sparse-fusion attention* to control how much different areas in the scene change over time.
- 🧍 **Consistent and Complete Characters**: Proposes *optimal-reference-frame attention* to maintain **identity, appearance, and limb completeness** across frames.
- 🎮 **ControlNet Integration**: Achieves **99.51% accuracy** in aligning generated characters with specified poses or counts.
- 📊 **Quantitative Temporal Control**: The temporal variation intensity (δ) shows strong correlation with frame-wise metrics:
  - MSE: **0.93**
  - SSIM: **–0.96**
- 👥 **Handles Complex Scenes**: Capable of handling:
  - Multiple interacting characters
  - Scene-to-character and person-to-person occlusions
  - Crowded environments
