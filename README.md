# PreciseVideo: A Dual-Process Framework for Zero-Shot Text-to-Video Generation with Quantitative Content Control
PreciseVideo is a **zero-shot text-to-video generation framework** that enables **precise and quantifiable control** over both background and character content. It is built upon pre-trained text-to-image diffusion models and explicitly separates **background generation** and **character generation** stages to enhance control and visual quality.

## ✨ Key Features

- 🎯 **Dual-Stage Design**: Separates video generation into **background** and **character** stages for enhanced control and flexibility.
- 🌌 **Region-Wise Temporal Control**: Introduces a *latent space noise modulator* and *sparse-fusion attention* to control how much different areas in the scene change over time.
- 🧍 **Consistent and Complete Characters**: Proposes *optimal-reference-frame attention* to maintain **identity, appearance, and limb completeness** across frames.
- 🎮 **ControlNet Integration**: Achieves **99.51% accuracy** in aligning generated characters with specified poses or counts.
- 📊 **Quantitative Temporal Control**: The  shows strong correlation with frame-wise metrics:
  - MSE: **0.93**
  - SSIM: **–0.96**
- 👥 **Handles Complex Scenes**: Capable of handling:
  - Multiple interacting characters
  - Scene-to-character and person-to-person occlusions
  - Crowded environments


## 🎬 Experimental Results
Here we showcase some representative generated videos demonstrating the capabilities of **PreciseVideo**.

### 1. We are able to control the temporal variation intensity of video backgrounds. The following results demonstrate the effect of varying the temporal variation intensity (δ) from left to right, with values ranging from 0 to 1 in increments of 0.1.


![If the animation does not display correctly, please refer to ./examples/bg1.gif.](./examples/bg1.gif)

### 2. Our control capability is independent across regions. In the following example, the temporal variation intensity of the "sky" region gradually decreases, while that of the "waves" region gradually increases.


![If the animation does not display correctly, please refer to ./examples/bg2.gif.](./examples/bg2.gif)


### 3. 

![If the animation does not display correctly, please refer to ./examples/bg2.gif.](./examples/character_resized.gif)



### 3. 

![If the animation does not display correctly, please refer to ./examples/bg2.gif.](./examples/Tai_Chi_seed_0-199.gif)



### 3. 

![If the animation does not display correctly, please refer to ./examples/bg2.gif.](./examples/skiing_seed_0-199.gif)




### 3. 

![If the animation does not display correctly, please refer to ./examples/bg2.gif.](./examples/Yoga_seed_0-199.gif)


### 3. 

![If the animation does not display correctly, please refer to ./examples/bg2.gif.](./examples/situp_seed_0-199.gif)
