# Block 4: Convolutional Neural Networks (CNNs)
## Architecture, Convolution, Pooling, FC Layer, PyTorch Code

> **Session:** Lecture 2 — The Transformer Architecture  
> **Topics covered:** 15–21

---

## Learning Roadmap for This Block

```
Why CNNs? → Architecture overview → Convolution operation
→ Pooling → Fully Connected Layer → CNN in PyTorch
→ Famous architectures → Drawbacks (bridge to RNNs/Transformers)
```

This block answers: **How do computers see? How do neural networks extract meaning from images?**

---

## Topic 15: What is a CNN?

### The Problem with Fully Connected Networks on Images

A 224×224 pixel RGB image has 224 × 224 × 3 = **150,528** input values. If the first hidden layer has 1000 neurons, that's **150 million parameters** — just for one layer. This is:
- Computationally expensive
- Prone to overfitting (too many parameters, not enough data)
- Ignores the spatial structure of images (nearby pixels are related)

### The CNN Solution

CNNs exploit three properties of visual data:

1. **Local connectivity:** A feature (edge, curve) is determined by nearby pixels, not pixels far away. So instead of connecting each neuron to all 150K inputs, connect it to a small 3×3 or 5×5 patch.

2. **Parameter sharing (weight sharing):** The same edge detector (filter) is useful everywhere in the image — at the top, middle, and bottom. So use the **same weights** across all spatial positions.

3. **Spatial hierarchy:** Low-level features (edges) → mid-level (shapes) → high-level (object parts) → semantic class. Stack layers to build this hierarchy.

### Definition

> A CNN is a deep learning architecture that uses **learnable filters** (kernels) that slide over input data, detecting spatial patterns through convolution. Inspired by the visual cortex — neurons respond to patterns in specific receptive fields.

Originally proposed by LeCun et al. (1989) for handwritten digit recognition (MNIST).

### CNN vs. Fully Connected Network

| Property | Fully Connected | CNN |
|---|---|---|
| Connection pattern | Every neuron → every input | Local patch → filter |
| Parameter sharing | No | Yes (same filter everywhere) |
| Spatial awareness | No | Yes |
| Translation invariance | No | Yes (via pooling) |
| Param count (224×224) | ~150M (1st layer alone) | ~few thousand (3×3 filter) |

---

## Topic 16: Convolution as a Matrix Operation

### The Core Mechanism

A filter (kernel) — a small matrix of learnable weights — slides over the input image. At each position, it computes an **element-wise dot product** between the filter and the overlapping patch of the image. This produces one value in the **feature map**.

### The Worked Example from Lecture

**Input image (5×5):**
```
1  2  3  0  1
0  1  2  3  0
1  3  2  1  2
2  1  0  1  3
1  0  1  2  1
```

**Filter/Kernel (3×3):**
```
-1  0  1
-2  0  2
-1  0  1
```
*(This is actually a Sobel filter — a classical edge detector!)*

**Computing the top-left value of the feature map:**

Patch (top-left 3×3 of image) × Filter:
```
(1×-1) + (2×0) + (3×1) +
(0×-2) + (1×0) + (2×2) +
(1×-1) + (3×0) + (2×1)
= -1 + 0 + 3 + 0 + 0 + 4 + -1 + 0 + 2
= 7
```

The filter slides one position right (stride=1), computes the next dot product, and so on.

**Resulting feature map (3×3):**
```
 7   0  -6
 3   1  -2
 1   4  -2
```

The feature map is 3×3 (smaller than 5×5 input) because a 3×3 filter on a 5×5 image with stride 1 produces a (5-3+1)×(5-3+1) = 3×3 output.

### Key Parameters

**Stride:** How many pixels the filter moves at each step.
- Stride 1: overlapping receptive fields, larger output
- Stride 2: non-overlapping, output is half the size

**Padding:** Adding zeros around the border of the input.
- `padding=0` (valid): output is smaller than input
- `padding=1` (same): output is the same size as input (for 3×3 filter)

**Formula for output size:**
```
output_size = (input_size − kernel_size + 2×padding) / stride + 1
```

### What Does Each Filter Detect?

A single filter detects ONE specific pattern everywhere in the image. With multiple filters, each detects something different:

- Filter 1 → horizontal edges
- Filter 2 → vertical edges
- Filter 3 → diagonal patterns
- Filter N → high-level semantic concepts (in deep layers)

These filters are **learned automatically** from data — not hand-designed.

### The Receptive Field

Each neuron in a feature map "sees" a specific region of the input called its receptive field. In deeper layers, the receptive field grows — neurons in layer 3 indirectly see a large portion of the input image.

---

## Topic 17: The Pooling Layer

### Purpose of Pooling

After convolution, the feature map still has many values. Pooling:
1. **Reduces spatial dimensions** (downsampling) → fewer parameters, less computation
2. **Provides translation invariance** → a feature detected slightly off-center still gets detected
3. **Selects the strongest signal** → takes the most activated value in a region

### Max Pooling — The Worked Example from Lecture

**Input feature map (4×4):**
```
12  20   8   4
18  36  14   6
 5  11  22  30
 2   7  17  25
```

**Max Pool (2×2 window, stride=2):**

Divide into 4 non-overlapping 2×2 windows and take the maximum:
```
TL region: max(12, 20, 18, 36) = 36
TR region: max( 8,  4, 14,  6) = 14
BL region: max( 5, 11,  2,  7) = 11
BR region: max(22, 30, 17, 25) = 30
```

**Max-pooled output (2×2):**
```
36  14
11  30
```

The 4×4 feature map becomes 2×2 — halved in each dimension.

### Average Pooling

Instead of taking the max, takes the average of each window. Less commonly used than max pooling. Max pooling tends to work better because it selects the strongest activation (the most "confident" feature detection).

### Global Average Pooling (GAP)

A special case where the pool window = the entire feature map. Produces a single value per channel. Used in modern architectures (ResNet, EfficientNet) to replace fully connected layers at the end.

---

## Topic 18: Fully Connected Layer

### What It Does

After the convolutional and pooling layers have extracted spatial features, the fully connected (FC) layer combines all these features to make the final classification/regression decision.

### The Step-by-Step from Lecture

**Step 1: Flatten**

The pooled feature map (2D) is flattened into a 1D vector:
```
[[36, 14],     →    [36, 22, 30, 25]   (4-dimensional vector)
 [11, 30]]
```
*(Note: the example uses slightly different values for illustration)*

**Step 2: Multiply by Weight Matrix**

```
W (3×4):
[ 0.2  -0.1   0.4   0.1]   → Cat score
[-0.3   0.5   0.1  -0.2]   → Dog score
[ 0.1   0.2  -0.3   0.5]   → Bird score

scores = W × x:
Cat:  0.2×36 + (-0.1)×22 + 0.4×30 + 0.1×25 = 19.5
Dog: -0.3×36 +  0.5×22  + 0.1×30 + (-0.2)×25 = -1.8
Bird: 0.1×36 +  0.2×22  + (-0.3)×30 + 0.5×25 = 11.5
```

**Step 3: Softmax → Probabilities**

```
Softmax([19.5, -1.8, 11.5]) ≈ [100%, 0%, 0%]
```

The model is essentially certain it's a cat.

### The FC Layer Interpretation

Each row of the weight matrix W learns "what combination of features looks like class X." The FC layer is the model's "reasoning" stage — it looks at all the extracted features and makes the final call.

---

## Topic 19: Full CNN Architecture & Flow

```
Input Image
(e.g., 224×224×3 RGB)
    │
    ▼
Conv Layer 1 (e.g., 32 filters, 3×3)
→ Feature maps: 224×224×32
    │
    ▼
ReLU Activation
    │
    ▼
Max Pooling (2×2)
→ Feature maps: 112×112×32   [spatial size halved]
    │
    ▼
Conv Layer 2 (e.g., 64 filters, 3×3)
→ Feature maps: 112×112×64
    │
    ▼
ReLU + Max Pooling
→ Feature maps: 56×56×64
    │
    ▼
...deeper layers...
    │
    ▼
Flatten
→ 1D vector
    │
    ▼
Fully Connected Layer(s)
    │
    ▼
Softmax Output
→ Class probabilities
```

**The pattern in deeper layers:**
- Feature map **depth increases** (more filters → more feature types detected)
- Feature map **spatial size decreases** (pooling reduces spatial dimensions)
- Early layers → edges, textures
- Mid layers → shapes, object parts
- Deep layers → high-level semantic concepts

---

## Topic 20: CNN in PyTorch

### The Building Blocks

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv Layer 1: 1 input channel (grayscale), 16 feature maps, 3×3 kernel
        # padding=1 keeps spatial size the same (224→224)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                               kernel_size=3, padding=1)
        
        # Conv Layer 2: 16 → 32 feature maps
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                               kernel_size=3, padding=1)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Max Pooling: 2×2 window, stride=2 → halves spatial dimensions
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # After 2 pooling layers: 28×28 → 14×14 → 7×7
        # 32 feature maps × 7×7 = 1568 flattened features
        self.fc = nn.Linear(32 * 7 * 7, 10)  # 10 output classes
    
    def forward(self, x):
        # Conv 1 → ReLU → Pool
        x = self.pool(self.relu(self.conv1(x)))  # [B, 16, 14, 14]
        
        # Conv 2 → ReLU → Pool
        x = self.pool(self.relu(self.conv2(x)))  # [B, 32, 7, 7]
        
        # Flatten: [B, 32, 7, 7] → [B, 1568]
        x = x.view(x.size(0), -1)
        
        # Classification
        return self.fc(x)  # [B, 10]

model = SimpleCNN()
print(model)
```

### Key PyTorch CNN Components

| Layer | PyTorch | Parameters |
|---|---|---|
| Convolution | `nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)` | Learnable filters |
| Activation | `nn.ReLU()` | None (stateless) |
| Max Pooling | `nn.MaxPool2d(kernel_size, stride)` | None (stateless) |
| Fully Connected | `nn.Linear(in_features, out_features)` | W and b |
| Batch Norm | `nn.BatchNorm2d(num_features)` | γ, β (learnable scale/shift) |
| Dropout | `nn.Dropout(p=0.5)` | None |

### Why ReLU?

```
ReLU(z) = max(0, z)
```

- Computationally simple
- **Doesn't saturate** for positive values (unlike Sigmoid) → no vanishing gradient for positive activations
- Sparse activations (many zeros) → efficient
- Default activation for CNNs

Variants: Leaky ReLU, ELU, GELU (used in Transformers).

### Parameter Count Example

```
Conv2d(1, 16, 3, padding=1):
  Params = (3 × 3 × 1 + 1) × 16 = 144 weights + 16 biases = 160

Conv2d(16, 32, 3, padding=1):
  Params = (3 × 3 × 16 + 1) × 32 = 4,640

Linear(32×7×7, 10):
  Params = 1568 × 10 + 10 = 15,690

Total: ~20,000 parameters (vs. millions for a fully connected network!)
```

---

## Topic 21: Famous CNN Architectures & Drawbacks

### The Evolution of CNNs

| Architecture | Year | Innovation | Performance |
|---|---|---|---|
| LeNet-5 | 1989 | First practical CNN, digit recognition | MNIST |
| AlexNet | 2012 | Deep + GPU + ReLU + Dropout | ImageNet 2012 winner (57% → 37% error) |
| VGGNet | 2014 | Very deep (16-19 layers), simple 3×3 convs | ImageNet 2014 |
| GoogLeNet/Inception | 2014 | Inception modules, parallel convolutions | ImageNet 2014 winner |
| ResNet | 2015 | **Skip connections** — trains 150+ layer networks | ImageNet 2015 winner |
| EfficientNet | 2019 | Compound scaling — balances depth/width/resolution | State-of-the-art efficiency |

### The Residual Connection (ResNet) — A Key Insight

ResNet solved the "deep networks don't train well" problem:

```
Without skip connection:
  x → [Conv → BN → ReLU → Conv → BN] → F(x) → output

With skip connection (residual block):
  x → [Conv → BN → ReLU → Conv → BN] → F(x)
  └────────────────────────────────────→ + → output (= F(x) + x)
```

The network learns the **residual** F(x) rather than the full mapping. This allows gradients to flow directly through the shortcut, enabling training of very deep networks.

### CNN Applications

**Computer Vision:**
- Image classification (ImageNet, medical imaging)
- Object detection: YOLO, Faster R-CNN
- Semantic segmentation: U-Net (widely used in medical imaging)
- Facial recognition

**Medical Imaging:**
- Tumor detection in MRI/CT scans
- Diabetic retinopathy screening
- COVID-19 detection from chest X-rays
- Histopathology analysis

**Other Domains:**
- Text classification (Text-CNN)
- Audio spectrogram classification
- Video understanding (frame-by-frame)

### Drawbacks of CNNs — Bridge to RNNs

**Drawback 1: Fixed Input Size**
> CNNs are designed for fixed-size inputs. A 224×224 image classifier can't directly process a 448×448 image or a variable-length sentence. 

**Drawback 2: No Temporal/Sequential Reasoning**
> CNNs process data as spatial grids. They have no mechanism to model sequences — where the order of elements matters. "The dog bit the man" and "The man bit the dog" would produce similar CNN representations but have opposite meanings.

**Drawback 3: No Long-Range Dependencies**
> The receptive field grows with depth, but for long sequences (like paragraphs of text), CNNs struggle to capture dependencies between distant words/tokens.

These limitations motivate **Recurrent Neural Networks (RNNs)** — the subject of Block 5.

---

## Interview Questions — Block 4

**Q1: Why are CNNs preferred over fully connected networks for image tasks?**

> Three key advantages: (1) Local connectivity — filters connect to local patches, not the entire input, reducing parameters dramatically. (2) Parameter sharing — the same filter is applied everywhere, enabling features to be detected regardless of position (translation equivariance). (3) Spatial hierarchy — stacking conv layers builds increasingly abstract feature representations. Together these reduce parameters by orders of magnitude while encoding prior knowledge about spatial structure.

**Q2: What is the difference between valid and same padding?**

> Valid padding (padding=0) produces an output smaller than the input — for a k×k filter, the output shrinks by (k-1) pixels on each side. Same padding adds enough zeros around the input so the output has the same spatial dimensions as the input. Same padding is more common in modern networks as it simplifies architecture design.

**Q3: What does max pooling do and why is it used?**

> Max pooling takes the maximum value in each window of the feature map. It (1) reduces spatial dimensions by a factor equal to the stride, reducing computation and parameters downstream; (2) provides translation invariance — a feature detected slightly off-center still gets the same max value; (3) selects the strongest activation, which represents the most confident feature detection in that region.

**Q4: What is a skip/residual connection and why did it revolutionize deep learning?**

> A skip connection adds the input of a block directly to its output: output = F(x) + x. This allows gradients to flow directly from later layers to earlier ones through the shortcut, preventing the vanishing gradient problem in very deep networks. ResNet used skip connections to train networks with 50-150+ layers, whereas networks deeper than ~20 layers without skip connections degraded in performance. Skip connections effectively let the network learn residual functions (differences from identity) which is easier to optimize.

**Q5: How does convolution reduce the number of parameters compared to a fully connected layer?**

> A fully connected layer on a 224×224×3 image with 1000 neurons = 150,528 × 1000 = 150M parameters. A convolutional layer with 32 filters of size 3×3 = (3×3×3 + 1) × 32 = 896 parameters — 167,000× fewer. The key is weight sharing: the same filter is reused at every spatial position.

**Q6: Explain the role of ReLU in CNNs.**

> ReLU (Rectified Linear Unit) is the standard non-linearity after conv layers: ReLU(z) = max(0, z). It solves the vanishing gradient problem that sigmoid/tanh suffer at saturation — for positive inputs, ReLU's gradient is exactly 1. It's computationally cheap (just a comparison), produces sparse activations (many zeros), and works well in practice. Modern variants like Leaky ReLU (small negative slope) and GELU (smooth approximation) address the "dying ReLU" problem where neurons get stuck at 0.

**Q7: What is batch normalization and where is it typically applied in a CNN?**

> Batch normalization normalizes the output of a layer to have zero mean and unit variance within each mini-batch, then applies learnable scale (γ) and shift (β) parameters. It's typically applied after the linear operation and before the activation: Conv → BN → ReLU. Benefits: (1) reduces internal covariate shift, allowing higher learning rates; (2) slight regularization effect; (3) reduces sensitivity to initialization; (4) enables deeper networks to train stably.

---

## Key Learning Insights

> **Insight 1:** The "convolution" in CNN is actually a **cross-correlation** in signal processing terms. True mathematical convolution flips the filter — but in deep learning, filters are learned anyway, so the flip is irrelevant. The terms are used interchangeably in ML.

> **Insight 2:** CNNs encode a powerful **inductive bias** — the assumption that patterns in images are translation-invariant and locally structured. This prior knowledge is baked into the architecture. When your data has this property, CNNs are dramatically more efficient than general-purpose networks.

> **Insight 3:** The **depth** of a CNN is what makes it powerful. Shallow CNNs detect simple patterns; deep CNNs detect complex, hierarchical concepts. AlexNet's 2012 ImageNet breakthrough wasn't just about CNNs — it was about showing that DEEP CNNs trained on GPUs beat everything else by a wide margin.

> **Insight 4:** Feature maps at different layers can be visualized. Tools like **Grad-CAM** show which regions of an image a CNN focuses on for a given prediction — a key tool for explainability in medical imaging.

> **Insight 5:** The drawbacks of CNNs (no sequence modeling, fixed input size) directly motivate the move to **RNNs and eventually Transformers**. Understanding what CNNs CAN'T do is as important as understanding what they can.

---

## Quick Reference Cheatsheet

```
CNN Pipeline:
  Input → [Conv → ReLU → Pool]× N → Flatten → FC → Softmax

Key operations:
  Convolution: (input_size − kernel + 2×padding) / stride + 1 = output_size
  Max Pool (2×2, stride=2): halves spatial dimensions
  FC: W×x + b  (each row learns one class)

Parameter counts:
  Conv2d(in, out, k): (k×k×in + 1) × out  [+ bias]
  Linear(in, out): in×out + out

Activation:
  ReLU: f(x) = max(0, x)  → no vanishing grad for positive values

Famous: LeNet(1989) → AlexNet(2012) → VGG(2014) → ResNet(2015) → EfficientNet(2019)

CNN Drawbacks → RNNs needed for:
  - Variable-length sequences
  - Temporal/sequential reasoning
  - Long-range dependencies
```
