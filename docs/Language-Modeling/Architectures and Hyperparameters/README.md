# Normalization and Activations

## Normalization

### Layer Normalization (LayerNorm)
- Used in the original Transformer ("Attention Is All You Need").
- Formula: let x be a vector, μ = mean(x), σ^2 = var(x):

  The LayerNorm operation is commonly written as:

  $$
  \mathrm{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \varepsilon}} + \beta
  $$

- Pre-LN vs Post-LN:
  - Post-LN (original): normalization after the residual block.
  - Pre-LN (common now): normalization before the sub-layer (attention/FFN). Pre-LN often improves gradient stability and training dynamics for deep models.

### RMS Normalization (RMSNorm)
- RMSNorm removes the mean-centering step and uses root-mean-square for normalization. It has fewer operations and parameters than LayerNorm and can be faster/more memory efficient.

  A common RMSNorm formulation is:

  $$
  \mathrm{RMSNorm}(x) = \gamma \frac{x}{\sqrt{\tfrac{1}{d}\sum_{i=1}^d x_i^2 + \varepsilon}}
  $$

- Notes:
  - No trainable bias (β) in the basic RMSNorm formulation.
  - Slightly cheaper (no subtraction/mean computation) and used in many modern architectures.

### Why choose one vs the other
- LayerNorm provides full centering and scaling; RMSNorm is a lightweight alternative with similar empirical performance in many cases.
- Pre-LN residual blocks + either norm usually give stable training for deep transformers.

## Feed-Forward Layers (FFN) — common forms
- Standard: FFN(x) = W2 * f(W1 x + b1) + b2
 - Standard: 

  $$
  \mathrm{FFN}(x) = W_2\, f(W_1 x + b_1) + b_2
  $$

  - W₁: input → hidden, activation f (ReLU/GeLU/etc.), W₂: hidden → output.
- Gated/Mixture variants are common (GLU/GeGLU/SwiGLU) where activations include element-wise gating for improved expressivity.

## Activations
Common activation functions used in transformer FFNs and variants:

- ReLU: f(x) = max(0, x)
 - ReLU: 

  $$
  \mathrm{ReLU}(x) = \max(0, x)
  $$
  - Simple, cheap; used in early Transformers.

- GeLU (Gaussian Error Linear Unit):
 - GeLU (Gaussian Error Linear Unit):
  - Common approximation:

  $$
  \mathrm{GeLU}(x) \approx x\,\Phi(x)
  $$

  where Φ is the standard normal CDF. Many implementations use a fast approximation for efficiency.
  - Used in GPT and many transformer variants; smoother than ReLU.

- SiLU / Swish: f(x) = x * sigmoid(x)
 - SiLU / Swish:

  $$
  \mathrm{SiLU}(x) = x\,\sigma(x)
  $$

  where σ(x)=1/(1+e^{-x}) is the logistic sigmoid.
  - Smooth, can improve performance in some models.

- Gated activations:
  FF ReGLU:

  $$
  \mathrm{FF\ ReGLU}(x) = (\max(0, xW_1) \otimes (xV)) W_2
  $$

  FF GeGLU:

  $$
  \mathrm{FF\ GeGLU}(x, W, V, W_2) = (\mathrm{GLU}(xW) \otimes (xV)) W_2
  $$

  SwiGLU (Swish is $x\cdot\sigma(x)$):

  $$
  \mathrm{FF\ SwiGLU}(x, W, V, W_2) = (\mathrm{Swish}(xW) \otimes (xV)) W_2
  $$
  
  V -> extra parameter
  Note: Gated models use smaller dimensions for dff by 2/3

### Practical notes
- Most modern transformer implementations use GeLU (or a gated variant) in the FFN.
- When optimizing for memory/speed, consider RMSNorm + GeLU (or gated GeLU) with pre-LN transformer blocks.


## Serial vs Parallel Layers

In standard transformer blocks, layers are computed **serially**: first attention, then MLP.

Some recent models use a **parallel** formulation, where the MLP and attention operate in parallel on the same normalized input.



**Serial (standard) formulation:**

<br>
$$
y = x + \mathrm{MLP}(\mathrm{LayerNorm}(x + \mathrm{Attention}(\mathrm{LayerNorm}(x))))
$$
<br>

**Parallel formulation:**

<br>
$$
y = x + \mathrm{MLP}(\mathrm{LayerNorm}(x)) + \mathrm{Attention}(\mathrm{LayerNorm}(x))
$$
<br>

**Benefits:**
- Parallel layers enable the MLP and attention input matrix multiplications to be fused, resulting in ~15% faster training at large scale.
- Ablation experiments show a small quality drop at 8B scale, but no degradation at 62B scale; extrapolation suggests parallel layers are quality-neutral at even larger scales (e.g., 540B).
