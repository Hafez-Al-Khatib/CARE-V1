# Dead Neuron Experiment - Research Report

**Date**: 2026-01-29
**Device**: CUDA (GPU)
**Architecture**: SpikingCNN (3-layer, 103,978 parameters)
**Dataset**: Fashion-MNIST (60,000 train / 10,000 test)
**Epochs**: 15
**Time Steps**: 8
**Dead Neuron Threshold**: 2% firing rate

---

## Executive Summary

**Synaptic Scaling reduces dead neurons by 7.7x** while improving accuracy by +0.9%.

| Metric | Control (Backprop) | Hybrid (+ Scaling) | Delta |
|--------|--------------------|--------------------|-------|
| Accuracy | 76.0% | **76.9%** | +0.9% |
| Dead Neurons | 32.0% | **4.2%** | **7.7x reduction** |

---

## Training Dynamics

### Control (Backprop Only)

| Epoch | Val Acc | Dead Ratio | Avg Rate |
|-------|---------|------------|----------|
| 1 | 64.0% | 52.3% | 0.073 |
| 5 | 68.6% | 45.3% | 0.072 |
| 10 | 75.2% | 33.3% | 0.072 |
| 15 | **76.0%** | **32.0%** | 0.074 |

### Hybrid (+ Synaptic Scaling)

| Epoch | Val Acc | Dead Ratio | Avg Rate |
|-------|---------|------------|----------|
| 1 | 61.9% | **14.1%** | 0.112 |
| 5 | 71.3% | **9.4%** | 0.095 |
| 10 | 71.0% | **6.8%** | 0.091 |
| 15 | **76.9%** | **4.2%** | 0.102 |

---

## Key Findings

### 1. Faster Dead Neuron Recovery

At Epoch 1:
- Control: 52.3% dead neurons
- Hybrid: 14.1% dead neurons
- **Hybrid has 3.7x fewer dead neurons from the start**

### 2. Sustained Activity Improvement

By Epoch 15:
- Control: Dead ratio plateaus at ~32%
- Hybrid: Dead ratio drops to ~4%
- **Final improvement: 7.7x fewer dead neurons**

### 3. Higher Average Firing Rates

- Control avg rate: 0.07 (below target 0.1)
- Hybrid avg rate: 0.10 (at target)
- **Synaptic scaling maintains healthy activity levels**

### 4. Accuracy Preservation

Despite dramatically different neuron utilization:
- Both methods achieve similar final accuracy (~76-77%)
- Hybrid slightly outperforms Control (+0.9%)
- **Dead neuron reduction does not hurt performance**

---

## Mechanism Analysis

### Synaptic Scaling Algorithm

```python
def apply_homeostatic_scaling(target_rate=0.1, lr=0.02):
    for layer in [conv1, conv2, conv3]:
        deviation = target_rate - layer.activity_rate
        scale = (1.0 + lr * deviation).clamp(0.95, 1.05)
        layer.weight *= scale
```

**How it works:**
1. Track exponential moving average (EMA) of firing rates per channel
2. Compute deviation from target rate (0.1 = 10% firing)
3. Scale weights up for under-active neurons, down for over-active
4. Clamp scaling factor to prevent runaway weight growth

---

## Conclusions

1. **Synaptic Scaling is effective**: Reduces dead neurons by 7.7x
2. **No accuracy trade-off**: Hybrid matches or exceeds Control
3. **Fast recovery**: Most dead neurons revived within first few epochs
4. **Simple implementation**: Only requires tracking firing rates per channel

---

## Raw Data

Results saved to: `results/experiment_20260129_125921.json`
