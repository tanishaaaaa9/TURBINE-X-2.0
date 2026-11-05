# TURBINE-X 2.0 — Deep Learning & Neural Architecture Methodology

## 1. Network Architectures
1. **Multilayer Perceptron (MLP) Regressor & Classifier**:
   - Depth: 2 dense hidden layers ($64 ightarrow 32$ neurons).
   - Activations: Rectified Linear Unit (ReLU) with continuous differentiability.
   - Optimization: Adam optimizer with adaptive moment estimation ($eta_1=0.9, eta_2=0.999$, initial learning rate $\eta=0.001$).
   - Regularization: Early stopping monitoring validation loss with a 10-epoch patience buffer.
2. **Deep Autoencoder Bottleneck**:
   - Compression from 9 input features to a 6-dimensional latent representation, reconstructing back to 9 features.
   - Learns non-linear principal thermodynamic manifolds.

## 2. Performance Summary
- **MLP Regressor**: Achieves sub-millisecond inference (0.0006 ms/row), test R² = 0.465 on CO and 0.983 on TEY.
- **MLP Classifier**: Achieves peak accuracy of 58.64% on multi-class CO emission risk tiers.
