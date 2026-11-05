# TURBINE-X Problem Formulation: Track A vs Track B

### Track A: Operational Emission Prediction
- **Inputs ($X_A \in \mathbb{R}^9$):** AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP
- **Targets ($Y_A \in \mathbb{R}^2$):** CO, NOX
- **Operational Rationale:** Emulates real-time emission monitoring during active power generation where generator energy yield (TEY) is known.

### Track B: Joint Multi-Target Emission & Energy Prediction
- **Inputs ($X_B \in \mathbb{R}^8$):** AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP (STRICTLY EXCLUDING TEY)
- **Targets ($Y_B \in \mathbb{R}^3$):** CO, NOX, TEY
- **Operational Rationale:** Simulates dispatch planning and joint optimization where energy yield and emissions must be simultaneously forecasted from environmental and thermodynamic cycle setpoints.
