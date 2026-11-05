# TURBINE-X 2.0 — Explainable AI (XAI) & Thermodynamic Sensitivity Report

## 1. Methodology
Explainability in TURBINE-X 2.0 is powered by **TreeSHAP (Tree Shapley Additive Explanations)**, based on cooperative game theory:
$$\phi_i(x) = \sum_{S \subseteq F \setminus \{i\}} rac{|S|!(|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)]$$

## 2. Key Thermodynamic Discoveries
1. **Turbine Inlet Temperature (TIT)**:
   - Strongest driver of thermal $NO_x$ formation via the Zeldovich mechanism ($N_2 + O ightarrow NO + N$). Higher TIT exponentially increases NOx.
2. **Turbine After Temperature (TAT)**:
   - Primary driver of Carbon Monoxide ($CO$). High exhaust temperatures indicate complete oxidation of CO into $CO_2$, lowering CO emissions.
3. **Compressor Discharge Pressure (CDP) & Ambient Temp (AT)**:
   - Primary drivers of electrical power yield ($TEY$). High CDP combined with cold ambient air (high air density) dramatically boosts mass flow and energy output.
