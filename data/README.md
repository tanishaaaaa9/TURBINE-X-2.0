# Gas Turbine CO and NOx Emission Dataset (UCI Repository)

## 1. Overview & Provenance
The dataset utilized in the **TURBINE-X 2.0** platform is the official **Gas Turbine CO and NOx Emission Data Set**, originally collected from a commercial gas turbine located in Turkey and hosted on the **UCI Machine Learning Repository**:
- **UCI Source Link:** [Gas Turbine CO and NOx Emission Data Set](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
- **Creators:** Heydar Soran & F. Kaya (2019)
- **License:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **Total Record Count:** 36,733 hourly operational observations spanning 5 consecutive calendar years (2011–2015).

---

## 2. Feature Schema & Sensor Descriptions

The raw dataset contains strictly 11 physical numerical attributes recorded across operational regimes:

| Column Name | Physical Parameter | Engineering Unit | Domain / Sensor Type |
|---|---|---|---|
| **AT** | Ambient Temperature | °C | Ambient Environment |
| **AP** | Ambient Pressure | mbar | Ambient Environment |
| **AH** | Ambient Humidity | % | Ambient Environment |
| **AFDP** | Air Filter Differential Pressure | mbar | Compressor Inlet Filter |
| **GTEP** | Gas Turbine Exhaust Pressure | mbar | Turbine Exhaust Duct |
| **TIT** | Turbine Inlet Temperature | °C | Combustor Discharge |
| **TAT** | Turbine After Temperature | °C | Exhaust Diffuser Thermocouples |
| **TEY** | Turbine Energy Yield | MWh | Generator Shaft Electric Power Output |
| **CDP** | Compressor Discharge Pressure | bar | Axial Compressor Exit |
| **CO** | Carbon Monoxide Concentration | mg/m³ | Continuous Emission Monitoring System (CEMS) |
| **NOX** | Nitrogen Oxides Concentration | mg/m³ | Continuous Emission Monitoring System (CEMS) |

---

## 3. Strict Chronological Partitioning Protocol

To avoid lookahead bias and reflect authentic industrial deployment where models are trained on historical plant data and deployed on future operating seasons, the repository applies a non-overlapping chronological split:

1. **Development Set (`2011–2013`):**
   - $N = 22,191$ observations ($60.41\%$)
   - Used exclusively for exploratory analysis, feature engineering, cross-validation tuning, scaler fitting, and operational risk quantile thresholding.
2. **Calibration Set (`2014`):**
   - $N = 7,158$ observations ($19.49\%$)
   - Used exclusively for conformal prediction interval calibration and post-hoc threshold validation.
3. **Locked Test Set (`2015`):**
   - $N = 7,384$ observations ($20.10\%$)
   - Strictly locked during development; evaluated only once for the final scientific benchmark.

---

## 4. Modeling Tracks

- **Track A (Emissions Focus):**
  - Features ($9$ inputs): `AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP`
  - Targets ($2$ outputs): `CO` and `NOX`
- **Track B (Joint Energy & Emissions Focus):**
  - Features ($8$ inputs, strictly excluding `TEY`): `AT, AP, AH, AFDP, GTEP, TIT, TAT, CDP`
  - Targets ($3$ outputs): `CO, NOX`, and `TEY`

