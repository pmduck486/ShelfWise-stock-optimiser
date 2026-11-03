ShelfWise — Autonomous Multi-Store Retail Replenishment & Inventory Optimization Engine

<p align="center">
  <strong>Predict Smarter. Optimize Inventory. Minimize Waste.</strong>
</p><p align="center">
  An intelligent, closed-loop retail inventory optimization system combining Quantile Machine Learning, Newsvendor Theory, Mixed-Integer Linear Programming, and LP Duality to make cost-efficient replenishment decisions across multiple retail stores.
</p><p align="center">
  <a href="https://shelf-wise-stock-optimiser.vercel.app/"><strong>🚀 Live Dashboard</strong></a> •
  <a href="https://github.com/pmduck486/ShelfWise-stock-optimiser"><strong>💻 GitHub Repository</strong></a>
</p><p align="center">
  <img src="https://img.shields.io/badge/Institution-IIT%20Patna-blue" />
  <img src="https://img.shields.io/badge/Course-CS2105-orange" />
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.14-blue?logo=python" />
  <img src="https://img.shields.io/badge/Frontend-React%2018-61DAFB?logo=react" />
  <img src="https://img.shields.io/badge/Optimization-MILP%20%7C%20LP-success" />
  <img src="https://img.shields.io/badge/Solver-COIN--OR%20CBC-orange" />
  <img src="https://img.shields.io/badge/Tests-299%20Passed-brightgreen" />
</p>---

📖 Table of Contents

1. "Overview" (#-overview)
2. "The Problem" (#-the-problem)
3. "Our Solution" (#-our-solution)
4. "Key Features" (#-key-features)
5. "System Architecture" (#-system-architecture)
6. "Mathematical Formulation" (#-mathematical-formulation)
7. "Technology Stack" (#-technology-stack)
8. "Benchmark Results" (#-benchmark-results)
9. "Interactive Dashboard" (#-interactive-dashboard)
10. "Repository Structure" (#-repository-structure)
11. "Installation and Setup" (#-installation-and-setup)
12. "Running the Project" (#-running-the-project)
13. "Experimental Methodology" (#-experimental-methodology)
14. "Academic Concepts Demonstrated" (#-academic-concepts-demonstrated)
15. "Future Scope" (#-future-scope)
16. "Course Information" (#-course-information)

---

🚀 Overview

Retail inventory management is a complex optimization problem involving uncertain customer demand, limited storage capacity, transportation constraints, procurement budgets, and product perishability.

Traditional replenishment strategies often rely on historical averages or fixed safety-stock rules. Although these approaches can maintain product availability, they frequently result in excessive inventory, unnecessary procurement costs, and substantial food wastage.

ShelfWise is an autonomous multi-store inventory replenishment and optimization engine designed to address these challenges.

Built using the Walmart M5 Forecasting benchmark dataset, ShelfWise simulates inventory management across:

- 10 retail stores
- 120 high-velocity food products
- 20 perishable product categories
- Multiple retail locations across California, Texas, and Wisconsin
- Daily replenishment decisions over extended simulation periods

The system combines machine learning-based demand forecasting with mathematical optimization to determine how much inventory each store should order every day.

Unlike conventional forecasting dashboards, ShelfWise does not stop at predicting demand. It converts predictions into physically feasible, cost-efficient, whole-carton procurement decisions.

The Core Innovation

ShelfWise follows a two-stage decision-making architecture:

Stage 1 — Predict Demand

A quantile-based LightGBM forecasting engine estimates multiple demand scenarios:

- q50: Median demand
- q80: Higher-demand scenario
- q90: High-demand scenario
- q95: Conservative high-service scenario

Stage 2 — Optimize Replenishment

A Mixed-Integer Linear Programming model determines optimal order quantities while considering:

- Truck capacity
- Shelf-space availability
- Procurement budget
- Minimum service-level requirements
- Product shelf life
- Inventory holding and disposal costs

The resulting decisions are evaluated through a closed-loop inventory simulation and independently audited.

---

🎯 The Problem

Retailers face two major inventory management challenges.

1. Understocking

When demand exceeds available inventory:

- Products become unavailable.
- Potential sales are lost.
- Customer satisfaction decreases.
- Revenue and profit margins are affected.

2. Overstocking

When inventory exceeds actual demand:

- Excess capital remains locked in stock.
- Storage capacity is wasted.
- Perishable products expire.
- Disposal and operational costs increase.

Why Traditional Approaches Are Insufficient

Traditional Strategy| Limitation
Last-week sales| Cannot adapt to changing demand
Historical averages| Ignores demand uncertainty
Fixed safety stock| Often produces excessive inventory
Point-forecast replenishment| Does not adequately model uncertainty
Unconstrained Newsvendor| Ignores physical procurement constraints

Our objective: Maintain approximately 95% product availability while minimizing total operational cost and unnecessary inventory waste.

---

💡 Our Solution

ShelfWise introduces an integrated forecasting and optimization pipeline.

Component| Responsibility
Demand Forecasting| Predict future demand using quantile LightGBM
Newsvendor Model| Determine economically optimal service targets
Perishability Engine| Control inventory according to product shelf life
Holiday Policy| Handle festival demand surges and post-holiday slumps
MILP Optimizer| Generate feasible whole-carton replenishment decisions
LP Dual Analysis| Calculate marginal economic values of constraints
Replay Engine| Simulate daily inventory movement
Independent Auditor| Verify inventory conservation and operational feasibility
Web Dashboard| Visualize decisions, performance, and what-if scenarios

---

✨ Key Features

1. Quantile-Based Demand Forecasting

ShelfWise uses LightGBM gradient-boosted quantile regression to predict demand at multiple confidence levels.

Instead of generating only one demand estimate, the forecasting engine produces a distribution of possible demand outcomes.

This allows the replenishment optimizer to account for uncertainty and different stockout-versus-overstock trade-offs.

Forecast quantiles: q50, q80, q90, q95.

2. Newsvendor-Based Inventory Targets

The Newsvendor model determines an economically appropriate inventory target based on underage and overage costs.

[
\tau^*=\frac{C_u}{C_u+C_o}
]

Where:

- (C_u): Cost of understocking.
- (C_o): Cost of overstocking.
- (\tau^*): Optimal critical fractile.

ShelfWise uses a target critical fractile of approximately 0.63 for perishable products.

3. Mixed-Integer Linear Programming

The optimizer converts inventory targets into discrete procurement decisions.

It considers real-world operational restrictions rather than assuming unlimited procurement capacity.

4. Perishable Inventory Management

Fresh products require different replenishment strategies from shelf-stable goods.

ShelfWise implements dynamic spoilage caps based on predicted consumption over the product's remaining shelf life.

5. Holiday-Aware Replenishment

The system incorporates two distinct seasonal policies:

- Pre-holiday demand surge preparation.
- Post-holiday inventory reduction to prevent unnecessary perishable stock accumulation.

6. LP Duality and Shadow Pricing

After obtaining the integer procurement decisions, ShelfWise performs a continuous LP analysis to estimate the marginal economic value of relaxing operational constraints.

This provides insights into whether additional truck capacity, storage space, or procurement budget would improve the system.

7. Closed-Loop Simulation and Auditing

Every replenishment decision is evaluated through a daily inventory replay engine.

An independent audit verifies physical inventory conservation, carton quantities, and operational constraints.

---

🏗️ System Architecture

ShelfWise consists of six interconnected layers.

flowchart TD
    A["Walmart M5 Dataset"] --> B["Data Preparation & Feature Engineering"]

    B --> C["Quantile LightGBM Forecasting"]
    C --> D["Newsvendor Target Generation"]

    D --> E["Perishable Inventory Controls"]
    D --> F["Holiday Demand Policy"]

    E --> G["Daily Store MILP Optimizer"]
    F --> G

    H["Truck Capacity"] --> G
    I["Shelf Space"] --> G
    J["Procurement Budget"] --> G
    K["Service-Level Requirements"] --> G

    G --> L["Optimal Carton Orders"]

    L --> M["LP Dual Shadow Pricing"]
    L --> N["Daily Inventory Replay"]

    N --> O["Independent Audit"]
    M --> P["Performance Analytics"]
    O --> P

    P --> Q["Interactive Web Dashboard"]

Workflow Explanation

1. Data Ingestion: Historical Walmart M5 sales, item information, and calendar data are processed.
2. Feature Engineering: Lag features, rolling statistics, and calendar features are generated.
3. Demand Forecasting: Quantile LightGBM models estimate future demand.
4. Target Generation: Newsvendor theory converts demand predictions into inventory targets.
5. Optimization: The MILP determines feasible procurement quantities.
6. Economic Analysis: LP dual variables estimate marginal constraint values.
7. Simulation: The replay engine updates inventory, sales, and spoilage.
8. Verification: The independent auditor validates the results.
9. Visualization: The dashboard presents the complete decision-making process.

---

📐 Mathematical Formulation

1. Newsvendor Model

The optimal critical fractile is:

[
\tau^*=\frac{C_u}{C_u+C_o}
]

For perishable products, the target is selected between the relevant forecast quantiles according to the critical fractile.

For non-perishable products, ShelfWise uses a higher service-level demand target based on q95.

2. Perishable Inventory Constraint

Inventory should not exceed the predicted consumption over its remaining shelf life.

[
Stock_{i,t}+x_i\cdot pack_i
\leq
\sum_{k=0}^{L_i-1}\hat q_{50,i,t+k}
]

Where:

- (Stock_{i,t}): Existing inventory.
- (x_i): Number of cartons ordered.
- (pack_i): Units per carton.
- (L_i): Remaining shelf life.
- (\hat q_{50,i,t+k}): Median forecast demand.

3. Mixed-Integer Linear Programming

The daily replenishment problem minimizes:

[
\begin{aligned}
\min \quad
&\sum_i \big[
c_i x_i pack_i+c_{line}y_i\
&+h_i I_{i,t+1}
+\pi_i Shortfall_i
+\gamma_i Waste_i
\big]\
&+c_{truck}T
\end{aligned}
]

Subject to the following operational constraints.

A. Integer Carton Quantization

[
x_i\in\mathbb Z_{\geq0}
]

[
x_i\leq My_i,\qquad y_i\in{0,1}
]

B. Truck Capacity

[
\sum_i x_i\leq400T
]

C. Shelf-Space Capacity

[
\sum_i v_i(Stock_{i,t}+x_i pack_i)
\leq V_{store}
]

D. Procurement Budget

[
\sum_i c_i x_i pack_i\leq B_{store}
]

E. Soft Service-Level Constraint

[
Stock_{i,t}+x_i pack_i+Slack_i
\geq0.90Target_{i,t}
]

F. Perishable Spoilage Constraint

[
Stock_{i,t}+x_i pack_i
\leq Cap_{i,t}^{spoilage}
]

The COIN-OR CBC solver finds the optimal feasible procurement decisions.

4. LP Duality and Shadow Pricing

Once the MILP solution is obtained, the integer decisions are fixed and the continuous LP is solved to extract dual values.

[
\lambda_{truck}

\frac{\partial Cost^*}{\partial C_{truck}}
]

[
\lambda_{space}

\frac{\partial Cost^*}{\partial V_{store}}
]

[
\lambda_{budget}

\frac{\partial Cost^*}{\partial B_{store}}
]

These values represent marginal changes in the continuous relaxation's optimal objective under the corresponding constraint perturbations.

Important: These LP dual values are not necessarily the exact marginal changes in the original integer MILP objective.

---

📊 Benchmark Results

ShelfWise was evaluated against four classical replenishment strategies across the core test window and two sealed holdout blocks.

Performance Comparison

Evaluation Period| Fill Rate| Waste (Units)| Net Cost
Core ILP| 94.87%| 8,043| $934,088
Core ILP + Additional Layers| 94.96%| 7,015| $932,094
Holdout Block 1| 94.14%| 4,469| $855,095
Holdout Block 2| 95.40%| 5,465| $887,087

Performance Against Baselines

Baseline| Strategy
habit| Repeat last week's sales
topup_avg| Replenish using a 28-day rolling average
topup_forecast| Point-forecast-based replenishment
store_model| Unconstrained Newsvendor replenishment
store_ilp| ShelfWise constrained MILP optimization

Key Observations

- Maintains approximately 95% fill rate across evaluated periods.
- Reduces excess inventory waste compared with the reported heuristic baselines.
- Achieves reported net-cost improvements of 5.2%–7.9% across the evaluated windows.
- Incorporates operational constraints that conventional replenishment rules generally ignore.

Experimental note: All reported figures are based on the project's simulation accounting and benchmark configuration. They should not be interpreted as independently verified real-world retail savings.

---

🌐 Interactive Dashboard

Explore the live application:

"🚀 Open ShelfWise Dashboard" (https://shelf-wise-stock-optimiser.vercel.app/)

The dashboard provides seven major interactive modules.

Dashboard Module| Description
Executive Overview| Compare fill rates, waste, and operational costs
Workflow Calendar| Explore daily replenishment performance
Day Decision Theatre| Inspect individual store-item-day decisions
Forecast Explorer| Visualize quantile forecasts and actual demand
Shadow Prices & What-If| Simulate budget, space, and truck-capacity changes
Optimization Layers| Compare optimization-layer ablations
How It Works| Understand the complete mathematical pipeline

What-If Simulation

The dashboard includes interactive controls for:

- Shelf-space factor: 0.10× to 1.00×
- Procurement budget factor: 0.10× to 1.25×
- Truck capacity: 10 to 400 cartons

The sensitivity interface uses monotonic cubic Hermite interpolation anchored to 19 empirical solver evaluations.

---

🛠️ Technology Stack

Layer| Technologies
Programming Language| Python 3.11 / 3.14
Machine Learning| LightGBM, Quantile Regression
Optimization| PuLP, Mixed-Integer Linear Programming
Solver| COIN-OR CBC
Mathematical Analysis| Linear Programming, Duality, Shadow Pricing
Data Processing| Pandas, NumPy, Parquet
Testing| Pytest
Frontend| React 18, TypeScript
Styling| Tailwind CSS 4
Animation| Motion
Build Tool| Vite
Deployment| Vercel
Visualization| Interactive charts and custom simulation components

---

📁 Repository Structure

ShelfWise/
│
├── analysis/
│   ├── store_ilp.py
│   ├── whatif.py
│   └── holiday_calibration.py
│
├── configs/
│   └── validation_fast.yaml
│
├── dashboard/
│   ├── src/
│   ├── public/data/
│   ├── export_data.py
│   ├── test_export.py
│   └── package.json
│
├── data/
│   ├── processed/
│   └── m5/
│
├── data_prep/
│   ├── extract.py
│   └── prepare.py
│
├── docs/
│   ├── decisions.md
│   ├── design-principles.md
│   └── technical documentation
│
├── forecast/
│   ├── model.py
│   ├── features.py
│   ├── calibration.py
│   └── freeze.py
│
├── holdout/
│   ├── make_block.py
│   ├── RUN_LIST_block1.md
│   └── RUN_LIST_block2.md
│
├── notebooks/
│   ├── 01_forecaster.ipynb
│   └── 02_order_model.ipynb
│
├── optimize/
│   ├── parameters.py
│   └── replay_audit.py
│
├── outputs/
│
├── replay/
│   ├── engine.py
│   ├── store_model.py
│   ├── baselines.py
│   └── scorecard.py
│
├── tests/
│
├── config.yaml
├── requirements.txt
└── run_holiday_replay.py

---

⚙️ Installation and Setup

Prerequisites

- Python 3.11 or a compatible supported Python environment.
- Node.js and npm for the frontend.
- Git.
- COIN-OR CBC solver (installed or accessible through PuLP).

Step 1: Clone the Repository

git clone https://github.com/pmduck486/ShelfWise-stock-optimiser.git

cd ShelfWise-stock-optimiser

Step 2: Create a Virtual Environment

python -m venv .venv

Activate it:

Windows

.venv\Scripts\activate

Linux / macOS

source .venv/bin/activate

Step 3: Install Dependencies

pip install -r requirements.txt

Step 4: Verify the Installation

python -m pytest tests -q

The project test suite contains 299 tests covering the major data-processing, forecasting, optimization, and simulation components.

---

▶️ Running the Project

1. Run a Quick Validation Simulation

python -m replay.cli \
  --profile full \
  --config configs/validation_fast.yaml \
  --methods store_model,store_ilp \
  --end-date 2015-11-07 \
  --run-name quick_validation

2. Run the Full Replenishment Replay

python -m replay.cli \
  --profile full \
  --config configs/validation_fast.yaml \
  --methods last_week,topup_avg,topup_forecast,store_ilp \
  --run-name production_8week

3. Launch Jupyter Notebooks

jupyter notebook

Explore:

- "01_forecaster.ipynb" — Data preparation, forecasting, leakage validation, and quantile calibration.
- "02_order_model.ipynb" — Newsvendor targets, MILP formulation, dual pricing, and inventory replay.

4. Launch the Frontend Dashboard

cd dashboard

npm install

npm run dev

Open:

http://localhost:5173

5. Regenerate Dashboard Datasets

python export_data.py

python -m pytest test_export.py -q

---

🔬 Experimental Methodology

ShelfWise uses a structured experimental protocol to reduce data leakage and improve the reliability of benchmark comparisons.

1. Strict Information Lag

The forecasting pipeline enforces a two-day informational lag barrier.

Features used for replenishment decisions do not access future sales observations.

2. Holdout Evaluation

Partition| Evaluation Period| Purpose
Core Test| 1 Nov 2015 – 26 Dec 2015| Model development and evaluation
Holdout 1| 27 Dec 2015 – 13 Feb 2016| Sealed evaluation
Holdout 2| 14 Feb 2016 – 2 Apr 2016| Sealed evaluation
Holdout 3| 3 Apr 2016 – 22 May 2016| Virgin holdout

3. Ablation Testing

Different optimization layers are evaluated independently to understand their contribution:

- Service-level floors.
- Perishable spoilage caps.
- Inventory stock caps.
- Holiday replenishment policies.

4. Independent Accounting Audit

The independent auditor checks:

- Inventory conservation.
- Replenishment feasibility.
- Carton quantization.
- Storage restrictions.
- Physical inventory movement.
- Spoilage accounting.

5. Automated Testing

The repository contains 299 tests covering forecasting, data integrity, optimization, simulation, and dashboard export.

---

🎓 Academic Concepts Demonstrated

ShelfWise applies several concepts from the CS2105 Optimization Techniques curriculum.

Academic Concept| Implementation
Linear Programming| Continuous inventory optimization
Integer Programming| Whole-carton procurement
Mixed-Integer Linear Programming| Store-level replenishment
Objective Function| Minimization of total operational cost
Constraints| Budget, truck capacity, and storage limits
Newsvendor Theory| Optimal inventory targets
LP Duality| Economic interpretation of constraints
Shadow Pricing| Marginal value of additional resources
Sensitivity Analysis| What-if operational simulations
Optimization Under Uncertainty| Quantile demand forecasting
Transportation Constraints| Limited truck capacity

The project demonstrates how theoretical optimization concepts can be integrated into a practical decision-support system.

---

🔮 Future Scope

Potential extensions include:

1. Real-Time Inventory Integration: Connect ShelfWise to actual point-of-sale and warehouse systems.
2. Multi-Echelon Optimization: Extend the model to distribution centers, warehouses, and individual stores.
3. Vehicle Routing Optimization: Optimize delivery routes alongside inventory replenishment.
4. Dynamic Pricing: Incorporate price adjustments to reduce near-expiry inventory.
5. Advanced Demand Forecasting: Explore Temporal Fusion Transformers and other probabilistic forecasting methods.
6. Carbon-Aware Optimization: Include transportation emissions and environmental costs in the objective function.
7. Live Supplier Integration: Incorporate supplier lead times, purchase-order constraints, and delivery schedules.

---

👨‍🎓 Course Information

Project: ShelfWise — Autonomous Multi-Store Retail Replenishment & Inventory Optimization Engine

Course: CS2105 — Optimization Technique
