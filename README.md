# 💲 Price Optimization with Competing Products

> **Machine learning + mathematical optimization:** predict demand with a regression model, embed that model *inside* a Gurobi optimization, and find the prices and space allocation that maximise revenue for two competing product categories.

**Course:** BMN 616 – Advanced Optimization Techniques for Managers · Demo Project
**Institute:** Department of Management Studies, IIT Roorkee

| Team member | Roll no. |
|---|---|
| Name 1 | Roll no. |
| Name 2 | Roll no. |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gurobi/modeling-examples/blob/master/pricing_competing_products/price_optimization_with_competing_products.ipynb)
&nbsp;Based on Gurobi's open-source modeling example ([source](https://github.com/Gurobi/modeling-examples/tree/master/pricing_competing_products)).

---

## 🎯 The problem

A seller has **200 units of limited space** (shelf space in retail, seats at an event, rooms in a hotel) to split between **two competing categories** of a product.

- Raising **Category 1's price** lowers its sales (*own-price effect*).
- Raising **Category 2's price** pushes buyers to Category 1 (*cross-price effect*).

**Question:** what prices should we set, and how much space should each category get, to maximise revenue while respecting the business rules?

## 🧠 The approach (3 steps)

**1. Predict.** Train a regression model on 1,000 historical observations to predict Category 1 sales `n[1]` from both prices `p[1]` and `p[2]`. The notebook assumes everything stocked sells, so Category 2 gets the remaining space: `n[2] = 200 − n[1]`.

**2. Optimize.** Build a Gurobi model:

| | |
|---|---|
| **Decision variables** | $p_c$ = price of category $c$; $n_c$ = units allocated to category $c$ |
| **Objective** | $\max \; \sum_c p_c n_c \;-\; \lambda \sum_c p_c^2$ (revenue minus an optional price penalty) |
| **Total space** | $n_1 + n_2 = 200$ |
| **Minimum stock** | $n_c \ge 50$ |
| **Price bounds** | $300 \le p_1 \le 400$, $\;100 \le p_2 \le 300$ |
| **Price ordering** | $50 \le p_1 - p_2 \le 100$ (Category 1 is the premium product) |

**3. Combine.** Use **[Gurobi Machine Learning](https://github.com/Gurobi/gurobi-machinelearning)** (`add_predictor_constr`) to turn the trained model into constraints. The solver can then only choose prices whose predicted demand matches `n[1]`. Because revenue is price × quantity (both decision variables), the model is **non-convex**, so `NonConvex = 2` lets Gurobi solve it to global optimality.

```python
m_feats = pd.DataFrame({"p[1]": [p[1]], "p[2]": [p[2]]})              # decision variables as ML features
pred_constr = add_predictor_constr(m, xgb_regressor, m_feats, n[1])    # embed the trained model
m.Params.NonConvex = 2
m.optimize()
```

## 📊 Results (from our run, λ = 0)

| | Category 1 | Category 2 |
|---|---|---|
| **Optimal price** | **$364.97** | **$300.00** |
| **Space allocated** | **91 units** | **109 units** |

**Optimal revenue: $65,931.83.** Gurobi proved optimality (0% gap) in about 0.1 s.

**Why it matters.** Without the demand model, the solver would simply set the highest prices ($400 / $300) with 150 / 50 units and "promise" **$75,000**. That plan ignores that demand falls when prices rise. With the ML model inside the solver, the recommendation is one that demand can actually support.

**Model comparison (5-fold cross-validation, mean R² on held-out folds):**

| Model | Mean test R² |
|---|---|
| Linear Regression | ≈ 0.79 |
| Gradient Boosting (10 trees) | ≈ 0.67 |

The notebook embeds the Gradient Boosting model. Swapping in the linear model is a natural extension (see *Further work*).

> Results come from the executed notebook in this repo (`gurobipy` 13.0, `scikit-learn` 1.9). Small differences are possible with other library versions.

---

## 🎮 The interactive presentation: "Price Lab"

Instead of static slides, we presented with a **3-level interactive HTML deck** (16:9). It is one self-contained file with no installation needed.

| Level | Speaker | What the audience sees |
|---|---|---|
| **LV1 · Predict** | Speaker 1 | Price sliders drive a 200-slot shelf grid and a demand curve. Four "discovery missions" explain own-price effects, cross-price effects, the sell-everything assumption, and linear vs boosted models. |
| **LV2 · Optimize** | Speaker 2 | Starts from a *broken* pricing plan: fix all four business rules live, explore the objective and λ penalty, then "Reality check" the naive plan against demand. |
| **LV3 · Combine** | Speaker 3 | Data → Train → Embed → Solve pipeline, an optimizer arena (demand-ignored vs demand-aware), annotated code, and a box to enter the real notebook results. |

<p>
  <img src="docs/images/level1-predict.jpg" width="49%" alt="Level 1: Predict demand">
  <img src="docs/images/level2-optimize.jpg" width="49%" alt="Level 2: Build the model">
</p>
<img src="docs/images/level3-combine.jpg" alt="Level 3: Combine ML and optimization">

**Controls:**
- `←` / `→` switch levels.
- **H** toggles a clean recording view.
- Each level has a **2:00 timer**.
- XP and badges reward exploring every mission.

> The deck uses a simplified **demo demand curve** for live interaction. The actual optimal values come from the notebook and can be typed into Level 3.

**▶ Open it:**
- Download `presentation/Competing_Products_Price_Lab.html` and open it in any browser, or
- enable GitHub Pages (see *How to upload*) and open it straight from a link.

---

## 📁 Repository structure

```
.
├── notebook/
│   ├── price_optimization_with_competing_products.ipynb   # executed notebook with outputs
│   └── price_optimization_with_competing_products.py      # same code as a Python script (Colab export)
├── presentation/
│   └── Competing_Products_Price_Lab.html                  # interactive 3-level demo deck
├── data/
│   └── price_value_data.csv                               # 1,000 rows: p[1], p[2], n[1]
├── docs/images/                                           # screenshots for this README
├── requirements.txt
├── LICENSE           # MIT: presentation + README
├── LICENSE-APACHE    # Apache 2.0: Gurobi notebook + data
└── NOTICE
```

## ⚙️ Run it yourself

**In Colab (easiest):** click the *Open in Colab* badge above and run all cells.

**Locally:**
```bash
pip install -r requirements.txt
jupyter notebook notebook/price_optimization_with_competing_products.ipynb
```

`pip install gurobipy` includes a **size-limited licence** that is enough for this model. Students can get a free full academic licence from [Gurobi's academic program](https://www.gurobi.com/academia/academic-program-and-licenses/).

The notebook loads the data from Gurobi's GitHub. To work offline, change the `read_csv` path to `../data/price_value_data.csv`.

## 🔭 Further work

- **Sweep λ:** re-solve for several penalty values to see how optimal prices respond.
- **Swap the model:** embed the Linear Regression (better R² here) and compare decisions.
- **Relax assumptions:** the data is synthetic, there are only two categories, and every unit is assumed to sell.

## 📚 Credits

- Modeling example and dataset: © 2024 Gurobi Optimization, LLC, [Gurobi modeling-examples](https://github.com/Gurobi/modeling-examples), Apache 2.0.
- [Gurobi Machine Learning](https://github.com/Gurobi/gurobi-machinelearning) for embedding scikit-learn models.

This is an independent academic project and is not affiliated with or endorsed by Gurobi Optimization. See [NOTICE](NOTICE) for licensing details.
