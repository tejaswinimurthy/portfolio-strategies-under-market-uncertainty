# Evaluating Portfolio Strategies Under Market Uncertainty

## 1. Context & Objective
This project evaluates how different portfolio construction strategies perform under market uncertainty, with a focus on the trade-off between optimality and robustness.

The task was framed as a financial analytics decision problem: constructing portfolios using historical return data and assessing whether constrained strategies can deliver more stable out-of-sample performance compared to unconstrained mean–variance optimisation.

The analysis compares portfolio strategies based on their risk–return profiles, sensitivity to estimation error, and performance consistency across time.

## 2. Data Description
The analysis uses monthly return data from the **Fama–French 10 Industry Portfolios** dataset.

This dataset contains value-weighted returns for ten broad industry portfolios, providing diversified exposure across the U.S. equity market. The use of industry-level portfolios reduces idiosyncratic noise while retaining meaningful cross-sectional variation in returns.

Returns are expressed in excess-return form and are suitable for mean–variance portfolio construction and out-of-sample performance evaluation.

## 3. Portfolio Construction Approach
Portfolio strategies were constructed using the classical mean–variance framework, where expected returns and the covariance matrix were estimated from historical data.

Two approaches were evaluated:

- **Unconstrained mean–variance optimisation**, allowing unrestricted portfolio weights.
- **Constrained mean–variance optimisation**, imposing limits on portfolio weights to reduce sensitivity to estimation error.

The constrained approach was designed to improve robustness by preventing extreme allocations driven by noisy return estimates.

## 4. Estimation Window & Target Return
Expected returns and covariances were estimated using a rolling historical window of past returns.

A fixed target return was specified and held constant across strategies to ensure comparability. Portfolios were constructed to minimise variance subject to meeting this target return.

This design isolates the effect of portfolio constraints on risk and stability, rather than differences driven by return targets.

## 5. In-Sample vs Out-of-Sample Evaluation
Portfolio performance was evaluated both in-sample and out-of-sample.

- **In-sample performance** reflects how well portfolios fit the historical data used for estimation.
- **Out-of-sample performance** reflects how portfolios perform when applied to future, unseen data.

Out-of-sample evaluation was prioritised, as it better represents real-world investment decision-making under uncertainty.

## 6. Evaluation Criteria
Strategies were compared using the following criteria:

- Realised mean return
- Portfolio volatility
- Stability of portfolio weights
- Sensitivity to estimation error

This evaluation framework emphasises robustness and consistency rather than purely optimising in-sample performance.

## 7. Results & Key Insights

### Executive Summary (Plain English)
When tested on historical data, unconstrained portfolios appeared attractive, offering higher apparent efficiency. However, when evaluated on new data, these portfolios showed greater instability and sensitivity to estimation error.

In contrast, constrained portfolios delivered more consistent performance across time, with lower volatility and more stable allocations. While they did not always appear optimal in-sample, they proved more reliable out-of-sample.

This highlights a common trade-off in analytics-driven decision-making: strategies that look optimal on paper may perform less predictably when conditions change.

---

### Comparative Insights

- **Unconstrained portfolios** tended to concentrate heavily in a small number of assets, making them highly sensitive to small changes in estimated returns.

- **Constrained portfolios** avoided extreme allocations, resulting in smoother weight distributions and more stable realised performance.

- The performance gap observed in-sample narrowed or reversed out-of-sample, indicating that robustness was more valuable than apparent optimality.

---

### Decision Implications

From a decision-making perspective, constrained optimisation offers a practical advantage. While it may sacrifice some theoretical optimality, it reduces exposure to estimation risk and improves performance consistency.

For real-world portfolio construction, this suggests that incorporating reasonable constraints can lead to better outcomes under uncertainty, especially when future market conditions differ from historical patterns.

## 8. Methods Appendix (Illustrative Code)

This appendix provides a simplified illustration of how portfolio weights were  constructed using historical return data. The code is included for transparency and reproducibility, and is not required to understand the main results.

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load industry portfolio returns (monthly)
returns = pd.read_csv("10_Industry_Portfolios.csv", index_col=0)

# Estimate expected returns and covariance
mu = returns.mean().values
cov = returns.cov().values

n_assets = len(mu)
target_return = 0.01  # illustrative monthly target

# Portfolio variance
def portfolio_variance(w, cov):
    return w.T @ cov @ w

# Constraints
constraints = [
    {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    {"type": "eq", "fun": lambda w: w @ mu - target_return},
]

# Unconstrained optimisation
w0 = np.ones(n_assets) / n_assets
unconstrained = minimize(portfolio_variance, w0, args=(cov,), constraints=constraints)

# Constrained optimisation (bounds on weights)
bounds = [(0, 0.3) for _ in range(n_assets)]
constrained = minimize(
    portfolio_variance, w0, args=(cov,), constraints=constraints, bounds=bounds
)
```
