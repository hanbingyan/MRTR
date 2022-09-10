# METR
 Source code for ``Can maker-taker fees prevent algorithmic cooperation in market making?``
 (ICAIF 2022)
 
- To reproduce statistics in tables, modify the parameters in `main.py` accordingly:
```
# Sensitivity analysis on these params
n_agents = 2
sig = 0.4
inven_factor = 0.05
# maker rebate is the incentive for liquidity provider
maker_rebate = 0.1
# taker fee is the cost for liquidity taker, usually >= maker_rebate
taker_fee = 0.15
GAMMA = 0.5
```

The `logs` folder contains log files for all tables. Use notebook `Q_Inven.ipynb` for the calculation.