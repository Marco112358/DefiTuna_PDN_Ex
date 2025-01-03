import numpy as np
import pandas as pd
from functions import (get_L, get_pa, get_LP_pos_price_range, add_empirical_delta, add_empirical_gamma,
                       process_dataframe_summaries, create_line_graph, add_hodl_position)

# Global Params
# Set a price range to test, MIN/MAX/STEP
price_range = np.arange(170, 230, 0.01)
# price_range = np.arange(150, 250, 1.0)
debt_daily_cost = (1 + 0.4) ** (1 / 365) - 1  # 40% APY Interest Cost
trading_fee_daily_yield = (1 + 1.0) ** (1 / 365) - 1  # 100% APY Yield from LP Fees

## Initial Params for example 0, 400 USDC position
X = 0  # SOL
Y = 100  # USDC
P = 200  # current price
TV_start = X * P + Y  # total value of position to start
# Leverage/Debt Params
leverage = 3  # amount of leverage (total position value)
x_pct = 0.75  # percent of leverage to take in token x
y_pct = 1 - x_pct  # percent of leverage to take in token y
x_debt = (leverage - 1) * x_pct * TV_start / P  # debt in x tokens
x_debt_val = x_debt * P  # debt of X valued in y tokens
y_debt = (leverage - 1) * y_pct * TV_start  # debt in y tokens
y_debt_val = y_debt  # debt in y tokens
tot_debt_val = x_debt_val + y_debt_val  # total debt value in y tokens
tot_lp_pos = TV_start + tot_debt_val  # total LP position

x_to_lp = X + x_debt
y_to_lp = tot_lp_pos - x_to_lp * P
pb = 210  # upper price tick on CLAMM LP position
# Get pa given X, Y, P, pb
pa = get_pa(x_to_lp, y_to_lp, P, pb)
# Get initial Total Value of LP position in Y Units

# Get Liquidity (L)
L = get_L(x_to_lp, y_to_lp, P, pa, pb)

# Get 7D, 1D and 30D dataframes
df0_7 = get_LP_pos_price_range(price_range,
                               pa,
                               pb,
                               L,
                               x_debt,
                               y_debt,
                               debt_daily_cost,
                               trading_fee_daily_yield,
                               7,
                               TV_start)
df0_1 = get_LP_pos_price_range(price_range,
                               pa,
                               pb,
                               L,
                               x_debt,
                               y_debt,
                               debt_daily_cost,
                               trading_fee_daily_yield,
                               1,
                               TV_start)
df0_30 = get_LP_pos_price_range(price_range,
                                pa,
                                pb,
                                L,
                                x_debt,
                                y_debt,
                                debt_daily_cost,
                                trading_fee_daily_yield,
                                30,
                                TV_start)

# Add HODL Columns
df0_7 = add_hodl_position(df0_7, 0.5, 0, P)
df0_1 = add_hodl_position(df0_1, 0.5, 0, P)
df0_30 = add_hodl_position(df0_30, 0.5, 0, P)


# Get Empirical Delta
df0_7_delta = add_empirical_delta(df0_7.copy())
df0_1_delta = add_empirical_delta(df0_1.copy())
df0_30_delta = add_empirical_delta(df0_30.copy())
# Get Empirical Gamma
df0_7_gamma = add_empirical_gamma(df0_7_delta.copy())
df0_1_gamma = add_empirical_gamma(df0_1_delta.copy())
df0_30_gamma = add_empirical_gamma(df0_30_delta.copy())

# Process net equity % PnL
df0_equity = process_dataframe_summaries(df0_1, df0_7, df0_30, ["Equity Return", "HODL Return"], P)
# Process net SOL position
df0_net_sol = process_dataframe_summaries(df0_1, df0_7, df0_30, ["Net X Tokens"], P)
# Process net empirical delta
df0_net_delta = process_dataframe_summaries(df0_1_delta, df0_7_delta, df0_30_delta, ["Net Empirical Delta"], P)
# Process net empirical delta
df0_net_gamma = process_dataframe_summaries(df0_1_gamma, df0_7_gamma, df0_30_gamma, ["Net Empirical Gamma"], P)

# Generate Net % PnL Graph and Show
highlight_low = pa / P * 100 - 100
highlight_high = pb / P * 100 - 100
fig1 = create_line_graph(data=df0_equity,
                         title=f"Profit/Loss vs. SOL Price Change with Daily Fees: {round(debt_daily_cost * 100, 4)}% and Daily Yield: {round(trading_fee_daily_yield * 100, 4)}%",
                         xaxis_title="Percent Change in SOL Price",
                         yaxis_title="Estimated % Profit/Loss",
                         legend_title="Time Period",
                         xaxis_dtick=2.0,
                         yaxis_dtick=2.0,
                         highlight_range=(-highlight_low, highlight_high))
fig1.show()

# Generate Net SOL Token Exposure Graph and Show
fig2 = create_line_graph(data=df0_net_sol,
                         title=f"Net SOL Tokens vs. SOL Price Change with Daily Fees: {round(debt_daily_cost * 100, 4)}% and Daily Yield: {round(trading_fee_daily_yield * 100, 4)}%",
                         xaxis_title="Percent Change in SOL Price",
                         yaxis_title="Estimated Amount of SOL Tokens you are Long/Short",
                         legend_title="Time Period",
                         xaxis_dtick=2.0,
                         yaxis_dtick=0.1)
fig2.show()

# Generate Net Empirical Delta Graph and Show
fig3 = create_line_graph(data=df0_net_delta,
                         title=f"Net Empirical Delta vs. SOL Price Change with Daily Fees: {round(debt_daily_cost * 100, 4)}% and Daily Yield: {round(trading_fee_daily_yield * 100, 4)}%",
                         xaxis_title="Percent Change in SOL Price",
                         yaxis_title="How 1$ Increase in SOL Price Affects your Equity Position",
                         legend_title="Time Period",
                         xaxis_dtick=2.0,
                         yaxis_dtick=0.1)
fig3.show()

# Generate Net Empirical Delta Graph and Show
fig4 = create_line_graph(data=df0_net_gamma,
                         title=f"Net Empirical Gamma vs. SOL Price Change with Daily Fees: {round(debt_daily_cost * 100, 4)}% and Daily Yield: {round(trading_fee_daily_yield * 100, 4)}%",
                         xaxis_title="Percent Change in SOL Price",
                         yaxis_title="How 1$ Increase in SOL Price Affects your Change in Equity Position",
                         legend_title="Time Period",
                         xaxis_dtick=2.0,
                         yaxis_dtick=0.01)
fig4.show()

df0_7_gamma.index = df0_7_gamma.index.round(2)
summary_df_7 = df0_7_gamma.loc[[180, 185, 190, 200, 205, 210], :].map(pd.to_numeric, errors='coerce').round(2)
summary_df_7.to_csv('summary_7d.csv')


print(f"The starting Price of USDC/SOL is: {P}")
print(f"The starting SOL Equity is: {X}")
print(f"The starting USDC Equity is: {Y}")
print(f"The starting Total Equity is: {TV_start}")
print(f"The starting SOL Debt Tokens is: {x_debt}")
print(f"The starting USDC Debt Tokens is: {y_debt}")
print(f"The starting Total Debt is: {tot_debt_val}")
print(f"The starting SOL to LP is: {x_to_lp}")
print(f"The starting USDC to LP is: {y_to_lp}")
print(f"The lower price bound of the CLAMM is is: {pa}")
print(f"The upper price bound of the CLAMM is is: {pb}")

'''
## Initial Params for example 1, 1 SOL and 200 USDC position
X=1   # SOL
Y = 200  # USDC
P = 200  # current price
TV_start = X * P + Y  # total value of position to start
# Leverage/Debt Params
leverage = 3  # amount of leverage (total position value)
x_pct = (X * P / (TV_start * (leverage - 1)))  # percent of leverage to take in token x
y_pct = 1 - x_pct  # percent of leverage to take in token y
x_debt = (leverage - 1) * x_pct * TV_start / P  # debt in x tokens
x_debt_val = x_debt * P  # debt of X valued in y tokens
y_debt = (leverage - 1) * y_pct * TV_start  # debt in y tokens
y_debt_val = y_debt  # debt in y tokens
tot_debt_val = x_debt_val + y_debt_val  # total debt value in y tokens
tot_lp_pos = TV_start + tot_debt_val  # total LP position

x_to_lp = X + x_debt
y_to_lp = tot_lp_pos - x_to_lp * P
pb = 210  # upper price tick on CLAMM LP position
# Get pa given X, Y, P, pb
pa = get_pa(x_to_lp, y_to_lp, P, pb)
# Get initial Total Value of LP position in Y Units

# Get Liquidity (L)
L = get_L(x_to_lp, y_to_lp, P, pa, pb)

df1 = get_LP_pos_price_range(price_range, pa, pb, L, x_debt, y_debt, debt_daily_cost, trading_fee_daily_yield, days, init_equity)
df1.loc[:, "Equity Return"] = df1.loc[:, 'Simplified Equity Value'] / TV_start - 1

'''
