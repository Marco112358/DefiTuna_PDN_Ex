import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = 'browser'


# https://atiselsts.github.io/pdfs/uniswap-v3-liquidity-math.pdf

def get_L(X: float,
          Y: float,
          P: float,
          pa: float,
          pb: float
          ) -> float:
    """
    Get the L (Liquidity) of the initial position
    X : float
        initial X value provided for liquidity
    Y : float
        initial Y value provided for liquidity
    P : float
        initial price value when liquidity was provided, x/y
    pa : float
         lower price value on CLAMM LP position
    pb : float
         upper price value on CLAMM LP position
    Returns
    -------
    float
        L, the liquidity of the initial position
    """
    Lx = X * (np.sqrt(P) * np.sqrt(pb)) / (np.sqrt(pb) - np.sqrt(P))
    Ly = Y / (np.sqrt(P) - np.sqrt(pa))
    L = min([Lx, Ly])
    return L


def get_y_lp_given_x(x: float,
                     p: float,
                     pa: float,
                     pb: float
                     ) -> float:
    """
    Get the y needed for the initial position
    X : float
        initial X value provided for liquidity
    P : float
        initial price value when liquidity was provided, x/y
    pa : float
         lower price value on CLAMM LP position
    pb : float
         upper price value on CLAMM LP position
    Returns
    -------
    float
        y
    """
    Lx = x * (np.sqrt(p) * np.sqrt(pb)) / (np.sqrt(pb) - np.sqrt(p))
    y = Lx * (np.sqrt(p) - np.sqrt(pa))
    return y


def get_pa(x: float,
           y: float,
           p: float,
           pb: float
           ) -> float:
    """
    Get the L (Liquidity) of the initial position
    x : float
        initial x value provided for liquidity
    y : float
        initial y value provided for liquidity
    p : float
        initial price value when liquidity was provided, x/y
    pb : float
         upper price value on CLAMM LP position
    Returns
    -------
    float
        pa, lower price value on CLAMM LP position
    """
    pa = (y / (np.sqrt(pb) * x) + np.sqrt(p) - y / (np.sqrt(p) * x)) ** 2
    return pa


def get_new_position(p_new: float,
                     pa: float,
                     pb: float,
                     L: float
                     ) -> dict:
    """
    Get the new x, y and total value (TV) of LP position given new price
    p_new : float
        new price of x/y
    pa : float
         lower price value on CLAMM LP position
    pb : float
         upper price value on CLAMM LP position
    L : float
        the liquidity of the initial position
    Returns
    -------
    dict
        {new x, new y, new total value}
    """
    if p_new <= pa:
        y_new = 0
        x_new = L * (np.sqrt(pb) - np.sqrt(pa)) / (np.sqrt(pa) * np.sqrt(pb))
    elif p_new >= pb:
        x_new = 0
        y_new = L * (np.sqrt(pb) - np.sqrt(pa))
    else:
        x_new = L * (np.sqrt(pb) - np.sqrt(p_new)) / (np.sqrt(p_new) * np.sqrt(pb))
        y_new = L * (np.sqrt(p_new) - np.sqrt(pa))

    tv_new = x_new * p_new + y_new
    return {'x': x_new,
            'y': y_new,
            'TV': tv_new}


def get_LP_pos_price_range(price_range: np.arange,
                           pa: float,
                           pb: float,
                           L: float,
                           init_x_debt: float,
                           init_y_debt: float,
                           debt_daily_cost: float,
                           trading_fee_daily_yield: float,
                           days: int,
                           init_equity: float
                           ) -> pd.DataFrame:
    """
    Get the new x, y and total value (TV) of LP position given new price
    price_range : np.arange
        range of new price of x/y
    pa : float
         lower price value on CLAMM LP position
    pb : float
         upper price value on CLAMM LP position
    L : float
        the liquidity of the initial position
    init_x_debt: float
        the initial debt in x tokens
    init_y_debt: float
        the initial debt in y tokens
    debt_daily_cost: float
        the daily interest cost on all debt, only being applied to total debt
    trading_fee_daily_yield: float
        the daily yield earned in trading fees, only being applied to total debt
    days: int
        the number of days the position is held
    init_equity: float
        the starting value of your equity position
    Returns
    -------
    pd.DataFrame
        dataframe of new x, new y, new total value
    """
    # Get LP Output
    df = pd.DataFrame(index=price_range,
                      columns=['X',
                               'X Value',
                               'Y',
                               'Y Value',
                               'TV',
                               'Initial X Debt Tokens',
                               'Initial Y Debt Tokens',
                               'Initial X Debt Value',
                               'Initial Y Debt Value',
                               'Initial Total Debt Value',
                               'Equity Value No Fees/Yield',
                               'TV with Trading Yield',
                               'Total Debt with Interest',
                               'Simplified Equity Value',
                               'Net X Tokens'])
    for prc in price_range:
        out = get_new_position(prc, pa, pb, L)
        df.loc[prc, 'X'] = out['x']
        df.loc[prc, 'Y'] = out['y']
        df.loc[prc, 'TV'] = out['TV']
    df.loc[:, 'X Value'] = df.loc[:, 'X'] * df.index
    df.loc[:, 'Y Value'] = df.loc[:, 'Y']
    df.loc[:, 'Initial X Debt Tokens'] = init_x_debt
    df.loc[:, 'Initial Y Debt Tokens'] = init_y_debt
    df.loc[:, 'Initial X Debt Value'] = df.loc[:, 'Initial X Debt Tokens'] * df.index
    df.loc[:, 'Initial Y Debt Value'] = df.loc[:, 'Initial Y Debt Tokens']
    df.loc[:, 'Initial Total Debt Value'] = df.loc[:, 'Initial X Debt Value'] + df.loc[:, 'Initial Y Debt Value']
    df.loc[:, 'Equity Value No Fees/Yield'] = df.loc[:, 'TV'] - df.loc[:, 'Initial Total Debt Value']
    df.loc[:, 'TV with Trading Yield'] = df.loc[:, 'TV'] * (1 + trading_fee_daily_yield) ** days
    df.loc[:, 'Total Debt with Interest'] = df.loc[:, 'Initial Total Debt Value'] * (1 + debt_daily_cost) ** days
    df.loc[:, 'Simplified Equity Value'] = df.loc[:, 'TV with Trading Yield'] - df.loc[:, 'Total Debt with Interest']
    df.loc[:, "Equity Return"] = (df.loc[:, 'Simplified Equity Value'] / init_equity - 1) * 100
    df.loc[:, 'Net X Tokens'] = df.loc[:, 'X'] - df.loc[:, 'Initial X Debt Tokens']
    return df


def add_empirical_delta(df):
    # Initialize the 'Net Empirical Delta' column with NaN
    df['Net Empirical Delta'] = np.nan

    # Iterate through the DataFrame to calculate Net Empirical Delta
    for i in range(1, len(df) - 1):  # Exclude the first and last rows
        # Current row
        current_price = df.index[i]
        current_eq_val = df.loc[current_price, 'Simplified Equity Value']
        # Previous row
        prev_price = df.index[i - 1]
        prev_eq_val = df.loc[prev_price, 'Simplified Equity Value']
        # Next row
        next_price = df.index[i + 1]
        next_eq_val = df.loc[next_price, 'Simplified Equity Value']

        # Changes in Net X Tokens
        change_prev = current_eq_val - prev_eq_val
        change_next = next_eq_val - current_eq_val

        # Changes in price
        price_change_prev = current_price - prev_price
        price_change_next = next_price - current_price

        # Calculate delta up and down
        delta_prev = change_prev / price_change_prev if price_change_prev != 0 else np.nan
        delta_next = change_next / price_change_next if price_change_next != 0 else np.nan

        # Average the deltas
        empirical_delta = np.nanmean([delta_prev, delta_next])

        # Assign the value to the DataFrame
        df.loc[current_price, 'Net Empirical Delta'] = empirical_delta
    return df


def add_empirical_gamma(df):
    # Initialize the 'Net Empirical Delta' column with NaN
    df['Net Empirical Gamma'] = np.nan

    # Iterate through the DataFrame to calculate Net Empirical Delta
    for i in range(1, len(df) - 1):  # Exclude the first and last rows
        # Current row
        current_price = df.index[i]
        current_net_delta = df.loc[current_price, 'Net Empirical Delta']
        # Previous row
        prev_price = df.index[i - 1]
        prev_net_delta = df.loc[prev_price, 'Net Empirical Delta']
        # Next row
        next_price = df.index[i + 1]
        next_net_delta = df.loc[next_price, 'Net Empirical Delta']

        # Changes in Net X Tokens
        change_prev = current_net_delta - prev_net_delta
        change_next = next_net_delta - current_net_delta

        # Changes in price
        price_change_prev = current_price - prev_price
        price_change_next = next_price - current_price

        # Calculate delta up and down
        gamma_prev = change_prev / price_change_prev if price_change_prev != 0 else np.nan
        gamma_next = change_next / price_change_next if price_change_next != 0 else np.nan

        # Average the deltas
        empirical_gamma = np.nanmean([gamma_prev, gamma_next])

        # Assign the value to the DataFrame
        df.loc[current_price, 'Net Empirical Gamma'] = empirical_gamma
    return df


def process_dataframe_summaries(
                          dfs: list[pd.DataFrame],
                          columns: list[str],
                          periods: list[str],
                          reference_price: float) -> pd.DataFrame:
    """
    Concatenate and rename columns for specific metrics from multiple DataFrames.

    Parameters
    ----------
    dfs : list of pd.DataFrame
        DataFrame containing data for 1 day.
    columns : list strings
        A list of column names to extract and concatenate from each DataFrame.
    periods: list of strings
        A list of the strings that reference the periods of the dfs
    reference_price : float
        The reference price used to normalize the index.

    Returns
   -------
    pd.DataFrame
        A new DataFrame with the concatenated and renamed columns and normalized index.
    """

    filtered_dfs = [df.loc[:, columns] for df in dfs]
    combined_df = pd.concat(filtered_dfs, axis=1)

    new_columns = [
        f"{period} - {col}"
        for period in periods
        for col in columns
    ]
    combined_df.columns = new_columns

    # Normalize index to percentage change from reference price
    combined_df.index = ((combined_df.index / reference_price - 1) * 100).round(4)
    return combined_df


def process_dataframe_summaries2(df1: pd.DataFrame,
                                 df7: pd.DataFrame,
                                 column_names: list,
                                 P: float) -> pd.DataFrame:
    """
    Concatenate and rename columns for specific metrics from multiple DataFrames.

    Parameters
    ----------
    df1 : pd.DataFrame
        DataFrame containing data for 1 day.
    df7 : pd.DataFrame
        DataFrame containing data for 7 days.
    df30 : pd.DataFrame
        DataFrame containing data for 30 days.
    column_names : list
        A list of column names to extract and concatenate from each DataFrame.
    P : float
        The reference price used to normalize the index.

    Returns
   -------
    pd.DataFrame
        A new DataFrame with the concatenated and renamed columns and normalized index.
    """
    # Filter the DataFrames to include only the specified columns
    filtered_df1 = df1.loc[:, column_names]
    filtered_df7 = df7.loc[:, column_names]

    # Concatenate the specified columns from each DataFrame
    combined_df = pd.concat([filtered_df1, filtered_df7], axis=1)

    # Rename columns
    new_column_names = [f"1 Day - {col}" for col in column_names] + \
                       [f"7 Days - {col}" for col in column_names]
    combined_df.columns = new_column_names

    # Normalize the index based on the reference price
    combined_df.index = (combined_df.index / P - 1).round(4) * 100
    return combined_df


def create_line_graph(data: pd.DataFrame,
                      title: str,
                      xaxis_title: str,
                      yaxis_title: str,
                      legend_title: str,
                      xaxis_dtick: float = None,
                      yaxis_dtick: float = None,
                      highlight_range: tuple = None,
                      highlight_color: str = 'rgba(255, 230, 153, 0.5)',
                      xaxis_min: float = None,
                      xaxis_max: float = None,
                      yaxis_min: float = None,
                      yaxis_max: float = None) -> go.Figure:
    """
    Create a line graph using Plotly from a given DataFrame with customizable gridlines,
    an optional highlighted vertical range, and axis ranges.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame containing the data to plot. Columns represent different series,
        and the index represents the x-axis values.
    title : str
        The title of the graph.
    xaxis_title : str
        The title for the x-axis.
    yaxis_title : str
        The title for the y-axis.
    legend_title : str
        The title for the legend.
    xaxis_dtick : float, optional
        Spacing between gridlines on the x-axis. If None, default spacing is used.
    yaxis_dtick : float, optional
        Spacing between gridlines on the y-axis. If None, default spacing is used.
    highlight_range : tuple, optional
        A tuple of two x-axis values (start, end) to highlight a vertical range. If None, no range is highlighted.
    highlight_color : str, optional
        The fill color for the highlighted range. Default is a semi-transparent yellow.
    xaxis_min : float, optional
        Minimum value for x-axis. If None, will auto-scale.
    xaxis_max : float, optional
        Maximum value for x-axis. If None, will auto-scale.
    yaxis_min : float, optional
        Minimum value for y-axis. If None, will auto-scale.
    yaxis_max : float, optional
        Maximum value for y-axis. If None, will auto-scale.

    Returns
    -------
    go.Figure
        A Plotly figure object representing the line graph.
    """
    # Create a Plotly Figure
    fig = go.Figure()

    # Add a line for each column in the DataFrame
    for column in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data.index,  # Use the index for the x-axis
                y=data[column],  # Use the column values for the y-axis
                mode='lines',
                name=column  # Use the column name as the legend label
            )
        )

    # Highlight the specified x-axis range if provided
    if highlight_range:
        start, end = highlight_range
        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",  # Extend the rectangle across the full y-axis
            x0=start,
            x1=end,
            y0=0,
            y1=1,
            fillcolor=highlight_color,
            opacity=0.5,
            layer="below",  # Place the rectangle below the lines
            line_width=0
        )

    # Update layout with custom titles and gridline settings
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white",
        legend_title=legend_title,
        xaxis=dict(
            tickmode='linear',  # Linear mode for evenly spaced ticks
            dtick=xaxis_dtick,  # Custom spacing if provided
            showgrid=True,  # Enable gridlines
            range=[xaxis_min, xaxis_max] if xaxis_min is not None and xaxis_max is not None else None
        ),
        yaxis=dict(
            tickmode='linear',  # Linear mode for evenly spaced ticks
            dtick=yaxis_dtick,  # Custom spacing if provided
            showgrid=True,  # Enable gridlines
            range=[yaxis_min, yaxis_max] if yaxis_min is not None and yaxis_max is not None else None
        )
    )
    return fig


def add_hodl_position(df: pd.DataFrame,
                      risky_token: float,
                      usdc: float,
                      p0: float):
    # Initialize the 'HODL Equity Value' column with NaN
    df['HODL Equity'] = risky_token * df.index + usdc
    df['HODL Return'] = (df['HODL Equity'] / (risky_token * p0 + usdc) - 1) * 100
    return df
