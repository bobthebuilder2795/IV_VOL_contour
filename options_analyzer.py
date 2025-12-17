import yfinance as yf
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys

def fetch_options_data(ticker_symbol):
    """
    Fetches options chain data for a given ticker from Yahoo Finance.
    Returns a DataFrame containing calls and puts for all available expirations.
    """
    print(f"Fetching data for {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        expirations = ticker.options
    except Exception as e:
        print(f"Error fetching expirations for {ticker_symbol}: {e}")
        return None

    if not expirations:
        print(f"No options data found for {ticker_symbol}")
        return None

    all_options = []
    today = datetime.now()

    # Yahoo Finance might return a lot of data, limit to reasonable range if needed
    # For now, we fetch all to build a complete surface
    
    total_expirations = len(expirations)
    for i, exp_date_str in enumerate(expirations):
        sys.stdout.write(f"\rProcessing expiration {i+1}/{total_expirations}: {exp_date_str}")
        sys.stdout.flush()
        
        try:
            # Parse expiration date
            exp_date = datetime.strptime(exp_date_str, '%Y-%m-%d')
            dte = (exp_date - today).days
            
            if dte < 0:
                continue # Skip expired

            # Fetch chain
            chain = ticker.option_chain(exp_date_str)
            
            # Process Calls
            calls = chain.calls.copy()
            calls['optionType'] = 'call'
            
            # Process Puts
            puts = chain.puts.copy()
            puts['optionType'] = 'put'
            
            # Combine
            opts = pd.concat([calls, puts])
            opts['expirationDate'] = exp_date
            opts['dte'] = dte
            
            all_options.append(opts)
            
        except Exception as e:
            print(f"\nError processing {exp_date_str}: {e}")
            continue
            
    print("\nData fetch complete.")
    
    if not all_options:
        return None

    df = pd.concat(all_options, ignore_index=True)
    
    # Ensure numeric types
    cols_to_numeric = ['strike', 'lastPrice', 'bid', 'ask', 'change', 'percentChange', 'volume', 'openInterest', 'impliedVolatility']
    for col in cols_to_numeric:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df

def process_data(df, min_volume=0, min_oi=0):
    """
    Cleans and filters the data.
    """
    # Filter out illiquid options if requested
    mask = (df['volume'].fillna(0) >= min_volume) & (df['openInterest'].fillna(0) >= min_oi)
    # Also filter out invalid IVs (e.g. 0 or very high which are often bad data)
    mask &= (df['impliedVolatility'] > 0.001) & (df['impliedVolatility'] < 5.0)
    
    cleaned_df = df[mask].copy()
    
    return cleaned_df

def interpolate_surface(df, value_col='impliedVolatility', grid_points=100):
    """
    Creates a grid for contour plotting using interpolation.
    Returns X (DTE), Y (Strike), Z (Value) grids.
    """
    # Points for interpolation
    x = df['dte'].values
    y = df['strike'].values
    z = df[value_col].values

    if len(x) < 10:
        return None, None, None

    # Create grid
    # We want the grid to span the min/max of our data
    xi = np.linspace(x.min(), x.max(), grid_points)
    yi = np.linspace(y.min(), y.max(), grid_points)
    X, Y = np.meshgrid(xi, yi)

    # Interpolate
    # 'linear' is often safer than 'cubic' for financial data to avoid wild oscillations
    # 'nearest' can be blocky. Let's try 'linear'.
    Z = griddata((x, y), z, (X, Y), method='linear')
    
    # Fill NaNs (extrapolation edges) if necessary, or leave them blank
    # For contour plots, blanks are fine (shows where data is missing)
    
    return X, Y, Z

def plot_surface(df, option_type, ticker, current_price=None):
    """
    Generates 2D contour plots for IV and Volume.
    """
    subset = df[df['optionType'] == option_type].copy()
    
    if subset.empty:
        print(f"No data for {option_type}s")
        return

    # --- Plot 1: Implied Volatility Surface ---
    X_iv, Y_iv, Z_iv = interpolate_surface(subset, 'impliedVolatility')
    
    if X_iv is None:
        print("Not enough data to plot IV surface.")
        return

    fig_iv = go.Figure(data=[
        go.Contour(
            z=Z_iv,
            x=X_iv[0], # DTEs (columns of meshgrid)
            y=Y_iv[:, 0], # Strikes (rows of meshgrid)
            colorscale='Viridis',
            contours=dict(
                start=np.nanmin(Z_iv),
                end=np.nanmax(Z_iv),
                size=(np.nanmax(Z_iv) - np.nanmin(Z_iv)) / 20,
            ),
            colorbar=dict(title='Implied Volatility'),
            hovertemplate='DTE: %{x:.1f}<br>Strike: %{y:.2f}<br>IV: %{z:.2%}<extra></extra>'
        )
    ])

    # Scatter points for actual data (optional, but good for verification)
    # fig_iv.add_trace(go.Scatter(
    #     x=subset['dte'],
    #     y=subset['strike'],
    #     mode='markers',
    #     marker=dict(
    #         size=2,
    #         color='rgba(255,255,255,0.3)'
    #     ),
    #     name='Actual Contracts',
    #     hovertemplate='DTE: %{x}<br>Strike: %{y}<br>IV: %{text:.2%}<extra></extra>',
    #     text=subset['impliedVolatility']
    # ))

    title_iv = f"{ticker} {option_type.title()}s - Implied Volatility Surface"
    if current_price:
        fig_iv.add_hline(y=current_price, line_dash="dash", line_color="white", annotation_text="Current Price")
        title_iv += f" (Current Price: {current_price})"

    fig_iv.update_layout(
        title=title_iv,
        xaxis_title="Days to Expiration (DTE)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        width=1000,
        height=800
    )
    
    # --- Plot 2: Liquidity (Volume) Heatmap ---
    # We map log volume because volume can vary by orders of magnitude
    subset['log_volume'] = np.log1p(subset['volume'])
    X_vol, Y_vol, Z_vol = interpolate_surface(subset, 'log_volume')

    fig_vol = go.Figure(data=[
        go.Contour(
            z=Z_vol,
            x=X_vol[0],
            y=Y_vol[:, 0],
            colorscale='Plasma',
            colorbar=dict(title='Log(Volume)'),
            hovertemplate='DTE: %{x:.1f}<br>Strike: %{y:.2f}<br>Log(Vol): %{z:.2f}<extra></extra>'
        )
    ])
    
    # Overlay real volume points sized by volume
    # fig_vol.add_trace(go.Scatter(
    #     x=subset['dte'],
    #     y=subset['strike'],
    #     mode='markers',
    #     marker=dict(
    #         size=np.log1p(subset['volume']),
    #         sizemode='area',
    #         sizeref=2.*max(np.log1p(subset['volume']))/(20.**2),
    #         sizemin=2,
    #         color=subset['volume'],
    #         colorscale='Plasma',
    #         showscale=False
    #     ),
    #     name='Volume',
    #     text=subset['volume'],
    #     hovertemplate='DTE: %{x}<br>Strike: %{y}<br>Vol: %{text}<extra></extra>'
    # ))

    fig_vol.update_layout(
        title=f"{ticker} {option_type.title()}s - Liquidity Map (Volume)",
        xaxis_title="Days to Expiration (DTE)",
        yaxis_title="Strike Price",
        template="plotly_dark",
        width=1000,
        height=800
    )
    
    if current_price:
        fig_vol.add_hline(y=current_price, line_dash="dash", line_color="white")

    fig_iv.show()
    fig_vol.show()

def get_current_price(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period='1d')
        if not hist.empty:
            return hist['Close'].iloc[-1]
        # Fallback to fast_info if history fails or is delayed
        return ticker.fast_info.last_price
    except:
        return None

def main():
    if len(sys.argv) > 1:
        ticker_symbol = sys.argv[1].upper()
    else:
        ticker_symbol = input("Enter Ticker Symbol (e.g. SPY, AAPL): ").upper().strip()
    
    current_price = get_current_price(ticker_symbol)
    print(f"Current Price of {ticker_symbol}: {current_price}")

    df = fetch_options_data(ticker_symbol)
    
    if df is None or df.empty:
        print("No data found.")
        return

    # Initial filtering
    df_clean = process_data(df, min_volume=1, min_oi=1)
    
    if df_clean.empty:
        print("No liquid data found after filtering.")
        return

    print(f"Plotting surfaces for {len(df_clean)} contracts...")
    
    # Ask user which type to plot or plot both
    plot_surface(df_clean, 'call', ticker_symbol, current_price)
    plot_surface(df_clean, 'put', ticker_symbol, current_price)
    
    print("\nDone. Check your browser for the plots.")
    print("Analysis Tips:")
    print("1. Look for 'Valleys' in IV (darker colors) near your target expiration/strike for BUYING.")
    print("2. Look for 'Peaks' in IV (lighter/yellow colors) for SELLING.")
    print("3. Ensure high Volume (Liquidity Map) at your target strike to ensure easy entry/exit.")

if __name__ == "__main__":
    main()

