import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_analytics import run_analytics

def plot_comprehensive_frontier(df_returns, optimizer, delta):
    # 1. Generate Frontier Data
    vols, rets, weights = optimizer.generate_frontier(n_points=100)
    
    # 2. Find Tangency Portfolio (Annualized rf = 4% -> Daily rf approx 0.00016)
    daily_rf = (1 + 0.04)**(1/252) - 1
    tangency = optimizer.find_tangency_portfolio(risk_free_rate=daily_rf)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 14), gridspec_kw={'hspace': 0.3})

    # --- Plot 1: The Efficient Frontier ---
    ax1.plot(vols * np.sqrt(252), rets * 252, 'b-', lw=3, label='Efficient Frontier')
    
    # Plot Individual Assets
    asset_vols = df_returns.std() * np.sqrt(252)
    asset_rets = df_returns.mean() * 252
    ax1.scatter(asset_vols, asset_rets, color='darkred', alpha=0.6, s=40)
    
    # Add Ticker Labels
    for i, txt in enumerate(optimizer.tickers):
        ax1.annotate(txt, (asset_vols.iloc[i], asset_rets.iloc[i]))

    # Plot Tangency Point
    ax1.scatter(tangency['volatility'] * np.sqrt(252), 
                tangency['return'] * 252, 
                color='gold', marker='*', s=400, label='Tangency Portfolio', edgecolors='black')

    ax1.set_title("Geometry of the Efficient Frontier (Annualized)", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Annualized Volatility (sigma)", fontsize=12)
    ax1.set_ylabel("Expected Annual Return (E[R])", fontsize=12)
    ax1.legend(loc='upper left', frameon=True, shadow=True)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- Plot 2: Weight Transition Plot ---
    df_w = pd.DataFrame(weights, index=rets * 252, columns=optimizer.tickers)
    df_w.plot.area(ax=ax2, colormap='tab20b', alpha=0.9)
    
    ax2.set_title("Asset Allocation Transition: Weight Dynamics", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Target Annualized Return", fontsize=12)
    ax2.set_ylabel("Portfolio Weight (w_i)", fontsize=12)
    ax2.legend(loc='upper center', 
    bbox_to_anchor=(0.5, -0.15), 
    ncol=5,                      
    fontsize=8, 
    frameon=True,
    title="Portfolio Constituents")

    ax2.set_ylim(0, 1)
    ax2.set_xlim(rets.min() * 252, rets.max() * 252)

    plt.savefig("efficient_frontier.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    df_returns, optimizer, delta = run_analytics()
    plot_comprehensive_frontier(df_returns, optimizer, delta)