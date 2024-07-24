import streamlit as st
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

# Set page configuration
st.set_page_config(page_title="Black-Scholes Option Pricing", layout="wide")

def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    
    return option_price

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    return delta, gamma, vega, theta, rho

def get_historical_volatility(ticker):
    data = yf.download(ticker, period='1y', interval='1d')
    data['returns'] = data['Adj Close'].pct_change()
    volatility = np.std(data['returns']) * np.sqrt(252)
    return volatility

def implied_volatility(S, K, T, r, market_price, option_type='call'):
    def difference(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price
    # Initial guesses for sigma
    a = 1e-6
    b = 1.0

    # Check signs
    while difference(a) * difference(b) > 0:
        b *= 2
        if b > 100:  # To prevent infinite loop
            raise ValueError("Could not bracket the root within reasonable sigma bounds.")

    return brentq(difference, a, b)

def monte_carlo_simulation(S, K, T, r, sigma, option_type='call', num_simulations=10000):
    dt = T / num_simulations
    prices = np.zeros(num_simulations)
    prices[0] = S
    for t in range(1, num_simulations):
        prices[t] = prices[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.normal())
    if option_type == 'call':
        payoff = np.maximum(prices[-1] - K, 0)
    else:
        payoff = np.maximum(K - prices[-1], 0)
    return np.exp(-r * T) * payoff

# Streamlit app layout
st.title("Black-Scholes Option Pricing Calculator")

st.markdown("""
### Calculate the price of European call and put options using the Black-Scholes formula.
""")

with st.sidebar:
    st.header("Input Parameters")
    S = st.slider("Current Stock Price (S)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    K = st.slider("Strike Price (K)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0)
    T = st.slider("Time to Maturity (T) in years", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    r = st.slider("Risk-Free Interest Rate (r)", min_value=0.0, max_value=0.5, value=0.05, step=0.01)
    sigma = st.slider("Volatility (sigma)", min_value=0.0, max_value=1.0, value=0.2, step=0.01)
    option_type = st.selectbox("Option Type", ["call", "put"])
    real_price = st.number_input("Real-time Option Price", min_value=0.0, value=0.0)
    ticker = st.text_input("Enter Ticker Symbol", "AAPL")

if st.button("Calculate"):
    option_price = black_scholes(S, K, T, r, sigma, option_type)
    st.write(f"### The {option_type} option price is: ${option_price:.2f}")

    tolerance = 0.01  # Tolerance level for price match
    if abs(option_price - real_price) <= tolerance:
        st.success("The real-time option price matches the predicted price within the tolerance level.")
    else:
        st.warning("The real-time option price does not match the predicted price within the tolerance level.")
    
    # Calculate Greeks
    delta, gamma, vega, theta, rho = black_scholes_greeks(S, K, T, r, sigma, option_type)
    st.write(f"### Delta: {delta:.2f}")
    st.write(f"### Gamma: {gamma:.2f}")
    st.write(f"### Vega: {vega:.2f}")
    st.write(f"### Theta: {theta:.2f}")
    st.write(f"### Rho: {rho:.2f}")

    # Calculate Historical Volatility
    hist_vol = get_historical_volatility(ticker)
    st.write(f"### Historical Volatility: {hist_vol:.2%}")
    
    # Calculate Implied Volatility
    imp_vol = implied_volatility(S, K, T, r, real_price, option_type)
    st.write(f"### Implied Volatility: {imp_vol:.2%}")

    # Run Monte Carlo Simulation
    mc_price = monte_carlo_simulation(S, K, T, r, sigma, option_type)
    st.write(f"### Monte Carlo {option_type.capitalize()} Option Price: ${mc_price:.2f}")

    # Plotting the option price sensitivity
    x = np.linspace(0, 5, 100)
    y = [black_scholes(S, K, t, r, sigma, option_type) for t in x]

    fig, ax = plt.subplots()
    ax.plot(x, y, label='Option Price', color='blue')
    ax.set_xlabel('Time to Maturity (Years)')
    ax.set_ylabel('Option Price')
    ax.legend()
    st.pyplot(fig)

    # Creating a heat map
    T_values = np.linspace(0.01, 2, 50)
    sigma_values = np.linspace(0.01, 1, 50)
    T_grid, sigma_grid = np.meshgrid(T_values, sigma_values)
    Z = np.array([[black_scholes(S, K, T_grid[i, j], r, sigma_grid[i, j], option_type) for j in range(T_grid.shape[1])] for i in range(T_grid.shape[0])])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(Z, xticklabels=np.round(T_values, 2), yticklabels=np.round(sigma_values, 2), cmap="YlGnBu", ax=ax)
    ax.set_xlabel("Time to Maturity (Years)")
    ax.set_ylabel("Volatility (Sigma)")
    st.write("### Option Price Heatmap")
    st.pyplot(fig)

    st.markdown("""
    ## Sensitivity Analysis
    ### Stock Price
    """)

    # Sensitivity analysis for Stock Price
    S_values = np.linspace(0, 200, 100)
    prices = [black_scholes(S_val, K, T, r, sigma, option_type) for S_val in S_values]

    fig, ax = plt.subplots()
    ax.plot(S_values, prices, label='Option Price vs Stock Price')
    ax.set_xlabel('Stock Price (S)')
    ax.set_ylabel('Option Price')
    ax.legend()
    st.pyplot(fig)

    st.markdown("""
    ### Interest Rate
    """)

    # Sensitivity analysis for Interest Rate
    r_values = np.linspace(0, 0.2, 100)
    prices = [black_scholes(S, K, T, r_val, sigma, option_type) for r_val in r_values]

    fig, ax = plt.subplots()
    ax.plot(r_values, prices, label='Option Price vs Interest Rate')
    ax.set_xlabel('Risk-Free Interest Rate (r)')
    ax.set_ylabel('Option Price')
    ax.legend()
    st.pyplot(fig)

with st.expander("See the formula"):
    st.markdown("""
    The Black-Scholes formula for a call option is:

    $$C = S \cdot N(d1) - K \cdot e^{-r \cdot T} \cdot N(d2)$$

    Where:

    $$d1 = \frac{\log(S / K) + (r + \sigma^2 / 2) \cdot T}{\sigma \cdot \sqrt{T}}$$

    $$d2 = d1 - \sigma \cdot \sqrt{T}$$
    """)
