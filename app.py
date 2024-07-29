import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import yfinance as yf


st.set_page_config(page_title="Option Pricing Calculator", layout="wide")

@st.cache_data
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

@st.cache_data
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = norm.cdf(d1) if option_type == 'call' else -norm.cdf(-d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type == 'call' else -d2)
    return delta, gamma, vega, theta, rho

@st.cache_data
def binomial_option_pricing(S, K, T, r, sigma, option_type='call', steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    
    ST = np.zeros((steps + 1, steps + 1))
    for i in range(steps + 1):
        for j in range(i + 1):
            ST[j, i] = S * (u ** (i - j)) * (d ** j)
    
    option_values = np.zeros((steps + 1, steps + 1))
    if option_type == 'call':
        option_values[:, steps] = np.maximum(0, ST[:, steps] - K)
    elif option_type == 'put':
        option_values[:, steps] = np.maximum(0, K - ST[:, steps])
    
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] + (1 - p) * option_values[j + 1, i + 1])
    
    return option_values[0, 0]

@st.cache_data
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

@st.cache_data
def get_historical_volatility(ticker):
    data = yf.download(ticker, period='1y', interval='1d')
    data['returns'] = data['Adj Close'].pct_change()
    volatility = np.std(data['returns']) * np.sqrt(252)
    return volatility

@st.cache_data
def implied_volatility(S, K, T, r, market_price, option_type='call'):
    def difference(sigma):
        return black_scholes(S, K, T, r, sigma, option_type) - market_price

    a = 1e-6
    b = 10.0

    try:
        return brentq(difference, a, b)
    except ValueError:
        return None  

# app layout
st.title("Option Pricing Calculator")

st.markdown("""
### Calculate the price of European call and put options using various models.
""")

with st.sidebar:
    st.markdown(
    """
    <div style='display: flex; align-items: left;'>
        Created by  <img src='https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png' alt='LinkedIn' style='width: 24px; margin-right: 10px;'>
        <a href='https://www.linkedin.com/in/pranav-muktevi' target='_blank'>Pranav Muktevi</a>
    </div>
    """,
    unsafe_allow_html=True
)
    st.header("Input Parameters")
    model = st.selectbox("Select Option Pricing Model", ["Black-Scholes", "Binomial", "Monte Carlo"])
    S = st.number_input("Current Stock Price (S)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0, key='S')
    K = st.number_input("Strike Price (K)", min_value=0.0, max_value=1000.0, value=100.0, step=1.0, key='K')
    T = st.number_input("Time to Maturity (T) in years", min_value=0.0, max_value=5.0, value=1.0, step=0.1, key='T')
    r = st.number_input("Risk-Free Interest Rate (r)", min_value=0.0, max_value=0.5, value=0.05, step=0.01, key='r')
    sigma = st.number_input("Volatility (sigma)", min_value=0.0, max_value=1.0, value=0.2, step=0.01, key='sigma')
    option_type = st.selectbox("Option Type", ["call", "put"], key='option_type')
    real_price = st.number_input("Real-time Option Price", min_value=0.0, value=0.0, key='real_price')
    ticker = st.text_input("Enter Ticker Symbol", "AAPL", key='ticker')
    


# plot selection
plot_options = {
    "Black-Scholes": ["Option Price Heatmap", "Option Price vs Stock Price", "Option Price vs Interest Rate", "Sensitivity Analysis"],
    "Binomial": ["Option Price Heatmap", "Option Price vs Stock Price", "Option Price vs Interest Rate", "Sensitivity Analysis"],
    "Monte Carlo": [f"Monte Carlo {option_type.capitalize()} Option Price"]
}

plot_type = st.selectbox("Select plot type to display", plot_options[model], key='plot_type')

if 'calc' not in st.session_state:
    st.session_state.calc = False

if st.button("Calculate") or st.session_state.calc:
    st.session_state.calc = True

    option_price = None
    delta, gamma, vega, theta, rho = [None] * 5
    hist_vol = None
    imp_vol = None

    if model == "Black-Scholes":
        option_price = black_scholes(S, K, T, r, sigma, option_type)
        delta, gamma, vega, theta, rho = black_scholes_greeks(S, K, T, r, sigma, option_type)
        hist_vol = get_historical_volatility(ticker)
        try:
            imp_vol = implied_volatility(S, K, T, r, real_price, option_type)
        except ValueError:
            imp_vol = None
        st.write(f"### The {option_type} option price using Black-Scholes is: ${option_price:.2f}")

    elif model == "Binomial":
        option_price = binomial_option_pricing(S, K, T, r, sigma, option_type)
        st.write(f"### The {option_type} option price using Binomial Model is: ${option_price:.2f}")

    elif model == "Monte Carlo":
        num_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000, key='num_simulations')
        option_price = monte_carlo_simulation(S, K, T, r, sigma, option_type, num_simulations)
        st.write(f"### The {option_type} option price using Monte Carlo Simulation is: ${option_price:.2f}")

    tolerance = 0.01
    if abs(option_price - real_price) <= tolerance:
        st.success("The real-time option price is close to the predicted price within the tolerance level.")
    else:
        st.warning("The real-time option price is not close to the predicted price within the tolerance level.")

    data = {
        "Parameter": ["Current Stock Price", "Strike Price", "Time to Maturity (Years)", "Risk-Free Interest Rate", "Volatility", "Option Type", "Real-time Option Price", "Predicted Option Price", "Delta", "Gamma", "Vega", "Theta", "Rho", "Historical Volatility", "Implied Volatility", "Monte Carlo Option Price"],
        "Value": [S, K, T, r, sigma, option_type, real_price, option_price, delta if model == "Black-Scholes" else "N/A", gamma if model == "Black-Scholes" else "N/A", vega if model == "Black-Scholes" else "N/A", theta if model == "Black-Scholes" else "N/A", rho if model == "Black-Scholes" else "N/A", hist_vol if model == "Black-Scholes" else "N/A", imp_vol if imp_vol is not None else "N/A", option_price if model == "Monte Carlo" else "N/A"]
    }
    
    df = pd.DataFrame(data)
    st.table(df)

    # Plot based on your selection
    if model == "Black-Scholes" or model == "Binomial":
        if plot_type == "Option Price Heatmap":
            T_min = st.slider("Minimum Time to Maturity", 0.1, 5.0, 0.1, key='T_min')
            T_max = st.slider("Maximum Time to Maturity", 0.1, 5.0, 5.0, key='T_max')
            sigma_min = st.slider("Minimum Volatility", 0.01, 1.0, 0.01, key='sigma_min')
            sigma_max = st.slider("Maximum Volatility", 0.01, 1.0, 1.0, key='sigma_max')
            
            T_range = np.linspace(T_min, T_max, 100)
            sigma_range = np.linspace(sigma_min, sigma_max, 100)
            
            T_grid, sigma_grid = np.meshgrid(T_range, sigma_range)
            if model == "Black-Scholes":
                option_prices = np.zeros_like(T_grid)
                for i in range(T_grid.shape[0]):
                    for j in range(T_grid.shape[1]):
                        option_prices[i, j] = black_scholes(S, K, T_grid[i, j], r, sigma_grid[i, j], option_type)
            else:
                option_prices = np.zeros_like(T_grid)
                for i in range(T_grid.shape[0]):
                    for j in range(T_grid.shape[1]):
                        option_prices[i, j] = binomial_option_pricing(S, K, T_grid[i, j], r, sigma_grid[i, j], option_type)
            
            fig = go.Figure(data=[go.Surface(z=option_prices, x=T_range, y=sigma_range)])
            fig.update_layout(title='Option Price Heatmap', autosize=True, scene=dict(
                              xaxis_title='Time to Maturity (T)', yaxis_title='Volatility (σ)', zaxis_title='Option Price'))
            st.plotly_chart(fig)

        elif plot_type == "Option Price vs Stock Price":
            S_range = np.linspace(0.1 * S, 2 * S, 100)
            if model == "Black-Scholes":
                option_prices = [black_scholes(s, K, T, r, sigma, option_type) for s in S_range]
            else:
                option_prices = [binomial_option_pricing(s, K, T, r, sigma, option_type) for s in S_range]
            fig = go.Figure(data=[go.Scatter(x=S_range, y=option_prices, mode='lines')])
            fig.update_layout(title='Option Price vs Stock Price', xaxis_title='Stock Price (S)', yaxis_title='Option Price')
            st.plotly_chart(fig)

        elif plot_type == "Option Price vs Interest Rate":
            r_range = np.linspace(0.0, 0.2, 100)
            if model == "Black-Scholes":
                option_prices = [black_scholes(S, K, T, rate, sigma, option_type) for rate in r_range]
            else:
                option_prices = [binomial_option_pricing(S, K, T, rate, sigma, option_type) for rate in r_range]
            fig = go.Figure(data=[go.Scatter(x=r_range, y=option_prices, mode='lines')])
            fig.update_layout(title='Option Price vs Interest Rate', xaxis_title='Interest Rate (r)', yaxis_title='Option Price')
            st.plotly_chart(fig)
        
        elif plot_type == "Sensitivity Analysis":
            fig = go.Figure()
            
            S_range = np.linspace(0.1 * S, 2 * S, 100)
            if model == "Black-Scholes":
                option_prices_S = [black_scholes(s, K, T, r, sigma, option_type) for s in S_range]
            else:
                option_prices_S = [binomial_option_pricing(s, K, T, r, sigma, option_type) for s in S_range]
            fig.add_trace(go.Scatter(x=S_range, y=option_prices_S, mode='lines', name='Stock Price Sensitivity'))
            
            r_range = np.linspace(0.0, 0.2, 100)
            if model == "Black-Scholes":
                option_prices_r = [black_scholes(S, K, T, rate, sigma, option_type) for rate in r_range]
            else:
                option_prices_r = [binomial_option_pricing(S, K, T, rate, sigma, option_type) for rate in r_range]
            fig.add_trace(go.Scatter(x=r_range, y=option_prices_r, mode='lines', name='Interest Rate Sensitivity'))

            fig.update_layout(title='Sensitivity Analysis', xaxis_title='Value', yaxis_title='Option Price')
            st.plotly_chart(fig)

    elif model == "Monte Carlo":
        if plot_type == f"Monte Carlo {option_type.capitalize()} Option Price":
            num_simulations = st.number_input("Number of Simulations", min_value=1000, max_value=100000, value=10000, step=1000, key='mc_num_simulations')
            mc_price = monte_carlo_simulation(S, K, T, r, sigma, option_type, num_simulations)
            st.write(f"### Monte Carlo Simulation Option Price ({num_simulations} simulations): ${mc_price:.2f}")

            # Optional: Plot distribution of simulation results
            bins = st.slider("Number of Bins", min_value=10, max_value=100, value=50, step=1, key='mc_bins')
            price_min = st.slider("Min Option Price", min_value=0.0, max_value=2*S, value=0.0, step=0.1, key='price_min')
            price_max = st.slider("Max Option Price", min_value=0.0, max_value=2*S, value=2*S, step=0.1, key='price_max')
            
            prices = np.array([monte_carlo_simulation(S, K, T, r, sigma, option_type, 1) for _ in range(num_simulations)])
            fig = go.Figure(data=[go.Histogram(x=prices, nbinsx=bins)])
            fig.update_layout(title='Monte Carlo Simulation Results', xaxis_title='Option Price', yaxis_title='Frequency', xaxis=dict(range=[price_min, price_max]))
            st.plotly_chart(fig)

# add on (bonus) Greeks visualization for Black-Scholes model
if model == "Black-Scholes" and plot_type == "Sensitivity Analysis":
    S_range = np.linspace(0.5 * S, 1.5 * S, 100)
    delta_values = []
    gamma_values = []
    vega_values = []
    theta_values = []
    rho_values = []

    for s in S_range:
        delta, gamma, vega, theta, rho = black_scholes_greeks(s, K, T, r, sigma, option_type)
        delta_values.append(delta)
        gamma_values.append(gamma)
        vega_values.append(vega)
        theta_values.append(theta)
        rho_values.append(rho)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_range, y=delta_values, mode='lines', name='Delta'))
    fig.add_trace(go.Scatter(x=S_range, y=gamma_values, mode='lines', name='Gamma'))
    fig.add_trace(go.Scatter(x=S_range, y=vega_values, mode='lines', name='Vega'))
    fig.add_trace(go.Scatter(x=S_range, y=theta_values, mode='lines', name='Theta'))
    fig.add_trace(go.Scatter(x=S_range, y=rho_values, mode='lines', name='Rho'))
    
    fig.update_layout(title='Greeks vs Stock Price', xaxis_title='Stock Price (S)', yaxis_title='Greek Value')
    st.plotly_chart(fig)


