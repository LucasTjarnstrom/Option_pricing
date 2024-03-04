import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# Constants


riskfree_rate_usd = riskfree_rate_eur = 0.03
eurusd_fx = 1.1
eurusd_sigma = 0.1
eurusd_rho = 0.5
q = 0


# Functions for data management and processing


def read_csv(file: str) -> pd.DataFrame:
    
    """ 
    Read a csv file and return a pandas DataFrame.
    
    :param file: File name
    """
    
    data = pd.read_csv(file, sep=",")
    return data


def prepare_data(data_frame: pd.DataFrame, col_name: str, func) -> pd.DataFrame:
    
    """ 
    Create a new column in a pandas DataFrame where the column value is the output
    of the passed function.
    
    :param data_frame: A pandas DataFrame 
    :param col_name: Column name in DataFrame
    """
    
    data_frame[col_name] = data_frame.apply(func, axis=1)
    return data_frame


def create_output(*input_file: str, output_file: str) -> pd.DataFrame:
    
    """ 
    Create the output file and write csv to disk.
    
    :param input_file: Names of the csv files to be read
    :param output_file: Name of the csv output file
    """
    
    lst = [read_csv(input_file) for input_file in input_file]
    output_data = pd.merge(lst[0], lst[1])
    
    output_data = prepare_data(output_data, 'price', lambda row: option_pricer(row['spot_price'], 
                                                                               row['strike'], 
                                                                               row['expiry'],
                                                                               row['payment_time'],
                                                                               riskfree_rate_usd,
                                                                               riskfree_rate_eur,
                                                                               q,
                                                                               row['volatility'],
                                                                               eurusd_sigma,
                                                                               eurusd_rho,
                                                                               eurusd_fx,
                                                                               row['call_put'],
                                                                               row['type']))

    output_data['pv'] = output_data.eval("price*quantity")
    
    output_data = prepare_data(output_data, 'equity_delta', lambda row: delta(row['spot_price'], 
                                                                              row['strike'], 
                                                                              row['expiry'],
                                                                              riskfree_rate_usd,
                                                                              q,
                                                                              row['volatility'], 
                                                                              row['call_put'],
                                                                              row['type']) * row['pv'])
    
    output_data = prepare_data(output_data, 'equity_vega', lambda row: vega(row['spot_price'], 
                                                                            row['strike'], 
                                                                            row['expiry'],
                                                                            row['payment_time'],
                                                                            riskfree_rate_usd,
                                                                            riskfree_rate_eur,
                                                                            q,
                                                                            row['volatility'],
                                                                            eurusd_sigma,
                                                                            eurusd_rho,
                                                                            row['call_put'],
                                                                            row['type']) * row['pv'])
    
    
    output_data = pd.concat([output_data.trade_id, output_data.pv, output_data.equity_delta, output_data.equity_vega], axis=1)
    
    output_data.to_csv(output_file)
    
    return output_data


# Pricing functions


def option_pricer(S: float, 
                  K: float, 
                  T: float, 
                  T_p: float, 
                  r_d: float, 
                  r_fx: float, 
                  q: float, 
                  sigma: float, 
                  sigma_fx: float, 
                  rho: float, 
                  fx_rate: float, 
                  option_style: str,
                  option_type: str):
    
    """ 
    Pass pricing parameters to Black-Scholes or Quanto model.
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param T_p: Time to delivery
    :param r_d: Domestic interest rate
    :param r_fx: Foreign interest rate
    :param q: Dividend yield
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param sigma_fx: Volatility in the returns of the FX rate
    :param rho: Correlation between the returns of the underlying and the FX rate
    :param fx_rate: FX rate 
    :param option_style: Option style, call or put
    :param option_type: Option type, REGULAR or ODD
    """
    
    if option_type == "REGULAR":
        return bs_price(S, K, T, T_p, r_d, sigma, option_style)
    elif option_type == "ODD":
        return quanto_price(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, fx_rate, option_style)
    else: 
        raise TypeError("No matching option style or type.")


def bs_price(S: float, 
             K: float, 
             T: float, 
             T_p: float, 
             r: float, 
             sigma: float,
             option_style: str) -> float:
    
    """ 
    Calculate option price given the Black-Scholes model. Returns the price. 
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param T_p: Time to delivery
    :param r: Interest rate
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param option_style: Option style, call or put
    """
    
    d1 = (np.log(S/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_style == "CALL":
        price = S*norm.cdf(d1,0,1) - K*np.exp(-r*T_p)*norm.cdf(d2,0,1)
    elif option_style == "PUT":
        price = K*np.exp(-r*T_p)*norm.cdf(-d2,0,1) - S*norm.cdf(-d1,0,1)
    return price


def quanto_price(S: float, 
                 K: float, 
                 T: float, 
                 T_p: float, 
                 r_d: float,
                 r_fx: float,
                 q: float, 
                 sigma: float, 
                 sigma_fx: float, 
                 rho: float, 
                 fx_rate: float, 
                 option_style: str) -> float:
    
    """ 
    Calculates the price of a Quanto style option. Returns the price.
    
    Reference for implementation is Baxter, M. & Rennie, A (1996) "Financial Calculus - An Introduction to Derivative Pricing" pp. 122-127 
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param T_p: Time to delivery
    :param r_d: Domestic interest rate
    :param r_fx: Foreign interest rate
    :param q: Dividend yield
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param sigma_fx: Volatility in the returns of the FX rate
    :param rho: Correlation between the returns of the underlying and the FX rate
    :param fx_rate: FX rate
    :param option_style: Option style, call or put
    """
    
    F = S*np.exp((r_fx-q)*T_p)*np.exp(-rho*sigma*sigma_fx*T_p)
    
    d1 = (np.log(F/K) + ((sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2 = d1-sigma*np.sqrt(T)
    
    if option_style == "CALL":
        price = fx_rate*np.exp(-r_d*T_p)*(F*norm.cdf(d1,0,1) - K*norm.cdf(d2,0,1))
    elif option_style == "PUT":
        price = fx_rate*np.exp(-r_d*T_p)*(K*norm.cdf(-d2,0,1) - F*norm.cdf(-d1,0,1))
    return price


def delta(S: float, 
          K: float, 
          T: float, 
          r: float, 
          q: float, 
          sigma: float, 
          option_style: str, 
          option_type: str) -> float:
    
    """ 
    Pass parameters to the Black-Scholes or Black model delta calculator function.
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param r: Domestic interest rate
    :param q: Dividend yield
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param option_style: Option style, call or put
    :param option_type: Option type, REGULAR or ODD
    """
    
    if option_type == "REGULAR":
        return bs_delta(S, K, T, r, sigma, option_style)
    elif option_type == "ODD":
        return black_delta(S, K, T, r, q, sigma, option_style)
    else: 
        raise TypeError("No matching option style or type.")


def vega(S: float, 
         K: float, 
         T: float, 
         T_p: float, 
         r_d: float, 
         r_fx: float, 
         q: float, 
         sigma: float,
         sigma_fx: float, 
         rho: float, 
         option_style: str, 
         option_type: str) -> float:
    
    """ 
    Pass parameters to the Black-Scholes or Black model vega calculator function.
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param T_p: Time to delivery
    :param r_d: Domestic interest rate
    :param r_fx: Foreign interest rate
    :param q: Dividend yield
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param sigma_fx: Volatility in the returns of the FX rate
    :param rho: Correlation between the returns of the underlying and the FX rate
    :param option_style: Option style, call or put
    :param option_type: Option type, REGULAR or ODD
    """
    
    if option_type == "REGULAR":
        return bs_vega(S, K, T, r_d, sigma, option_style)
    elif option_type == "ODD":
        return black_vega(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, option_style)
    else: 
        raise TypeError("No matching option style or type.")


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, option_style: str) -> float:
    
    """ 
    Calculate option delta given Black-Scholes dynamics. 
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param r: Interest rate
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param option_style: Option style, call or put
    """
    
    d1 = (np.log(S/K) + (r + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    
    if option_style == "CALL":
        delta = norm.cdf(d1,0,1)
    elif option_style == "PUT":
        delta = -norm.cdf(-d1,0,1)
    return delta


def bs_vega(S: float, K: float, T: float, r: float, sigma: float, option_style: str) -> float:
    
    """ 
    Calculate option vega given Black-Scholes dynamics. 
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param r: Interest rate
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param option_style: Option style, call or put
    """
    
    d1 = (np.log(S/K) + (r - q + (sigma**2)/2)*T)/(sigma*np.sqrt(T))
    return S*norm.pdf(d1,0,1)*np.sqrt(T)/100


def black_delta(S: float, K: float, T: float, r: float, q: float, sigma: float, option_style: str) -> float:
    
    """
    Calculates the option vega for the Black model.
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param r: Domestic interest rate
    :param q: Dividend yield
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param option_style: Option style, call or put
    """
    
    F = S*np.exp((r-q)*T)
    d1 = (np.log(F/K) + (sigma**2)/2*T)/(sigma*np.sqrt(T))
    
    if option_style == "CALL":
        delta = np.exp(-r*T)*norm.cdf(d1,0,1)
    elif option_style == "PUT":
        delta = -np.exp(-r*T)*norm.cdf(-d1,0,1)
    return delta


def black_vega(S: float, 
               K: float, 
               T: float, 
               T_p: float, 
               r_d: float, 
               r_fx: float, 
               q: float, 
               sigma: float, 
               sigma_fx: float, 
               rho: float, 
               option_style: str) -> float:
    
    """
    Calculates the option vega for the Black model.
    
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param T_p: Time to delivery
    :param r_d: Domestic interest rate
    :param r_fx: Foreign interest rate
    :param q: Dividend yield
    :param sigma: Volatility in the returns of the underlying or implied volatility
    :param sigma_fx: Volatility in the returns of the FX rate
    :param rho: Correlation between the returns of the underlying and the FX rate
    :param option_style: Option style, call or put
    """
    
    F = S*np.exp((r_fx-q)*T_p)*np.exp(-rho*sigma*sigma_fx*T_p)
    d1 = (np.log(F/K) + (sigma**2)/2*T)/(sigma*np.sqrt(T))
    return F*np.exp(-r_d*T_p)*norm.pdf(d1,0,1)*np.sqrt(T)/100


def implied_volatility(market_price: float, 
                       S: float, 
                       K: float, 
                       T: float, 
                       T_p: float,
                       r: float, 
                       option_style: str, 
                       max_iter = 1000) -> float:
    
    """ 
    Calculates implied volatility using the brentq algorithm. Calibrates the implied volatility
    to the observable market price.
    
    :param market_price: Market price of the option
    :param S: Underlying asset price
    :param K: Strike price
    :param T: Time to expiry
    :param T_p: Time to delivery
    :param r: Interest rate
    :param option_style: Option style, call or put
    :param max_iter: Maximum number of iterations that the optimization algorithm will run
    """
    
    lower_bound = 1e-8
    upper_bound = 100
    
    func = lambda sigma: market_price - bs_price(S, K, T, T_p, r, sigma, option_style)
    return brentq(func, lower_bound, upper_bound, maxiter=max_iter)


# Validation


def validate_models():
    """ Check model output prices against correct results. """
    
    tol = 1e-4
    
    def validate_price(func, result: float):
        if np.abs(func-result) < tol:
            return "Passed"
        else:
            return "Failed"
    
    # Black-Scholes model test data
    
    S = 60; K = 65; T = 0.25; T_p = 0.25; r_d = 0.08; r_fx = 0; q = 0; sigma = 0.3; sigma_fx = 0; rho = 0; fx_rate = 0;
    
    BS_call = option_pricer(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, fx_rate, option_style="CALL", option_type="REGULAR")
    BS_call_res = 2.133372
    BS_put = option_pricer(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, fx_rate, option_style="PUT", option_type="REGULAR")
    BS_put_res = 5.846286
    
    BS_delta_call = bs_delta(S, K, T, r_d, sigma, option_style="CALL")
    BS_delta_call_res = 0.3724829
    BS_delta_put = bs_delta(S, K, T, r_d, sigma, option_style="PUT")
    BS_delta_put_res = -0.6275171

    BS_vega_call = bs_vega(S, K, T, r_d, sigma, option_style="CALL")
    BS_vega_call_res = 11.35154/100
    BS_vega_put = bs_vega(S, K, T, r_d, sigma, option_style="PUT")
    BS_vega_put_res = 11.35154/100

    # Black model test data
    
    S = 48.06673213004505; K = 50; T = 0.3846; T_p = 0.3846; r_d = 0.05; r_fx = 0.05; q = 0; sigma = 0.2;
    sigma_fx = 0.3; rho = 0;
    
    Black_delta_call = black_delta(S, K, T, r_d, q, sigma, option_style="CALL")
    Black_delta_call_res = 0.45107017482201828
    Black_delta_put = black_delta(S, K, T, r_d, q, sigma, option_style="PUT")
    Black_delta_put_res = -0.5298835421176763

    Black_vega_call = black_vega(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, option_style="CALL")
    Black_vega_call_res = 0.118317785624
    Black_vega_put = black_vega(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, option_style="PUT")
    Black_vega_put_res = 0.118317785624

    # Implied volatility test data
    
    market_price_call = 2.13372; market_price_put = 5.846286; S = 60; K = 65; T = 0.25; T_p = 0.25; r = 0.08;
    
    iv_call = implied_volatility(market_price_call, S, K, T, T_p, r, "CALL")
    iv_call_res = 0.3
    iv_put = implied_volatility(market_price_put, S, K, T, T_p, r, "PUT")
    iv_put_res = 0.3

    # Quanto test data
    # Example prices taken from E.G Haug "The Complete Guide to Option Pricing Formula"
    
    S = 100; K = 105; T = 0.5; T_p = 0.5; r_d = 0.08; r_fx = 0.05; q = 0.04; sigma = 0.2; sigma_fx = 0.1; rho = 0.3; fx_rate = 1;
    
    Quanto_call = option_pricer(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, fx_rate, option_style="CALL", option_type="ODD")
    Quanto_call_res = 3.5520
    Quanto_put = option_pricer(S, K, T, T_p, r_d, r_fx, q, sigma, sigma_fx, rho, fx_rate, option_style="PUT", option_type="ODD")
    Quanto_put_res = 8.1636

    Quanto_call2 = option_pricer(S=100, K=100, T=0.5, T_p=0.5, r_d=0.05, r_fx=0.04, q=0.08, sigma=0.2, sigma_fx=0.3, rho=0.3, fx_rate=1.5, option_style="CALL", option_type="ODD")
    Quanto_call2_res = 6.2083
    Quanto_put2 = option_pricer(S=100, K=100, T=0.5, T_p=0.5, r_d=0.05, r_fx=0.04, q=0.08, sigma=0.2, sigma_fx=0.3, rho=0.3, fx_rate=1.5, option_style="PUT", option_type="ODD")
    Quanto_put2_res = 10.3900

    
    print("----- BS model -----" \
          "\nCall: ", validate_price(BS_call, BS_call_res), \
          "\nPut: ", validate_price(BS_put, BS_put_res), \
          "\nDelta (call): ", validate_price(BS_delta_call, BS_delta_call_res), \
          "\nDelta (put): ", validate_price(BS_delta_put, BS_delta_put_res), \
          "\nVega (call): ", validate_price(BS_vega_call, BS_vega_call_res), \
          "\nVega (put): ", validate_price(BS_vega_put, BS_vega_put_res), \
          "\n----- Black model -----" \
          "\nDelta (call): ", validate_price(Black_delta_call, Black_delta_call_res), \
          "\nDelta (put): ", validate_price(Black_delta_put, Black_delta_put_res), \
          "\nVega (call): ", validate_price(Black_vega_call, Black_vega_call_res), \
          "\nVega (put): ", validate_price(Black_vega_put, Black_vega_put_res), \
          "\n----- Implied volatility -----" \
          "\nImplied vol (call): ", validate_price(iv_call, iv_call_res), \
          "\nImplied vol (put): ", validate_price(iv_put, iv_put_res), \
          "\n----- Quanto prices -----" \
          "\nCall (1): ", validate_price(Quanto_call, Quanto_call_res), \
          "\nPut (1): ", validate_price(Quanto_put, Quanto_put_res), \
          "\nCall (2): ",validate_price(Quanto_call2, Quanto_call2_res), \
          "\nPut (2): ", validate_price(Quanto_put2, Quanto_put2_res))