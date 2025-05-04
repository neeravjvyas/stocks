import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
from dateutil import relativedelta
import numpy_financial as npf
import numpy as np

# Reading input Excel file containing ticker information and investment parameters
def read_input_excel(file_path):
    """
    Reads an Excel file with ticker data and validates required columns.
    Returns a DataFrame with the input data.
    """
    try:
        df = pd.read_excel(file_path)
        required_columns = ['Ticker']
        df.columns = df.columns.str.strip()
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        return df
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

# Fetching historical price and dividend data for a given ticker
def fetch_historical_data(ticker, start_date=None, end_date=None, frequency='1mo', duration=None):
    """
    Fetches historical price and dividend data for a ticker using yfinance.
    Returns price data, dividend data, fund name, and exchange as DataFrames and strings.
    """
    try:
        stock = yf.Ticker(ticker)
        # Get fund info (name and exchange)
        info = stock.info
        fund_name = info.get('longName', 'Unknown Fund')
        exchange = info.get('exchange', 'Unknown Exchange')

        if duration == 'MAX':
            data = stock.history(period='max', interval=frequency)
            dividends = stock.dividends
        else:
            data = stock.history(start=start_date, end=end_date, interval=frequency)
            start_date = start_date.replace(tzinfo=None) if start_date else None
            end_date = end_date.replace(tzinfo=None) if end_date else None
            dividends = stock.dividends
            if not dividends.empty:
                dividends.index = dividends.index.tz_localize(None)
                dividends = dividends[(dividends.index >= start_date) & (dividends.index <= end_date)]
        
        if data.empty:
            print(f"No data found for {ticker}")
            return None, None, fund_name, exchange
        
        data.reset_index(inplace=True)
        data['Date'] = data['Date'].dt.tz_localize(None)
        data['Ticker'] = ticker
        
        # Process dividends
        if not dividends.empty:
            dividends = dividends.reset_index()
            dividends['Date'] = pd.to_datetime(dividends['Date']).dt.tz_localize(None)
            dividends = dividends.rename(columns={'Dividends': 'Dividend'})
            dividends['Ticker'] = ticker
        else:
            dividends = pd.DataFrame(columns=['Ticker', 'Date', 'Dividend'])
            dividends['Date'] = pd.to_datetime(dividends['Date'], errors='coerce')
        
        return data[['Ticker', 'Date', 'Close']], dividends, fund_name, exchange
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None, None, 'Unknown Fund', 'Unknown Exchange'

# Parsing start date from string input
def parse_start_date(start_str, max_start=False):
    """
    Converts a string in 'MMM-YYYY' or 'Month-YYYY' format to a datetime object.
    Returns None if max_start is True or input is invalid.
    """
    if pd.isna(start_str) or max_start:
        return None
    try:
        return datetime.strptime(str(start_str), '%b-%Y') if len(str(start_str).split('-')[0]) <= 3 else datetime.strptime(str(start_str), '%B-%Y')
    except Exception as e:
        print(f"Error parsing start date {start_str}: {e}")
        return None

# Calculating XIRR for cash flows
def calculate_xirr(cash_flows, dates):
    """
    Calculates the Extended Internal Rate of Return (XIRR) for a series of cash flows.
    Returns annualized XIRR value or None if calculation fails.
    """
    try:
        base_date = min(dates)
        days = [(d - base_date).days for d in dates]
        if len(cash_flows) != len(days) or len(cash_flows) < 2:
            return None
        
        def xirr_func(rate):
            return sum(cf / (1 + rate) ** (day / 365.0) for cf, day in zip(cash_flows, days))
        
        xirr_value = npf.irr(cash_flows)
        if np.isnan(xirr_value):
            return None
        return (1 + xirr_value) ** 365 - 1
    except Exception:
        return None

# Calculating CAGR for investment period
def calculate_cagr(data, years):
    """
    Calculates the Compound Annual Growth Rate (CAGR) based on price data.
    Returns CAGR value or None if data is invalid.
    """
    if data is None or data.empty or len(data) < 2:
        return None
    try:
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        if start_price <= 0 or end_price <= 0:
            return None
        cagr = (end_price / start_price) ** (1 / years) - 1
        return cagr
    except Exception:
        return None

# Calculating SIP with dividend reinvestment
def calculate_sip(data, dividends, investment_amount, frequency, start_date=None):
    """
    Simulates a Systematic Investment Plan (SIP) with dividend reinvestment.
    Returns total invested, current value, units held, XIRR, and total dividends.
    """
    if data is None or data.empty:
        return 0, 0, 0, None, 0
    
    if start_date:
        data = data[data['Date'] >= start_date]
        dividends = dividends[dividends['Date'] >= start_date] if not dividends.empty else pd.DataFrame()
    
    if data.empty:
        return 0, 0, 0, None, 0
    
    units = 0
    total_invested = 0
    cash_flows = []
    cash_flow_dates = []
    total_dividends = 0
    
    freq_days = {'1mo': 30, '1wk': 7, '1d': 1}
    freq_interval = freq_days.get(frequency, 30)
    
    for _, row in data.iterrows():
        units += investment_amount / row['Close']
        total_invested += investment_amount
        cash_flows.append(-investment_amount)
        cash_flow_dates.append(row['Date'])
    
    if not dividends.empty:
        for _, div_row in dividends.iterrows():
            div_date = div_row['Date']
            units_held = units
            dividend_per_share = div_row['Dividend']
            dividend_payment = dividend_per_share * units_held
            if dividend_payment > 0:
                total_dividends += dividend_payment
                closest_data = data[data['Date'] >= div_date].head(1)
                if not closest_data.empty:
                    reinvest_price = closest_data['Close'].iloc[0]
                    additional_units = dividend_payment / reinvest_price
                    units += additional_units
    
    latest_price = data['Close'].iloc[-1]
    current_value = units * latest_price
    
    cash_flows.append(current_value)
    cash_flow_dates.append(data['Date'].iloc[-1])
    
    xirr = calculate_xirr(cash_flows, cash_flow_dates)
    
    return total_invested, current_value, units, xirr, total_dividends

# Calculating annualized returns for multiple periods
def calculate_annualized_returns(ticker, end_date):
    """
    Calculates annualized returns (CAGR) for different time periods.
    Returns a dictionary with returns for Lifetime, 1Y, 5Y, 10Y, 15Y, 20Y.
    """
    periods = {'Lifetime': None, '1Y': 1, '5Y': 5, '10Y': 10, '15Y': 15, '20Y': 20}
    results = {'Ticker': ticker}
    
    for period, years in periods.items():
        if period == 'Lifetime':
            data, _, _, _ = fetch_historical_data(ticker, duration='MAX', frequency='1mo')
        else:
            start_date = end_date - relativedelta.relativedelta(years=years)
            data, _, _, _ = fetch_historical_data(ticker, start_date=start_date, end_date=end_date, frequency='1mo')
        
        cagr = calculate_cagr(data, years) if years else calculate_cagr(data, (data['Date'].iloc[-1] - data['Date'].iloc[0]).days / 365.0) if data is not None else None
        results[period] = round(cagr, 4) if cagr is not None else None
    
    return results

# Main program to orchestrate the SIP calculation process
def main():
    """
    Main function to run the SIP calculator.
    Checks for default input files (ETF_list.xlsx or Stock_List.xlsx) in the input folder.
    Saves results to the output folder with date-stamped filenames.
    """
    print("=== SIP Calculator with XIRR, Dividend Reinvestment, and Annualized Returns ===")
    print("Expected Excel Input Columns:")
    print("- Ticker: e.g., VAS.AX, SPY, TQQQ")
    print("- Start Month/Year: e.g., Jan-2015 (optional, default: earliest)")
    print("- Duration: 1Y, 2Y, 5Y, MAX (optional, default: MAX)")
    print("- Frequency: monthly, weekly, daily (optional, default: monthly)")
    print("- Investment Amount: e.g., 100 (optional, default: 100 USD)")
    print("Output includes historical data, SIP results with XIRR and reinvested dividends, and annualized returns.")
    print("==================================================================")

    # Define input and output directories relative to the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check for default input files
    default_files = ['ETF_list.xlsx', 'Stock_List.xlsx']
    input_file = None
    file_type = None

    for fname in default_files:
        file_path = os.path.join(input_dir, fname)
        if os.path.exists(file_path):
            input_file = file_path
            file_type = 'ETF' if fname == 'ETF_list.xlsx' else 'Stock'
            break

    if not input_file:
        # If no default file is found, prompt user for input file
        input_file = input("No default input file (ETF_list.xlsx or Stock_List.xlsx) found in input folder. Enter the path to your input Excel file: ")
        file_type = input("Is this an ETF or Stock input file? (Enter 'ETF' or 'Stock'): ").strip().capitalize()
        if file_type not in ['ETF', 'Stock']:
            print("Invalid file type. Defaulting to ETF.")
            file_type = 'ETF'

    # Generate output filename with current date
    current_date = datetime.today().strftime('%Y-%m-%d')
    output_filename = f"{file_type}_SIP_output_{current_date}.xlsx"
    output_file = os.path.join(output_dir, output_filename)

    # Read input Excel
    try:
        input_df = read_input_excel(input_file)
        print(f"Found {len(input_df)} tickers in {os.path.basename(input_file)}.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Process each ticker
    all_data = []
    sip_results = []
    annualized_returns = []
    end_date = datetime.today().replace(tzinfo=None)
    
    for index, row in input_df.iterrows():
        ticker = str(row['Ticker']).strip()
        start_str = row.get('Start Month/Year', None)
        duration = row.get('Duration', 'MAX').strip() if pd.notna(row.get('Duration')) else 'MAX'
        freq_input = row.get('Frequency', 'monthly').strip().lower() if pd.notna(row.get('Frequency')) else 'monthly'
        try:
            investment_amount = float(row.get('Investment Amount', 100)) if pd.notna(row.get('Investment Amount')) else 100
        except (ValueError, TypeError) as e:
            print(f"Invalid Investment Amount for {ticker}: {row.get('Investment Amount')}. Using default 100.")
            investment_amount = 100

        freq_map = {'monthly': '1mo', 'weekly': '1wk', 'daily': '1d'}
        frequency = freq_map.get(freq_input, '1mo')
        if frequency not in freq_map.values():
            print(f"Invalid frequency for {ticker}: {freq_input}. Using monthly.")
            frequency = '1mo'

        print(f"Processing {ticker}...")

        start_date = parse_start_date(start_str, duration.upper() == 'MAX')
        
        if duration.upper() != 'MAX' and start_date:
            duration_map = {'1Y': 1, '2Y': 2, '5Y': 5}
            if duration.upper() in duration_map:
                start_date = end_date - relativedelta.relativedelta(years=duration_map[duration.upper()])
            else:
                print(f"Invalid duration for {ticker}: {duration}. Using MAX.")
                duration = 'MAX'

        data, dividends, fund_name, exchange = fetch_historical_data(ticker, start_date, end_date, frequency, duration)
        if data is not None:
            all_data.append(data)
            
            total_invested, current_value, total_units, xirr, total_dividends = calculate_sip(data, dividends, investment_amount, frequency, start_date)
            
            # Determine actual start date (earliest date if not specified)
            actual_start_date = data['Date'].iloc[0] if data is not None and not data.empty else None
            
            sip_results.append({
                'Ticker': ticker,
                'Fund Name': fund_name,
                'Exchange': exchange,
                'SIP Start Date': actual_start_date,
                'Start Month/Year': start_str if pd.notna(start_str) else 'Earliest',
                'Duration': duration,
                'Frequency': freq_input.capitalize(),
                'Investment per Period (USD)': investment_amount,
                'Total Invested (USD)': round(total_invested, 2),
                'Total Dividends Reinvested (USD)': round(total_dividends, 2),
                'Current Value (USD)': round(current_value, 2),
                'Total Units': round(total_units, 4),
                'Gain/Loss (USD)': round(current_value - total_invested, 2),
                'XIRR (%)': round(xirr * 100, 2) if xirr is not None else None
            })

        returns = calculate_annualized_returns(ticker, end_date)
        annualized_returns.append(returns)

    # Save output to Excel
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)
        sip_result_df = pd.DataFrame(sip_results)
        returns_df = pd.DataFrame(annualized_returns)
        
        try:
            with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
                combined_data.to_excel(writer, sheet_name='Historical_Data', index=False)
                sip_result_df.to_excel(writer, sheet_name='SIP_Results', index=False)
                returns_df.to_excel(writer, sheet_name='Annualized_Returns', index=False)
        except ImportError:
            print("xlsxwriter not found, using openpyxl instead.")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                combined_data.to_excel(writer, sheet_name='Historical_Data', index=False)
                sip_result_df.to_excel(writer, sheet_name='SIP_Results', index=False)
                returns_df.to_excel(writer, sheet_name='Annualized_Returns', index=False)
        
        print(f"Data and results saved to {output_file}")
        print(f"Historical data rows: {len(combined_data)}")
        print(f"Tickers processed: {len(sip_results)}")
    else:
        print("No data fetched successfully.")

if __name__ == "__main__":
    main()