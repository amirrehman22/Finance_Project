import pandas as pd
import requests
import streamlit as st
import math
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import numpy_financial as npf
warnings.filterwarnings('ignore')
from datetime import datetime


ticker_df = pd.read_excel("ticker.xlsx")

st.title('Finance Genius')
# Sidebar option
option = st.sidebar.radio("Select an Option", ('Stock Analysis','Financial Analysis','Budget Analysis','Amortization Calculator'))

## Amortization
if option == "Amortization Calculator":
    st.title("Loan Repayment Calculator")
    Principal_Amount = st.text_input("Loan amount requested by the client", placeholder="Enter loan amount")
    Interest_Rate = st.text_input("Annual interest rate (%)", placeholder="Enter interest rate")
    Loan_Duration = st.text_input("Loan term (in months)", placeholder="Enter loan term")
    
    if Principal_Amount and Interest_Rate and Loan_Duration:
        try:
            Principal_Amount = float(Principal_Amount)
            Interest_Rate = float(Interest_Rate)
            Loan_Duration = int(Loan_Duration)
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")


        if Principal_Amount and Interest_Rate and Loan_Duration :
            def calculate_monthly_installment(Principal_Amount, Interest_Rate, Loan_Duration):
                monthly_interest_rate = Interest_Rate / 12 / 100
                pmt = (Principal_Amount * monthly_interest_rate * (1 + monthly_interest_rate) ** Loan_Duration) / \
                    (((1 + monthly_interest_rate) ** Loan_Duration) - 1)
                return round(pmt, 2), monthly_interest_rate

            def calculate_monthly_schedule(Principal_Amount, monthly_interest_rate, pmt, Loan_Duration):
                months, payments, interest_payments, principal_payments, remaining_balances = [], [], [], [], []
                remaining_balance = Principal_Amount
                for i in range(1, Loan_Duration + 1):
                    monthly_interest_payment = round(remaining_balance * monthly_interest_rate, 2)
                    principal_payment = round(pmt - monthly_interest_payment, 2)
                    remaining_balance = round(remaining_balance - principal_payment, 2)
                    months.append(i)
                    payments.append(pmt)
                    interest_payments.append(monthly_interest_payment)
                    principal_payments.append(principal_payment)
                    remaining_balances.append(remaining_balance)
                return months, payments, interest_payments, principal_payments, remaining_balances

            pmt, monthly_interest_rate = calculate_monthly_installment(Principal_Amount, Interest_Rate, Loan_Duration)
            months, payments, interest_payments, principal_payments, remaining_balances = \
                calculate_monthly_schedule(Principal_Amount, monthly_interest_rate, pmt, Loan_Duration)

            st.subheader(f"Monthly Installment: {pmt:.2f}")
            df = pd.DataFrame({
                "Month": months,
                "Payment": payments,
                "Interest Payment": interest_payments,
                "Principal Payment": principal_payments,
                "Remaining Balance": remaining_balances
            })
            st.write(df)

            total_interest_payment = sum(interest_payments)
            st.write(f"Total interest payment: {total_interest_payment:.2f}")
        
        
        

def fetch_stock_data(ticker_symbol, start_date, end_date):
    """Fetch stock data from the Brecorder API."""
    url = f"https://markets.brecorder.com/ajax/daily_archive/?kseid={ticker_symbol}&term=1&term_table=company_rates&from={end_date}&to={start_date}"
    response = requests.post(url)
    if response.status_code == 200:
        json_data = response.json()
        data_list = json_data.get("data", [])
        df = pd.DataFrame(data_list)
        df.drop('sdate', axis=1, inplace=True)
        if not df.empty and 'fdate' in df.columns:
            df['fdate'] = pd.to_datetime(df['fdate'])
        return df
    else:
        st.error(f"Failed to retrieve data. Status code: {response.status_code}")
        return pd.DataFrame()
    
def statistic(df_stock):
    columns_to_convert = ['open', 'high', 'low', 'close', 'average', 'turnover', 'ldcp']
    df_stock[columns_to_convert] = df_stock[columns_to_convert].apply(pd.to_numeric, errors='coerce')
    statistic_matrix=df_stock.describe().T.drop('fdate')
    return statistic_matrix

def Return_and_Risk_Analysis(df_stock):
    df_1 = df_stock
    df_1["Close_t-1"] = df_1["close"].shift(-1)
    df_1["HPR"] = df_1["close"] / df_1["Close_t-1"]
    df_1["HPY"] = df_1["HPR"] - 1

    
    avg_return = df_1["HPY"].mean()
    Geo_return = (df_1["HPY"]+1).mean()
    Variance = df_1["HPY"].var(ddof=0)
    std_dev = math.sqrt(Variance)
    cv = std_dev / avg_return

    # Display the results with Streamlit
    st.subheader("Return and Risk Analysis")
    st.write(f"From {end_date} To {start_date}")
    
    st.write(f"**Average Return :** {avg_return:.6f}")
    st.write(f"**Geometric Return:** {Geo_return:.6f}")
    st.write(f"**Variance(Risk):** {Variance:.6f}")
    st.write(f"**Standard Deviation:** {std_dev:.6f}")
    st.write(f"**Coefficient of Variation (CV):** {cv:.6f}")
    
    # Return results for additional use if needed
    return {
        "Average Return": avg_return,
        "Geometric Return": Geo_return,
        "Variance": Variance,
        "Standard Deviation": std_dev,
        "Coefficient of Variation": cv
    }
    
def market_summary():
    # URL of the PSX market summary page
    url = "https://www.psx.com.pk/market-summary/"

    # Send a GET request to fetch the HTML content of the page
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return  # Use return instead of exit to avoid stopping the script

    # Parse the content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the div with the class specific to the "Automobile Assembler" section
    _div = soup.find('div', class_='col-sm-12 tab-pane inner-content-table automobile-div active')
    if not _div:
        print("Could not find the 'Automobile Assembler' section.")
        return

    # Find all divs with the class 'table-responsive' inside the parent '_div'
    table_divs = _div.find_all('div', class_='table-responsive')
    if not table_divs:
        print("No tables found in the 'Automobile Assembler' section.")
        return

    # Initialize a list to store the extracted data
    all_data = []

    # Loop through each table div and extract rows
    for table_div in table_divs:
        # Find the sector by looking for <h4> within the <thead> section
        sector = table_div.find('h4').text.strip() if table_div.find('h4') else 'Unknown Sector'
        
        # Find all rows in the current table div
        rows = table_div.find_all('tr')

        # Loop through each row and extract the data (skip the header row)
        for row in rows[1:]:  # Assuming first row is the header
            cols = row.find_all('td')
            if len(cols) > 0:
                symbol = cols[0].get('data-srip', np.nan)
                company_name = cols[0].text.strip() if cols[0].text.strip() else np.nan
                ldcp = cols[1].text.replace(",", "").strip() if len(cols) > 1 else np.nan
                open_price = cols[2].text.replace(",", "").strip() if len(cols) > 2 else np.nan
                high = cols[3].text.replace(",", "").strip() if len(cols) > 3 else np.nan
                low = cols[4].text.replace(",", "").strip() if len(cols) > 4 else np.nan
                current = cols[5].text.replace(",", "").strip() if len(cols) > 5 else np.nan
                change = cols[6].text.replace(",", "").strip() if len(cols) > 6 else np.nan
                volume = cols[7].text.replace(",", "").strip() if len(cols) > 7 else np.nan

                # Append the row's data to the list
                all_data.append([sector, symbol, company_name, ldcp, open_price, high, low, current, change, volume])

    # Convert the collected data to a DataFrame
    columns = ["SECTOR", "SYMBOL", "COMPANY NAME", "LDCP", "OPEN", "HIGH", "LOW", "CURRENT", "CHANGE", "VOLUME"]
    df = pd.DataFrame(all_data, columns=columns)

    # Convert numeric columns to appropriate types, handling empty strings
    numeric_columns = ["LDCP", "OPEN", "HIGH", "LOW", "CURRENT", "CHANGE", "VOLUME"]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to float, handle errors with NaN

    return df


def calculate_daily_return(data):
    def compute_return(row):
        if row['OPEN'] != 0 and pd.notna(row['OPEN']):
            return (row['CURRENT'] - row['OPEN']) / row['OPEN'] * 100
        return 0

    data['RETURN'] = data.apply(compute_return, axis=1)
    
    # Fill remaining NaN values with 0
    data.fillna(0, inplace=True)
    return data

def describe_sector():
   return df_summary.describe()[1:]

def maximum_value(data, column):
    # Check if the column exists in the DataFrame
    message = f'Stock with the maximum {column} value'
    st.write(message)
    
    # Find the row(s) where the specified column has the maximum value
    max_value_row = data[data[column] == data[column].max()]
    
    return max_value_row
def minimum_value(data, column):
    # Check if the column exists in the DataFrame
    message = f'Stock with the minimum {column} value'
    st.write(message)

    # Find the row(s) where the specified column has the maximum value
    min_value_row = data[data[column] == data[column].min()]
    
    return min_value_row

def plot_bargraph(data, column='RETURN'):
    """
    Return the top 10 stocks based on the given column.
    """
    data_sorted = data.sort_values(by=column, ascending=False)
    top_stocks = data_sorted[['SYMBOL', 'COMPANY NAME', 'LDCP','OPEN', 'HIGH','LOW','CHANGE', 'VOLUME', 'RETURN']].head(10)


    plt.figure(figsize=(12, 5))
    sns.set(style="whitegrid", palette="muted")
    barplot = sns.barplot(x='COMPANY NAME', y=column, data=top_stocks, hue='COMPANY NAME', palette='coolwarm', edgecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.title(f'Top 10 Stocks by {column}', fontsize=18, fontweight='bold', color='darkblue')
    plt.xlabel('Stock Symbol', fontsize=14)
    plt.ylabel(f'{column} ', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    for p in barplot.patches:
        barplot.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center', fontsize=12, color='black', fontweight='bold',
                         xytext=(0, 5), textcoords='offset points')
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(plt)
    



def top_company_each_group(data):
    
    sorted_data = data.sort_values(by='RETURN', ascending=False)
    first_row_each_group = sorted_data.groupby('SECTOR').first().reset_index()
    first_row_each_group = first_row_each_group.sort_values(by='RETURN', ascending=False)
    # Return the result containing top companies by sector
    return first_row_each_group[['SECTOR','COMPANY NAME','OPEN', 'HIGH', 'LOW', 'CURRENT', 'CHANGE', 'VOLUME', 'RETURN']]


def plot_sector_data(data, column_name='RETURN'):
    # Group by 'SECTOR' and calculate the mean of the specified return column
    group_sector = data.groupby('SECTOR')[column_name].mean()
    
    # Convert the series to a DataFrame and reset the index
    sector_data = group_sector.reset_index()
    
    # Sort the DataFrame by the return column in descending order
    sector_data_sorted = sector_data.sort_values(by=column_name, ascending=False)
    
    # Sort the data by the specified column in descending order
    sorted_data = sector_data_sorted[['SECTOR', column_name]].sort_values(by=column_name, ascending=False)
    
    # Create a figure for the plot
    plt.figure(figsize=(12, 7))
    
    # Create the bar plot using the specified column
    sns.barplot(data=sorted_data, x=column_name, y='SECTOR', palette='coolwarm')
    
    # Add titles and labels
    plt.title(f'Average {column_name} Value of Stock in Sector', fontsize=14, fontweight='bold')
    plt.xlabel(column_name, fontsize=12)
    plt.ylabel('Sector', fontsize=12)
    
    # Add the values on top of the bars
    for index, value in enumerate(sorted_data[column_name]):
        plt.text(value + 0.05, index, f'{value:.2f}', va='center', ha='left', fontsize=10, color='black')
    
    # Add gridlines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Adjust layout for better space
    plt.tight_layout()
    
    # Use Streamlit to render the plot
    st.pyplot(plt)


def plot_sector_counts(data):
   
    # Sort data by count in descending order
    sector_counts = data['SECTOR'].value_counts().reset_index()
    sector_counts.columns = ['Sector', 'Count']

    # Create the bar plot
    plt.figure(figsize=(12, 7))
    sns.barplot(data=sector_counts, x='Count', y='Sector', palette='coolwarm')

    # Add count labels on top of the bars
    for index, value in enumerate(sector_counts['Count']):
        plt.text(value + 2, index, str(value), va='center', ha='left', fontsize=10, color='black')

    # Add titles and labels
    plt.title('Company Counts by Sector', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Companies', fontsize=12)
    plt.ylabel('Sector', fontsize=12)

    # Add gridlines for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.6)

    # Adjust layout for better space
    plt.tight_layout()

    # Display the plot using Streamlit
    st.pyplot(plt)
    


    
    


    
if option == "Stock Analysis":
    
    option_stock = st.sidebar.radio("Select an Option", ('Individual Stock Analysis','Market Summary Analysis'))
    
    if option_stock == "Individual Stock Analysis":
    
        st.title("Stock Data Analysis")
        ticker_symbol = st.selectbox("Select Company Ticker (e.g., EFERT)", ticker_df )

        end_date = st.date_input("From Date")
        start_date = st.date_input("To Date")

        if st.button("Fetch Data"):
            if ticker_symbol and start_date and end_date:
                if start_date > end_date:
                    df_stock = fetch_stock_data(ticker_symbol, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                    if not df_stock.empty:
                        st.write(f"Price History of {ticker_symbol}") 

                        st.write(df_stock)
                        
                        st.header('Descriptive statistics')
                        st.write(f"From {end_date} To {start_date}")
                        
                        st.write(statistic(df_stock))
                        Return_and_Risk_Analysis(df_stock)
                        
                    else:
                        st.write("No data found for the given inputs.")
                else:
                    st.error("Start date must be before end date.")
                    
    if option_stock == "Market Summary Analysis":
        df_summary_0 = market_summary()
        
        st.header(f'Market Summary of {datetime.now().date()}')
        
        st.write(df_summary_0)
        
        df_summary= calculate_daily_return(df_summary_0)
        
        st.header('Descriptive statistics')
        
        st.write(describe_sector())
        
        columns = ["RETURN","LDCP", "OPEN", "HIGH", "LOW", "CURRENT", "CHANGE", "VOLUME"]
        
        st.header('Stock Insight')
        sel_columns = st.selectbox("Select Columns (e.g., RETURN):", columns, key="column_select_1" )
        sort_option = st.selectbox("Sort By", ["By Maximum Value", "By Minimum Value"])

        # Sort data based on user selection
        

        if sort_option == "By Maximum Value":
           st.write(maximum_value(df_summary, sel_columns))
        elif sort_option == "By Minimum Value":
            st.write(minimum_value(df_summary, sel_columns))


        st.header('Graph of the top 10 stocks')
        user_column = st.selectbox("Top 10 stocks (e.g., RETURN):", columns, key="column_select_2" )
        plot_bargraph(df_summary, user_column)
        
        st.header('SectorWise Analysis')
        plot_option = st.selectbox("Select Columns (e.g., RETURN):", columns, key="column_select_3" )

        plot_sector_data(df_summary, plot_option)
        
        st.header('Top Company in Sector by Return')
        st.write(top_company_each_group(df_summary))
        
        st.header('No of Company in Sector')
        st.write(plot_sector_counts(df_summary))

        
        
        

def process_cashflow(cashflow_input):
    try:
        # Split the string into a list of integers
        values = [int(value.strip()) for value in cashflow_input.split(",")]
        return values
    except ValueError:
        # Handle invalid inputs
        st.error("Please enter valid numeric values for all fields.")
        
        return None
    
    
def calculate_payback_period(Cash_Flow):
    initial_investment = abs(Cash_Flow[0])  # Initial investment as a positive value
    cumulative_cash = 0

    # Iterate over cash inflows (excluding the first value, which is the initial investment)
    for year, inflow in enumerate(Cash_Flow[1:], start=1):
        cumulative_cash += inflow
        if cumulative_cash >= initial_investment:
            excess = cumulative_cash - initial_investment
            return year - (excess / inflow)  # Interpolation for mid-year recovery
    return None  # If payback period is not reached    
    
def Net_Present_Value(rate,cashflow):
    npv_value = npf.npv(rate,cashflow)
    return npv_value

def IRR_value(Cash_Flow):
    irr_value = npf.irr(Cash_Flow)
    return irr_value*100

def Profitability_Index(discount_rate,Cash_Flow):
    npv_value = npf.npv(discount_rate,Cash_Flow[1:])
    inital_investment = abs(Cash_Flow[0])
    return npv_value/inital_investment


if option == 'Budget Analysis':

        st.title("Financial Analysis of Project")
        
        # Input field for cashflow
        cashflow_input = st.text_input("Enter your CashFlow of Project separated by commas ( , )", placeholder="Enter CashFlow amount") 
        rate = st.text_input("Enter Discount Rate %", placeholder="Enter discount rate") 
        if rate and cashflow_input:
            try:
                rate = float(rate) / 100
                cash_flow = process_cashflow(cashflow_input)
                if cash_flow is not None:
                    st.write("CashFlow")  # Safely use cash_flow within try block
                    st.write(cash_flow)
                
                    PBP=calculate_payback_period(cash_flow)
                    st.write(f"**Pay Back Period of Project :** {PBP:.2f} years")
        
        
                    npv_value = Net_Present_Value(rate,cash_flow)
                    st.write(f"**NPV of Project :** {npv_value:.2f}")
        
                    IRR=IRR_value(cash_flow)
                    st.write(f"**IRR of Project :** {IRR:.2f}%")
        
                    PI=Profitability_Index(rate,cash_flow)
                    st.write(f"**Profitability_Index of Project :** {PI:.2f}")
                else:
                    st.error("Please enter valid numeric values for all fields.")
                    
            except ValueError:
                st.error("Please enter valid numeric values for all fields.")



def get_financial_data(ticker):
    url = f"https://markets.brecorder.com/ajax/financial_data/{ticker}?ksecode={ticker}"

    # Sending the request
    response = requests.post(url)

    # Check if the request was successful
    if response.status_code == 200:
        data_raw = response.json()  # Parse the JSON response
        return data_raw
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")
        return None


def process_balance_sheet(data):
    balance_sheet =pd.DataFrame(data["data"]).T
    return balance_sheet
    

def process_income_statement(data):
    income_statement=pd.DataFrame(data["data"]).T
    return income_statement

def process_cash_flow(data):
    cash_flow=pd.DataFrame(data["data"]).T
    return cash_flow

def process_equity(data):
    equity=pd.DataFrame(data["data"]).T
    return equity

def process_ratio(data):
    ratio=pd.DataFrame(data["data"]).T
    return ratio




if option == 'Financial Analysis':  
    st.title('Financial Highlights')
    
    ticker = st.selectbox("Select Company Ticker (e.g., EFERT)", ticker_df )
    
    if st.button("Fetch Data"):
            
        data_raw=get_financial_data(ticker)
        
        # Example usage:
        balance_sheet = process_balance_sheet(data_raw['balance_sheet'])
        st.write('Balance Sheet')
        
        st.write(balance_sheet)
        
        
        income_statement = process_income_statement(data_raw['income_statement'])
        st.write('Income Statement')
        st.write(income_statement)
        
        
        cash_flows = process_cash_flow(data_raw['cash_flow'])
        st.write("Cash Flows")
        st.write(cash_flows)
        
        equity = process_equity(data_raw['equity'])
        st.write('Equity')
        st.write(equity)
        
        ratio = process_ratio(data_raw['ratio'])
        st.write('Ratio')
        st.write(ratio)
        

      
        
        
        
        

        
        
        
        
        
        
        
        
   
    
        
