import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os
import sys

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb['cells'] = []

    def add_md(text):
        nb['cells'].append(nbf.v4.new_markdown_cell(text.strip()))

    def add_code(code):
        nb['cells'].append(nbf.v4.new_code_cell(code.strip()))

    # Title and Table of Contents
    add_md("""
# Pandas Tutorial - Full Examples Collection
This notebook contains the complete set of Pandas code examples from Chapter 17 (*Pandas*) of **Python Programming for Economics and Finance** by Thomas J. Sargent & John Stachurski (QuantEcon).

---
## Table of Contents
- **0. Common Imports & Setup**: Core scientific libraries (`pandas`, `numpy`, `matplotlib.pyplot`, `requests`, `wbgapi`, `yfinance`)
- **17.2 Series**:
  - Example 1: Creating a Series of random observations
  - Example 2: Arithmetic operations on Series (`s * 100`)
  - Example 3: Applying NumPy universal functions (`np.abs(s)`)
  - Example 4: Summary statistics (`s.describe()`)
  - Example 5: Custom string index assignment (`s.index = ...`)
  - Example 6: Dictionary-style value lookup and in-place assignment (`s['AMZN']`)
  - Example 7: Membership test with `in` operator
- **17.3 DataFrames**:
  - Example 8: Loading CSV data (`read_csv`) and inspecting type
  - Example 9: Displaying the entire DataFrame
  - **17.3.1 Select Data by Position**:
    - Example 10: Row slicing by integer position (`df[2:5]`)
    - Example 11: Column selection with list of strings (`df[['country', 'tcgdp']]`)
    - Example 12: Integer-based row & column selection (`df.iloc[2:5, 0:4]`)
    - Example 13: Mixed label and integer indexing (`df.loc[df.index[2:5], ...]`)
  - **17.3.2 Select Data by Conditions**:
    - Example 14: Filtering rows with boolean condition (`df[df.POP >= 20000]`)
    - Example 15: Evaluating boolean Series (`df.POP >= 20000`)
    - Example 16: Compound filtering with `.isin()` and bitwise AND (`&`)
    - Example 17: Querying rows with `.query()`
    - Example 18: Compound querying with string expressions in `.query()`
    - Example 19: Column arithmetic inside boolean filter
    - Example 20: Column arithmetic inside `.query()`
    - Example 21: Locating row with maximum value (`max(df.cc)`)
    - Example 22: Conditional row filtering with specific column projection
    - Example 23: Application - Subsetting DataFrame (`df[['country', 'POP', 'tcgdp']]`)
    - Example 24: Saving subset DataFrame to CSV file (`to_csv`)
  - **17.3.3 Apply Method**:
    - Example 25: Applying built-in `max` function across columns
    - Example 26: Applying lambda row identity along `axis=1`
    - Example 27: Constructing complex conditions with `.apply()`
    - Example 28: Filtering with `.loc` using complex condition tuple
  - **17.3.4 Make Changes in DataFrames**:
    - Example 29: Conditional masking with `df.where()`
    - Example 30: Modifying specific elements using `.loc`
    - Example 31: Modifying rows with custom function and `.apply(axis=1)`
    - Example 32: Element-wise formatting and transformation with `df.map()`
    - Example 33: Application - Inserting NaN values for imputation testing
    - Example 34: Replacing NaNs with 0 using custom function and `df.map()`
    - Example 35: Imputing missing values with column means using `.fillna()`
  - **17.3.5 Standardization and Visualization**:
    - Example 36: Extracting target columns (`country`, `POP`, `tcgdp`)
    - Example 37: Setting column as index (`set_index('country')`)
    - Example 38: Renaming DataFrame columns
    - Example 39: Rescaling population variable to single units
    - Example 40: Calculating Real GDP per capita
    - Example 41: Generating a bar plot of GDP per capita
    - Example 42: Sorting DataFrame by GDP per capita (`sort_values`)
    - Example 43: Re-generating sorted bar plot of GDP per capita
- **17.4 On-Line Data Sources**:
  - **17.4.1 Accessing Data with requests**:
    - Example 44: Testing HTTP GET connection to FRED API
    - Example 45: Reading and inspecting raw CSV lines from response
    - Example 46: Parsing FRED CSV into Pandas DataFrame with `parse_dates=True`
    - Example 47: Inspecting first rows with `data.head()`
    - Example 48: Summary statistics of unemployment rate (`data.describe()`)
    - Example 49: Time-series date slicing and plotting (`data['2006':'2012'].plot()`)
  - **17.4.2 Using wbgapi and yfinance to Access Data**:
    - Example 50: Inspecting World Bank indicator metadata with `wbgapi`
    - Example 51: Fetching World Bank government debt data, transposing, and plotting
- **17.5 Exercises**:
  - **Exercise 17.5.1: Percentage Price Change over 2021 for Stocks**:
    - Example 52: Defining `read_data` function and fetching stock data with `yfinance`
    - Example 53: Solution 1 - Calculating percentage change using first and last prices
    - Example 54: Solution 2 - Alternative calculation using `.pct_change()`
    - Example 55: Plotting stock percentage changes as a bar chart
  - **Exercise 17.5.2: Year-on-Year Percentage Change for Indices (1971–2021)**:
    - Example 56: Fetching historical index data using `read_data` (1971 to 2021)
    - Example 57: Calculating yearly returns using `groupby(indices_data.index.year)`
    - Example 58: Summary statistics of yearly returns using `.describe()`
    - Example 59: Plotting 2x2 grid of time series subplots for indices
""")

    # 0. Common Imports
    add_md("""
## 0. Common Imports & Setup
Import the core libraries used throughout Chapter 17: `pandas`, `numpy`, `matplotlib.pyplot`, and `requests`.
""")
    add_code("""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Configure matplotlib inline display and formatting
%matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)
""")

    # 17.2 Series
    add_md("""
---
## 17.2 Series
A `Series` is a one-dimensional array-like object containing an array of data and an associated array of data labels, called its *index*.
""")

    add_md("""
### Example 1: Creating a Series
We begin by creating a series of four random observations with a name attribute.
""")
    add_code("""
np.random.seed(42)  # For reproducible random observations
s = pd.Series(np.random.randn(4), name='daily returns')
s
""")

    add_md("""
### Example 2: Arithmetic Operations on Series
Pandas Series support element-wise scalar arithmetic, similar to NumPy arrays.
""")
    add_code("""
s * 100
""")

    add_md("""
### Example 3: Applying NumPy Functions to Series
NumPy universal functions (ufuncs) such as `np.abs()` can be applied directly to a Pandas Series.
""")
    add_code("""
np.abs(s)
""")

    add_md("""
### Example 4: Descriptive Statistics with `s.describe()`
Series provide convenient statistical summary methods such as `.describe()`.
""")
    add_code("""
s.describe()
""")

    add_md("""
### Example 5: Custom Index Assignment
Unlike NumPy arrays, Pandas Series allow custom label-based indices, such as stock ticker symbols.
""")
    add_code("""
s.index = ['AMZN', 'AAPL', 'MSFT', 'GOOG']
s
""")

    add_md("""
### Example 6: Dictionary-Style Access and In-Place Assignment
Series support dictionary-like key lookup, item access, and value modification.
""")
    add_code("""
print("Value for AMZN:", s['AMZN'])
s['AMZN'] = 0
s
""")

    add_md("""
### Example 7: Membership Test with `in` Operator
We can check if an index label exists in a Series using the standard Python `in` keyword.
""")
    add_code("""
'AAPL' in s
""")

    # 17.3 DataFrames
    add_md("""
---
## 17.3 DataFrames
A `DataFrame` is a two-dimensional tabular data structure with labeled axes (rows and columns).
We load data from the Penn World Tables (`test_pwt.csv`).
""")

    add_md("""
### Example 8: Loading CSV Data into a DataFrame
We read the Penn World Tables data from a URL using `pd.read_csv()` (with fallback to local file if offline).
""")
    add_code("""
# Primary source URL from QuantEcon repository with local fallback
url = 'https://raw.githubusercontent.com/QuantEcon/lecture-source-py/master/source/_static/lecture_specific/pandas/data/test_pwt.csv'
try:
    df = pd.read_csv(url)
except Exception:
    # Fallback to local copy if network is unavailable
    df = pd.read_csv('test_pwt.csv')

type(df)
""")

    add_md("""
### Example 9: Displaying the DataFrame
Inspect the full contents of `df`.
""")
    add_code("""
df
""")

    # 17.3.1 Select Data by Position
    add_md("""
---
### 17.3.1 Select Data by Position
Methods for slicing and indexing rows and columns by integer positions.
""")

    add_md("""
### Example 10: Row Slicing by Integer Position
Select particular rows using standard Python array slicing notation (`df[2:5]`).
""")
    add_code("""
df[2:5]
""")

    add_md("""
### Example 11: Column Selection by Column Name List
Select columns by passing a list of column names represented as strings.
""")
    add_code("""
df[['country', 'tcgdp']]
""")

    add_md("""
### Example 12: Integer-Based Row and Column Selection with `.iloc`
Select both rows and columns using integer indices with the format `.iloc[rows, columns]`.
""")
    add_code("""
df.iloc[2:5, 0:4]
""")

    add_md("""
### Example 13: Mixed Row and Column Indexing with `.loc`
Select rows and columns using a mixture of integer row indices and column label names.
""")
    add_code("""
df.loc[df.index[2:5], ['country', 'tcgdp']]
""")

    # 17.3.2 Select Data by Conditions
    add_md("""
---
### 17.3.2 Select Data by Conditions
Filtering and querying DataFrames using boolean masks, compound conditions, and `.query()`.
""")

    add_md("""
### Example 14: Filtering Rows with Boolean Condition
Filter rows where population is greater than or equal to 20,000 using the `[]` operator.
""")
    add_code("""
df[df.POP >= 20000]
""")

    add_md("""
### Example 15: Evaluating the Boolean Condition Series
Inspect the underlying Series of boolean values generated by `df.POP >= 20000`.
""")
    add_code("""
df.POP >= 20000
""")

    add_md("""
### Example 16: Compound Filtering with `.isin()` and Bitwise AND (`&`)
Filter rows matching specific countries and having a population greater than 40,000.
""")
    add_code("""
df[(df.country.isin(['Argentina', 'India', 'South Africa'])) & (df.POP > 40000)]
""")

    add_md("""
### Example 17: Querying Rows with `df.query()`
The `.query()` method provides a clean, concise string syntax for filtering.
""")
    add_code("""
# Equivalent to df[df.POP >= 20000]
df.query("POP >= 20000")
""")

    add_md("""
### Example 18: Compound Query with `df.query()`
Combining multiple conditions inside `.query()`.
""")
    add_code("""
df.query("country in ['Argentina', 'India', 'South Africa'] and POP > 40000")
""")

    add_md("""
### Example 19: Column Arithmetic within Boolean Filtering
Perform arithmetic between different columns (`cc + cg >= 80`) combined with population condition.
""")
    add_code("""
df[(df.cc + df.cg >= 80) & (df.POP <= 20000)]
""")

    add_md("""
### Example 20: Column Arithmetic with `df.query()`
The same arithmetic condition expressed via `.query()`.
""")
    add_code("""
# Equivalent to the boolean filter above
df.query("cc + cg >= 80 & POP <= 20000")
""")

    add_md("""
### Example 21: Locating Row with Maximum Value
Select the country with the largest household consumption share of GDP (`cc`).
""")
    add_code("""
df.loc[df.cc == max(df.cc)]
""")

    add_md("""
### Example 22: Conditional Filtering with Specific Column Selection
Combine conditional row filtering with specific column selection via `.loc[condition, columns]`.
""")
    add_code("""
df.loc[(df.cc + df.cg >= 80) & (df.POP <= 20000), ['country', 'year', 'POP']]
""")

    add_md("""
### Example 23: Application - Subsetting DataFrame
Extract a subset of columns (`country`, `POP`, `tcgdp`) to reduce memory and focus analysis.
""")
    add_code("""
df_subset = df[['country', 'POP', 'tcgdp']]
df_subset
""")

    add_md("""
### Example 24: Saving Subset DataFrame to CSV
Save the subset DataFrame to a CSV file for further analysis.
""")
    add_code("""
df_subset.to_csv('pwt_subset.csv', index=False)
print("Saved pwt_subset.csv successfully.")
""")

    # 17.3.3 Apply Method
    add_md("""
---
### 17.3.3 Apply Method
Using `df.apply()` to apply functions across rows (`axis=1`) or columns (`axis=0`).
""")

    add_md("""
### Example 25: Applying Built-in `max` Function across Columns
Apply the built-in `max` function to each numeric column (`axis=0`, default).
""")
    add_code("""
df[['year', 'POP', 'XRAT', 'tcgdp', 'cc', 'cg']].apply(max)
""")

    add_md("""
### Example 26: Applying Lambda Row Identity along `axis=1`
Apply a lambda function to each row (`axis=1`).
""")
    add_code("""
df.apply(lambda row: row, axis=1)
""")

    add_md("""
### Example 27: Constructing a Complex Condition with `df.apply()`
Define a complex conditional expression across rows using `df.apply(..., axis=1)` paired with selected column labels.
""")
    add_code("""
complexCondition = df.apply(
    lambda row: row.POP > 40000 if row.country in ['Argentina', 'India', 'South Africa'] else row.POP < 20000,
    axis=1), ['country', 'year', 'POP', 'XRAT', 'tcgdp']
complexCondition
""")

    add_md("""
### Example 28: Applying the Complex Condition to `df.loc`
Pass the `complexCondition` tuple into `df.loc` to filter rows and select columns simultaneously.
""")
    add_code("""
df.loc[complexCondition]
""")

    # 17.3.4 Make Changes in DataFrames
    add_md("""
---
### 17.3.4 Make Changes in DataFrames
Modifying values, handling missing data, masking with `.where()`, and element-wise mapping with `.map()`.
""")

    add_md("""
### Example 29: Conditional Masking with `df.where()`
Keep rows meeting a condition (`df.POP >= 20000`) and replace non-matching rows with `NaN`.
""")
    add_code("""
df.where(df.POP >= 20000)
""")

    add_md("""
### Example 30: Modifying Values In-Place with `.loc`
Assign `NaN` to the `cg` column for the observation with the maximum `cg` value.
""")
    add_code("""
df.loc[df.cg == max(df.cg), 'cg'] = np.nan
df
""")

    add_md("""
### Example 31: Modifying Rows with a Custom Function and `.apply(axis=1)`
Define a custom row updater function to adjust `POP` and `XRAT` values.
""")
    add_code("""
def update_row(row):
    # modify POP
    row.POP = np.nan if row.POP <= 10000 else row.POP
    # modify XRAT
    row.XRAT = row.XRAT / 10
    return row

df.apply(update_row, axis=1)
""")

    add_md("""
### Example 32: Element-Wise Formatting with `df.map()`
Round all numeric entries in the DataFrame to 2 decimal places while leaving strings untouched.
""")
    add_code("""
# Round all decimal numbers to 2 decimal places
df.map(lambda x: round(x, 2) if type(x) != str else x)
""")

    add_md("""
### Example 33: Application - Inserting NaN Values for Imputation Testing
Introduce `NaN` values at specific row and column indices using `zip()` and `.iloc`.
""")
    add_code("""
for idx in list(zip([0, 3, 5, 6], [3, 4, 6, 2])):
    df.iloc[idx] = np.nan
df
""")

    add_md("""
### Example 34: Replacing NaNs with 0 using `df.map()`
Define a function to replace any `NaN` values with 0 across the entire DataFrame.
""")
    add_code("""
# Replace all NaN values with 0
def replace_nan(x):
    if type(x) != str:
        return 0 if np.isnan(x) else x
    else:
        return x

df.map(replace_nan)
""")

    add_md("""
### Example 35: Imputing Missing Values with Column Means using `df.fillna()`
Perform single mean imputation on numeric columns using `.fillna()`.
""")
    add_code("""
df = df.fillna(df.iloc[:, 2:8].mean())
df
""")

    # 17.3.5 Standardization and Visualization
    add_md("""
---
### 17.3.5 Standardization and Visualization
Data transformation, index manipulation, column operations, and plotting bar charts.
""")

    add_md("""
### Example 36: Extracting Target Columns
Subset the DataFrame down to `country`, `POP`, and `tcgdp`.
""")
    add_code("""
df = df[['country', 'POP', 'tcgdp']]
df
""")

    add_md("""
### Example 37: Setting the Index to `country`
Set the `country` column as the DataFrame index.
""")
    add_code("""
df = df.set_index('country')
df
""")

    add_md("""
### Example 38: Renaming Columns
Rename the columns to more readable descriptive labels: `'population'` and `'total GDP'`.
""")
    add_code("""
df.columns = 'population', 'total GDP'
df
""")

    add_md("""
### Example 39: Rescaling Population Units
Convert the population variable from thousands into single unit counts (`* 1e3`).
""")
    add_code("""
df['population'] = df['population'] * 1e3
df
""")

    add_md("""
### Example 40: Calculating Real GDP Per Capita
Calculate Real GDP per capita in single dollars (`total GDP * 1e6 / population`).
""")
    add_code("""
df['GDP percap'] = df['total GDP'] * 1e6 / df['population']
df
""")

    add_md("""
### Example 41: Generating a Bar Plot of GDP Per Capita
Create a bar plot of GDP per capita using Pandas' built-in Matplotlib wrapper.
""")
    add_code("""
ax = df['GDP percap'].plot(kind='bar')
ax.set_xlabel('country', fontsize=12)
ax.set_ylabel('GDP per capita', fontsize=12)
plt.title('GDP per Capita by Country')
plt.show()
""")

    add_md("""
### Example 42: Sorting DataFrame by GDP Per Capita
Sort the DataFrame in descending order by `GDP percap`.
""")
    add_code("""
df = df.sort_values(by='GDP percap', ascending=False)
df
""")

    add_md("""
### Example 43: Re-Plotting Sorted GDP Per Capita
Re-generate the bar plot after sorting countries from highest to lowest GDP per capita.
""")
    add_code("""
ax = df['GDP percap'].plot(kind='bar')
ax.set_xlabel('country', fontsize=12)
ax.set_ylabel('GDP per capita', fontsize=12)
plt.title('GDP per Capita by Country (Sorted)')
plt.show()
""")

    # 17.4 On-Line Data Sources
    add_md("""
---
## 17.4 On-Line Data Sources
Querying online databases programmatically: Federal Reserve Economic Data (FRED) and World Bank (`wbgapi`).
""")

    add_md("""
### 17.4.1 Accessing Data with requests
Accessing FRED data using the `requests` library and parsing into Pandas.
""")

    add_md("""
### Example 44: Testing Connection to FRED API with `requests.get`
Fetch the unemployment rate CSV directly from FRED and verify the request status.
""")
    add_code("""
fred_url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE'
r = requests.get(fred_url)
print("HTTP Status Code:", r.status_code)
""")

    add_md("""
### Example 45: Reading and Inspecting Raw Lines from Response
Inspect the first few raw header and data lines from the FRED CSV response.
""")
    add_code("""
url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE'
source = requests.get(url).content.decode().split("\\n")
print("Line 0:", source[0])
print("Line 1:", source[1])
print("Line 2:", source[2])
""")

    add_md("""
### Example 46: Parsing FRED CSV into Pandas DataFrame
Use `pd.read_csv()` with `index_col=0` and `parse_dates=True` to parse dates automatically.
""")
    add_code("""
data = pd.read_csv(url, index_col=0, parse_dates=True)
print("Type of data:", type(data))
""")

    add_md("""
### Example 47: Inspecting First Rows with `data.head()`
Inspect the top observations of the unemployment dataset.
""")
    add_code("""
data.head()  # A useful method to get a quick look at a data frame
""")

    add_md("""
### Example 48: Summary Statistics with `data.describe()`
Configure precision and compute summary statistics for the unemployment rate.
""")
    add_code("""
pd.set_option('display.precision', 1)
data.describe()  # Summary statistics
""")

    add_md("""
### Example 49: Date-Range Slicing and Time Series Plotting
Slice the DataFrame by date range (`'2006':'2012'`) and plot the unemployment rate.
""")
    add_code("""
ax = data['2006':'2012'].plot(title='US Unemployment Rate', legend=False)
ax.set_xlabel('year', fontsize=12)
ax.set_ylabel('%', fontsize=12)
plt.show()
""")

    # 17.4.2 Using wbgapi and yfinance to Access Data
    add_md("""
---
### 17.4.2 Using wbgapi to Access World Bank Data
The `wbgapi` library queries the World Bank's extensive indicators databases.
""")

    add_md("""
### Example 50: Inspecting World Bank Indicator Metadata with `wbgapi`
Query metadata for indicator `GC.DOD.TOTL.GD.ZS` (Central government debt as % of GDP).
""")
    add_code("""
import wbgapi as wb

wb.series.info('GC.DOD.TOTL.GD.ZS')
""")

    add_md("""
### Example 51: Fetching World Bank Government Debt Data and Plotting
Retrieve government debt data for the USA and Australia (2005-2015), transpose, and plot.
""")
    add_code("""
govt_debt = wb.data.DataFrame('GC.DOD.TOTL.GD.ZS', economy=['USA', 'AUS'], time=range(2005, 2016))
govt_debt = govt_debt.T  # Move years from columns to rows for plotting
ax = govt_debt.plot(xlabel='year', ylabel='Government debt (% of GDP)')
plt.title('Central Government Debt (% of GDP) - USA vs Australia')
plt.show()
""")

    # 17.5 Exercises
    add_md("""
---
## 17.5 Exercises
Practical exercises covering Yahoo Finance data retrieval, percentage price change calculations, and multi-asset visualization.
""")

    add_md("""
### Exercise 17.5.1: Percentage Price Change over 2021 for Stocks
Calculate and visualize the percentage price change over 2021 for 11 major global stocks using `yfinance`.
""")

    add_md("""
### Example 52: Defining `read_data` and Fetching Stock Price Data
Define a helper function to read historical closing prices from Yahoo Finance for a dictionary of tickers.
""")
    add_code("""
import datetime as dt
import yfinance as yf

ticker_list = {
    'INTC': 'Intel',
    'MSFT': 'Microsoft',
    'IBM': 'IBM',
    'BHP': 'BHP',
    'TM': 'Toyota',
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'C': 'Citigroup',
    'QCOM': 'Qualcomm',
    'KO': 'Coca-Cola',
    'GOOG': 'Google'
}

def read_data(ticker_list,
              start=dt.datetime(2021, 1, 1),
              end=dt.datetime(2021, 12, 31)):
    \"\"\"
    This function reads in closing price data from Yahoo
    for each tick in the ticker_list.
    \"\"\"
    ticker = pd.DataFrame()
    for tick in ticker_list:
        stock = yf.Ticker(tick)
        prices = stock.history(start=start, end=end)
        # Change the index to date-only
        prices.index = pd.to_datetime(prices.index.date)
        closing_prices = prices['Close']
        ticker[tick] = closing_prices
    return ticker

ticker = read_data(ticker_list)
ticker.head()
""")

    add_md("""
### Example 53: Solution 1 - Calculating Percentage Change from First and Last Prices
Extract the first and last prices as Series and compute `(p2 - p1) / p1 * 100`.
""")
    add_code("""
p1 = ticker.iloc[0]   # Get the first set of prices as a Series
p2 = ticker.iloc[-1]  # Get the last set of prices as a Series
price_change = (p2 - p1) / p1 * 100
price_change
""")

    add_md("""
### Example 54: Solution 2 - Alternative Calculation with `.pct_change()`
Use the built-in `.pct_change()` method with `periods=len(ticker)-1` across rows.
""")
    add_code("""
change = ticker.pct_change(periods=len(ticker) - 1, axis='rows') * 100
price_change = change.iloc[-1]
price_change
""")

    add_md("""
### Example 55: Plotting Stock Percentage Change as a Bar Chart
Sort the percentage changes, map ticker symbols to company names, and plot as a horizontal/vertical bar chart.
""")
    add_code("""
price_change = price_change.copy()
price_change.sort_values(inplace=True)
price_change.rename(index=ticker_list, inplace=True)

fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlabel('stock', fontsize=12)
ax.set_ylabel('percentage change in price', fontsize=12)
price_change.plot(kind='bar', ax=ax)
plt.title('Percentage Price Change over 2021')
plt.show()
""")

    # Exercise 17.5.2
    add_md("""
---
### Exercise 17.5.2: Year-on-Year Percentage Change for Indices (1971–2021)
Calculate and visualize the year-on-year percentage return from 1971 to 2021 for four major market indices: S&P 500, NASDAQ, Dow Jones, and Nikkei.
""")

    add_md("""
### Example 56: Fetching Historical Market Indices Data (1971–2021)
Query historical closing prices for market indices over the 50-year period using `read_data`.
""")
    add_code("""
indices_list = {
    '^GSPC': 'S&P 500',
    '^IXIC': 'NASDAQ',
    '^DJI': 'Dow Jones',
    '^N225': 'Nikkei'
}

indices_data = read_data(
    indices_list,
    start=dt.datetime(1971, 1, 1),  # Common Start Date
    end=dt.datetime(2021, 12, 31)
)
indices_data.head()
""")

    add_md("""
### Example 57: Calculating Yearly Returns using `groupby`
Group prices by year, extract first and last prices per year, and calculate annual returns.
""")
    add_code("""
yearly_returns = pd.DataFrame()
for index, name in indices_list.items():
    p1 = indices_data.groupby(indices_data.index.year)[index].first()  # First price per year
    p2 = indices_data.groupby(indices_data.index.year)[index].last()   # Last price per year
    returns = (p2 - p1) / p1
    yearly_returns[name] = returns

yearly_returns
""")

    add_md("""
### Example 58: Summary Statistics with `.describe()`
Display descriptive summary statistics for the yearly returns across all four indices.
""")
    add_code("""
yearly_returns.describe()
""")

    add_md("""
### Example 59: Plotting 2x2 Grid of Subplots for Market Indices
Generate a 2x2 grid of time-series line charts showing annual percentage returns for each index over time.
""")
    add_code("""
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for iter_, ax in enumerate(axes.flatten()):
    index_name = yearly_returns.columns[iter_]
    ax.plot(yearly_returns[index_name])
    ax.set_ylabel("percent change", fontsize=12)
    ax.set_title(index_name)

plt.tight_layout()
plt.show()
""")

    out_file = 'pandas_all_examples.ipynb'
    with open(out_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Created {out_file} with {len(nb['cells'])} total cells.")

if __name__ == '__main__':
    create_notebook()
