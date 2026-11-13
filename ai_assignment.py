'''Importing Libraries'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''----------------------Loading and exploring Data------------------------------'''

#reading the dataset to python 
df = pd.read_csv('Earnings_Hours_Labour_Employment.csv')

#printing the total number of rows and columns in the dataset
print ("Total rows:", len(df))
print ("Total columns:", len(df.columns))

#Printing the first and last 5 rows of the dataset
print("\nFirst five rows:")
print(df.head())
print("\nLast five row:")
print(df.tail())

#printing the column names of the dataset
print("\nColumn names:")
print(df.columns.tolist())

#printing the shape of the dataset
print("\nDataset Shape: ")
print(df.shape)
#printing the number of unique values
print("\nUnique values per column: ")
print(df.nunique())

#printing the number of null values
print("\nMissing values per column: ")
print(df.isnull().sum())

'''-----------------------Checking Statistics-----------------------'''
#displaying the statistics and their available values
summary = (
    df.groupby("Statistic Label")["VALUE"]
    .agg(
        total_rows="size",
        available_values="count"
    )
    .reset_index()
)
#printing Statistics Summary
print("\nStatistics Summary: ")
print(summary.to_string(index=False))

'''-------------------Exploring Average Weekly Earnings----------------------------------'''

earnings_df =df[
    df["Statistic Label"] == "Average Weekly Earnings"
].copy()

#Printing Total rows and available earnings values
print("\nWeekly Earnings Data: ")
print("Total rows:", len(earnings_df))
print("Available values:", earnings_df["VALUE"].notna().sum())
print("Missing values: ", earnings_df["VALUE"].isna().sum())

#Printing Employee Categories
print("\nEmployee Categories: ")
print(earnings_df["Type of Employee"].unique())

'''----------------------Investiationg missing values------------------'''
#Inspecting distribution of observations
summary = (
    earnings_df
    .groupby([
        "Economic Sector NACE Rev 2",
        "Type of Employee"
    ])["VALUE"]
    .agg(
        available="count",
        missing=lambda x: x.isna().sum()
        
    )
    .reset_index()
)

#Total available and missing values for each economic sector and type of employee
sector_summary = (
    summary.groupby("Economic Sector NACE Rev 2")[
        ["available", "missing"]
    ]
    .sum()
)

'''-----------------Visualising DATA-------------------'''

sector_summary[["available","missing"]].plot(
    kind="bar",
    figsize=(12, 6),
    color=["steelblue", "lightcoral"]
)

plt.title("Total Available and Missing Values by Economic Sector")
plt.xlabel("Economic Sector")
plt.ylabel("Number of Observations")
plt.xticks(rotation=45, ha="right")
plt.legend(title="Data Status")
plt.tight_layout()
plt.show()

#Calculate the percentage of missing observations
sector_summary["missing_percentage"] = (
    sector_summary["missing"]/
    (sector_summary["available"] + sector_summary["missing"])
    ) * 100
    
sector_summary = sector_summary.sort_values(
    "missing_percentage",
    ascending=True
)    

#Plot percentage of missing observations
plt.figure(figsize=(12, 8))

plt.barh(
    sector_summary.index,
    sector_summary["missing_percentage"],
    color="coral"
    
)

plt.title("Percentage of Missing Observations by Economic Sector")
plt.xlabel("Missing Observations (%)")
plt.ylabel("Economic Sector")
plt.xlim(0, 100)
plt.tight_layout()
plt.show()


'''Calculating the correlation between time and average weekly earnings - all employees '''

all_employees = earnings_df[
    (earnings_df["Type of Employee"] == "All employees") &
    (earnings_df["Economic Sector NACE Rev 2"] == "All NACE economic sectors")
].copy()

print("\nAll employees ovservations: ", len(all_employees))
print("Available earnings: ", all_employees["VALUE"].notna().sum())

#removing the missing data
correlation_df = all_employees.dropna(subset=["VALUE"]).copy()

#sort by quarter
correlation_df = correlation_df.sort_values("TLIST(Q1)")
correlation_df = correlation_df.reset_index(drop=True)

#Checking for duplicate and missing quarters
quarters = pd.PeriodIndex(correlation_df["Quarter"], freq="Q")

if quarters.has_duplicates:
    raise ValueError("Duplicate quarters found in earnings data.")

expected_quarters =pd.period_range(
    start=quarters.min(),
    end=quarters.max(),
    freq="Q"
)

missing_quarters = expected_quarters.difference(quarters)
print("\nMissing quarters: ", list(missing_quarters))


if len(missing_quarters) > 0:
    raise ValueError("Missing quarters found. Check data before analysis.")

#creating a numberical time variable
correlation_df["Time"] = range(len(correlation_df))

print("\nData for correlation: ")
print(correlation_df[["Quarter", "VALUE"]].head(10))
print("Number of observations: ", len(correlation_df))   

'''----------------Calculating the correlation-------------------'''

correlation = correlation_df["Time"].corr(
    correlation_df["VALUE"]
)

print(f"\nCorrelation coefficient: {correlation:.3f}")


#Visualising the correlation and regression line
plt.figure(figsize=(12, 6) )

sns.regplot(
    data=correlation_df,
    x="Time",
    y="VALUE",
    scatter_kws={"color": "steelblue", "alpha": 0.7},
    line_kws={"color": "red"}
)

plt.title(f"Time and Average Weekly earnings (r = {correlation:.3f})")
plt.xlabel("Time(Quarterly Observations)")
plt.ylabel("Average Weekly Earnings (€)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

#Calculating the percentage change in earnings from quarter to quarter
correlation_df["Earnings_Growth"] = (
    correlation_df["VALUE"].pct_change() * 100
)

#Display recent quarterly changes
print(
    correlation_df[
        ["Quarter", "VALUE", "Earnings_Growth"]
        
    ].tail(10)
)

#shown one label per year (every 4 quarters) for both charts
tick_positions = range(0, len(correlation_df), 4)

tick_labels = correlation_df.iloc[
    list(tick_positions)
    
]["Quarter"].str[:4]

'''visualise quarterly earnings growth'''

plt.figure(figsize=(12, 6))

plt.plot(
    correlation_df["Time"],
    correlation_df["Earnings_Growth"],
    marker="o",
    markersize=3,
    color="steelblue",
)
plt.axhline(
    0,
    color="red",
    linestyle="--",
    label="No Growth"
)

#show fewer quarter lables
plt.xticks(
    tick_positions,
    tick_labels,
    rotation=45
)

plt.title("Quarterly Growth in Average Weekly Earnings")
plt.xlabel("Year")
plt.ylabel("Quarterly Earnings Growth (%)")
plt.legend()

plt.grid(alpha=0.3 )
plt.tight_layout()
plt.show()

'''Calculating and plotting year-on-year percentage growth'''
correlation_df["Annual_Growth"] = (
    correlation_df["VALUE"].pct_change(4) * 100
    
)

print("\nRecent Year-on-Year Earnings Growth: ")
print(
    correlation_df[
        ["Quarter", "VALUE", "Annual_Growth"]
    ].tail(10)
)

#plot year on year earnings growth
plt.figure(figsize=(12, 5))

plt.plot(
    correlation_df["Time"],
    correlation_df["Annual_Growth"],
    marker="o", 
    markersize=3,
    color="darkgreen"
    
)

plt.axhline(0, color="red", linestyle="--", label="No growth")
plt.xticks(tick_positions, tick_labels, rotation=45)
plt.title("Year-on-Year Growth in AVerage Weekly Earnings")
plt.xlabel("Year")
plt.ylabel("Annual Earnings Growth (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
           