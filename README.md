# ReatilAnalytics
# ============================================================
# RETAIL ANALYTICS USING PYTHON
# Sales Analysis, Customer Segmentation,
# Market Basket Analysis and Business Insights
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. CREATE SAMPLE RETAIL TRANSACTION DATA
# ------------------------------------------------------------

data = {
    "Transaction_ID": [
        1001, 1002, 1003, 1004, 1005,
        1006, 1007, 1008, 1009, 1010,
        1011, 1012, 1013, 1014, 1015,
        1016, 1017, 1018, 1019, 1020
    ],

    "Date": [
        "2026-01-05", "2026-01-05", "2026-01-06", "2026-01-07",
        "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-10",
        "2026-01-11", "2026-01-12", "2026-01-13", "2026-01-14",
        "2026-01-15", "2026-01-16", "2026-01-17", "2026-01-18",
        "2026-01-19", "2026-01-20", "2026-01-21", "2026-01-22"
    ],

    "Customer_ID": [
        "C101", "C102", "C101", "C103", "C104",
        "C102", "C105", "C101", "C106", "C107",
        "C103", "C108", "C105", "C109", "C102",
        "C110", "C106", "C101", "C108", "C105"
    ],

    "Product": [
        "Laptop", "Mouse", "Keyboard", "Laptop", "Headphones",
        "Mouse", "Keyboard", "Laptop", "Monitor", "Mouse",
        "Headphones", "Laptop", "Mouse", "Monitor", "Keyboard",
        "Laptop", "Headphones", "Mouse", "Keyboard", "Laptop"
    ],

    "Category": [
        "Electronics", "Accessories", "Accessories", "Electronics",
        "Accessories", "Accessories", "Accessories", "Electronics",
        "Electronics", "Accessories", "Accessories", "Electronics",
        "Accessories", "Electronics", "Accessories", "Electronics",
        "Accessories", "Accessories", "Accessories", "Electronics"
    ],

    "Quantity": [
        1, 2, 1, 1, 2,
        3, 2, 1, 1, 2,
        1, 1, 4, 1, 2,
        1, 2, 3, 1, 1
    ],

    "Price": [
        60000, 800, 1500, 60000, 2500,
        800, 1500, 60000, 12000, 800,
        2500, 60000, 800, 12000, 1500,
        60000, 2500, 800, 1500, 60000
    ]
}


df = pd.DataFrame(data)


# ------------------------------------------------------------
# 2. DISPLAY RAW DATA
# ------------------------------------------------------------

print("\n========== RAW TRANSACTION DATA ==========\n")
print(df)

print("\nDataset Shape:", df.shape)


# ------------------------------------------------------------
# 3. DATA PREPARATION
# ------------------------------------------------------------

# Convert Date column into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Check missing values
print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing important values
df = df.dropna(
    subset=[
        "Transaction_ID",
        "Customer_ID",
        "Product",
        "Category",
        "Quantity",
        "Price"
    ]
)


# ------------------------------------------------------------
# 4. CALCULATE REVENUE
# ------------------------------------------------------------

# Revenue = Quantity × Price
df["Revenue"] = df["Quantity"] * df["Price"]

print("\n========== TRANSACTION DATA WITH REVENUE ==========\n")
print(
    df[
        [
            "Transaction_ID",
            "Customer_ID",
            "Product",
            "Quantity",
            "Price",
            "Revenue"
        ]
    ]
)


# ------------------------------------------------------------
# 5. TOTAL REVENUE
# ------------------------------------------------------------

total_revenue = df["Revenue"].sum()

print("\n========== SALES SUMMARY ==========\n")
print("Total Revenue: ₹", round(total_revenue, 2))

print("Total Quantity Sold:", df["Quantity"].sum())

print("Number of Transactions:", df["Transaction_ID"].nunique())

print("Number of Customers:", df["Customer_ID"].nunique())

print("Number of Products:", df["Product"].nunique())


# ------------------------------------------------------------
# 6. DAILY SALES
# ------------------------------------------------------------

daily_sales = (
    df.groupby("Date")["Revenue"]
    .sum()
    .sort_index()
)

print("\n========== DAILY SALES ==========\n")
print(daily_sales)


# ------------------------------------------------------------
# 7. MONTHLY SALES
# ------------------------------------------------------------

df["Month"] = df["Date"].dt.to_period("M").astype(str)

monthly_sales = (
    df.groupby("Month")["Revenue"]
    .sum()
    .sort_index()
)

print("\n========== MONTHLY SALES ==========\n")
print(monthly_sales)


# ------------------------------------------------------------
# 8. CATEGORY-WISE SALES
# ------------------------------------------------------------

category_sales = (
    df.groupby("Category")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== CATEGORY-WISE REVENUE ==========\n")
print(category_sales)


# ------------------------------------------------------------
# 9. PRODUCT-WISE SALES
# ------------------------------------------------------------

product_sales = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== PRODUCT-WISE REVENUE ==========\n")
print(product_sales)


# ------------------------------------------------------------
# 10. TOP-SELLING CATEGORY
# ------------------------------------------------------------

top_category = category_sales.idxmax()
top_category_revenue = category_sales.max()

print("\n========== TOP-SELLING CATEGORY ==========\n")
print("Top Category:", top_category)
print("Revenue: ₹", top_category_revenue)


# ------------------------------------------------------------
# 11. TOP-SELLING PRODUCT
# ------------------------------------------------------------

top_product = product_sales.idxmax()
top_product_revenue = product_sales.max()

print("\n========== TOP-SELLING PRODUCT ==========\n")
print("Top Product:", top_product)
print("Revenue: ₹", top_product_revenue)


# ------------------------------------------------------------
# 12. CUSTOMER SPENDING
# ------------------------------------------------------------

customer_spending = (
    df.groupby("Customer_ID")["Revenue"]
    .sum()
    .sort_values(ascending=False)
)

print("\n========== CUSTOMER SPENDING ==========\n")
print(customer_spending)


# ------------------------------------------------------------
# 13. CUSTOMER SEGMENTATION USING QUARTILES
# ------------------------------------------------------------

q1 = customer_spending.quantile(0.25)
q2 = customer_spending.quantile(0.50)
q3 = customer_spending.quantile(0.75)

print("\n========== CUSTOMER SPENDING QUARTILES ==========\n")
print("Q1:", q1)
print("Q2:", q2)
print("Q3:", q3)


def customer_segment(amount):

    if amount <= q1:
        return "Low Value"

    elif amount <= q2:
        return "Medium Value"

    elif amount <= q3:
        return "High Value"

    else:
        return "Premium"


customer_segments = customer_spending.reset_index()

customer_segments.columns = [
    "Customer_ID",
    "Total_Spending"
]

customer_segments["Segment"] = (
    customer_segments["Total_Spending"]
    .apply(customer_segment)
)

print("\n========== CUSTOMER SEGMENTS ==========\n")
print(customer_segments)


# ------------------------------------------------------------
# 14. MARKET BASKET ANALYSIS
# ------------------------------------------------------------

# Create a dictionary containing products purchased
# in each transaction.

transaction_products = (
    df.groupby("Transaction_ID")["Product"]
    .apply(list)
)

print("\n========== PRODUCTS IN EACH TRANSACTION ==========\n")

for transaction, products in transaction_products.items():

    print(
        "Transaction",
        transaction,
        ":",
        products
    )


# Generate product pairs
from itertools import combinations

pair_count = {}

for products in transaction_products:

    # Remove duplicate products within a transaction
    products = list(set(products))

    # Generate all possible product pairs
    pairs = combinations(sorted(products), 2)

    for pair in pairs:

        if pair not in pair_count:
            pair_count[pair] = 0

        pair_count[pair] += 1


# Convert pairs into DataFrame
basket_analysis = pd.DataFrame(
    pair_count.items(),
    columns=["Product_Pair", "Frequency"]
)

basket_analysis = basket_analysis.sort_values(
    by="Frequency",
    ascending=False
)

print("\n========== MARKET BASKET ANALYSIS ==========\n")
print(basket_analysis)


# ------------------------------------------------------------
# 15. MOST FREQUENT PRODUCT PAIR
# ------------------------------------------------------------

if len(basket_analysis) > 0:

    most_frequent_pair = basket_analysis.iloc[0]

    print("\n========== MOST FREQUENT PRODUCT PAIR ==========\n")

    print(
        "Product Pair:",
        most_frequent_pair["Product_Pair"]
    )

    print(
        "Purchase Frequency:",
        most_frequent_pair["Frequency"]
    )

else:

    print("\nNo product pairs found.")


# ------------------------------------------------------------
# 16. DAILY SALES VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    daily_sales.index,
    daily_sales.values,
    marker="o"
)

plt.title("Daily Revenue Trend")

plt.xlabel("Date")

plt.ylabel("Revenue (₹)")

plt.xticks(rotation=45)

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 17. MONTHLY SALES VISUALIZATION
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    monthly_sales.index,
    monthly_sales.values,
    marker="o"
)

plt.title("Monthly Revenue Trend")

plt.xlabel("Month")

plt.ylabel("Revenue (₹)")

plt.grid(True)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 18. CATEGORY SALES BAR CHART
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

category_sales.plot(
    kind="bar"
)

plt.title("Revenue by Category")

plt.xlabel("Category")

plt.ylabel("Revenue (₹)")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 19. CUSTOMER SEGMENT VISUALIZATION
# ------------------------------------------------------------

segment_counts = (
    customer_segments["Segment"]
    .value_counts()
)

plt.figure(figsize=(8, 5))

segment_counts.plot(
    kind="bar"
)

plt.title("Customer Segmentation")

plt.xlabel("Customer Segment")

plt.ylabel("Number of Customers")

plt.xticks(rotation=0)

plt.tight_layout()

plt.show()


# ------------------------------------------------------------
# 20. BUSINESS INSIGHTS
# ------------------------------------------------------------

print("\n============================================")
print("           BUSINESS INSIGHTS")
print("============================================")

print(
    "\n1. Total revenue generated: ₹",
    round(total_revenue, 2)
)

print(
    "2. Top-selling category:",
    top_category
)

print(
    "3. Revenue from top category: ₹",
    round(top_category_revenue, 2)
)

print(
    "4. Top-selling product:",
    top_product
)

print(
    "5. Revenue from top product: ₹",
    round(top_product_revenue, 2)
)

print(
    "6. Highest-value customer:",
    customer_spending.idxmax()
)

print(
    "7. Highest customer spending: ₹",
    customer_spending.max()
)


# ------------------------------------------------------------
# 21. FINAL RECOMMENDATIONS
# ------------------------------------------------------------

print("\n========== BUSINESS RECOMMENDATIONS ==========\n")

print(
    "1. Maintain sufficient inventory for the top-selling category."
)

print(
    "2. Promote frequently purchased product pairs as bundles."
)

print(
    "3. Offer loyalty rewards to Premium and High Value customers."
)

print(
    "4. Use daily and monthly sales trends to plan inventory."
)

print(
    "5. Focus marketing campaigns on products generating high revenue."
)

print(
    "6. Use customer segments to create personalized offers."
)


# ------------------------------------------------------------
# PROJECT COMPLETED
# ------------------------------------------------------------

print("\n============================================")
print("       RETAIL ANALYTICS PROJECT COMPLETED")
print("============================================")
