"""
Coffee Shop Data Analysis and Prediction
- Data Cleaning
- Data Visualization
- AI Model for predicting next 10 orders
- Recommendations for coffee stock
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. DATA LOADING AND CLEANING
# =============================================================================

print("=" * 60)
print("COFFEE SHOP DATA ANALYSIS AND PREDICTION")
print("=" * 60)

# Load data
df = pd.read_csv('index_1.csv')

print("\n--- ORIGINAL DATA INFO ---")
print(f"Total records: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# Check for missing values
print(f"\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Data Cleaning
print("\n--- DATA CLEANING ---")

# Remove duplicates
initial_count = len(df)
df = df.drop_duplicates()
print(f"Removed {initial_count - len(df)} duplicate records")

# Fill missing values
# For cash_type - fill with 'unknown'
df['cash_type'] = df['cash_type'].fillna('unknown')
# For card - fill with 'N/A'
df['card'] = df['card'].fillna('N/A')
# For money - fill with median
df['money'] = df['money'].fillna(df['money'].median())
# For coffee_name - fill with 'Unknown'
df['coffee_name'] = df['coffee_name'].fillna('Unknown')

# Convert date columns
df['date'] = pd.to_datetime(df['date'])
df['datetime'] = pd.to_datetime(df['datetime'])

# Extract additional features
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day

# Remove invalid money values (negative or zero)
df = df[df['money'] > 0]

print(f"Final cleaned records: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# =============================================================================
# 2. DATA VISUALIZATION
# =============================================================================

print("\n--- GENERATING VISUALIZATIONS ---")

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Figure 1: Coffee Type Distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Coffee type counts
coffee_counts = df['coffee_name'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(coffee_counts)))

ax1 = axes[0, 0]
bars = ax1.bar(coffee_counts.index, coffee_counts.values, color=colors)
ax1.set_title('Coffee Type Sales Distribution', fontsize=14, fontweight='bold')
ax1.set_xlabel('Coffee Type')
ax1.set_ylabel('Number of Orders')
ax1.tick_params(axis='x', rotation=45)
for bar, count in zip(bars, coffee_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom', fontsize=9)

# Pie chart
ax2 = axes[0, 1]
ax2.pie(coffee_counts.values, labels=coffee_counts.index, autopct='%1.1f%%', 
        colors=colors, startangle=90)
ax2.set_title('Coffee Type Percentage', fontsize=14, fontweight='bold')

# Daily sales trend
daily_sales = df.groupby('date').size()
ax3 = axes[1, 0]
ax3.plot(daily_sales.index, daily_sales.values, alpha=0.7, linewidth=1)
ax3.fill_between(daily_sales.index, daily_sales.values, alpha=0.3)
ax3.set_title('Daily Sales Trend', fontsize=14, fontweight='bold')
ax3.set_xlabel('Date')
ax3.set_ylabel('Number of Orders')
ax3.tick_params(axis='x', rotation=45)

# Hourly sales pattern
hourly_sales = df.groupby('hour').size()
ax4 = axes[1, 1]
ax4.bar(hourly_sales.index, hourly_sales.values, color='steelblue', alpha=0.8)
ax4.set_title('Hourly Sales Pattern', fontsize=14, fontweight='bold')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Number of Orders')
ax4.set_xticks(range(0, 24))

plt.tight_layout()
plt.savefig('coffee_analysis_charts.png', dpi=150, bbox_inches='tight')
print("Saved: coffee_analysis_charts.png")

# Figure 2: Additional Analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

# Payment type distribution
payment_counts = df['cash_type'].value_counts()
ax5 = axes2[0]
bars2 = ax5.bar(payment_counts.index, payment_counts.values, color=['#2ecc71', '#e74c3c', '#3498db'])
ax5.set_title('Payment Type Distribution', fontsize=14, fontweight='bold')
ax5.set_xlabel('Payment Type')
ax5.set_ylabel('Number of Orders')
for bar, count in zip(bars2, payment_counts.values):
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
             str(count), ha='center', va='bottom', fontsize=10)

# Coffee by day of week
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
dow_sales = df.groupby('day_of_week').size()
ax6 = axes2[1]
ax6.bar(dow_names, dow_sales.values, color='coral', alpha=0.8)
ax6.set_title('Sales by Day of Week', fontsize=14, fontweight='bold')
ax6.set_xlabel('Day of Week')
ax6.set_ylabel('Number of Orders')
ax6.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('coffee_analysis_charts2.png', dpi=150, bbox_inches='tight')
print("Saved: coffee_analysis_charts2.png")

# =============================================================================
# 3. STATISTICAL ANALYSIS
# =============================================================================

print("\n--- STATISTICAL ANALYSIS ---")

# Coffee type statistics
print("\nCoffee Type Sales Ranking:")
for i, (coffee, count) in enumerate(coffee_counts.items(), 1):
    percentage = (count / len(df)) * 100
    print(f"  {i}. {coffee}: {count} orders ({percentage:.1f}%)")

# Average daily sales
avg_daily = daily_sales.mean()
print(f"\nAverage daily sales: {avg_daily:.1f} orders")

# Peak hours
peak_hours = hourly_sales.nlargest(3)
print(f"\nPeak hours: {list(peak_hours.index)}")

# =============================================================================
# 4. AI MODEL FOR PREDICTION
# =============================================================================

print("\n--- BUILDING AI MODEL ---")

# Prepare features for prediction
# We'll predict coffee type based on: day_of_week, hour, month

# Create feature matrix
features = df[['day_of_week', 'hour', 'month']].copy()

# Encode coffee names
le = LabelEncoder()
df['coffee_encoded'] = le.fit_transform(df['coffee_name'])

# Target variable
y = df['coffee_encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': ['day_of_week', 'hour', 'month'],
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(f"\nFeature Importance:")
print(feature_importance)

# =============================================================================
# 5. PREDICT NEXT 10 ORDERS
# =============================================================================

print("\n" + "=" * 60)
print("PREDICTING NEXT 10 ORDERS")
print("=" * 60)

# Get the last date in the dataset
last_date = df['date'].max()
print(f"\nLast date in dataset: {last_date.strftime('%Y-%m-%d')}")

# Predict next 10 orders
# We'll simulate the next 10 order times based on typical patterns
predictions = []

# Get typical hours for next orders (using distribution)
typical_hours = hourly_sales.nlargest(5).index.tolist()

for i in range(10):
    # Simulate next order features
    day_of_week = (last_date.dayofweek + (i // 3 + 1)) % 7  # Rotate through days
    hour = typical_hours[i % len(typical_hours)]
    month = last_date.month
    
    # Create feature vector
    features_pred = np.array([[day_of_week, hour, month]])
    
    # Predict
    pred = model.predict(features_pred)
    pred_coffee = le.inverse_transform(pred)[0]
    
    # Get prediction probability
    proba = model.predict_proba(features_pred)[0]
    confidence = max(proba) * 100
    
    predictions.append({
        'order_num': i + 1,
        'day_of_week': dow_names[day_of_week],
        'hour': hour,
        'predicted_coffee': pred_coffee,
        'confidence': confidence
    })
    
    print(f"Order {i+1}: {dow_names[day_of_week]} at {hour}:00 - Predicted: {pred_coffee} (Confidence: {confidence:.1f}%)")

# =============================================================================
# 6. RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 60)
print("RECOMMENDATIONS FOR COFFEE STOCK")
print("=" * 60)

# Analyze which coffee to stock more
top_coffees = coffee_counts.head(5)

print("\n📊 RECOMMENDATION: Stock more of these coffees (by popularity):")
print("-" * 50)

for i, (coffee, count) in enumerate(top_coffees.items(), 1):
    percentage = (count / len(df)) * 100
    bar = "█" * int(percentage / 2)
    print(f"  {i}. {coffee:20s} {bar} {percentage:.1f}%")

# Calculate recommended stock ratio
total_orders = len(df)
print(f"\n📈 Recommended Stock Ratio for Next Day:")
print("-" * 50)
for coffee, count in coffee_counts.items():
    ratio = (count / total_orders) * 100
    print(f"  {coffee:20s}: {ratio:5.1f}%")

# Best time recommendations
print(f"\n⏰ Peak Hours (stock up during these times):")
print("-" * 50)
for hour in peak_hours.index:
    print(f"  - {hour}:00 ({peak_hours[hour]} orders)")

# Day recommendations
print(f"\n📅 Best Days (highest sales):")
print("-" * 50)
best_days = dow_sales.nlargest(3)
for day in best_days.index:
    print(f"  - {dow_names[day]}: {best_days[day]} orders")

# =============================================================================
# 7. SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

# Most popular coffee
most_popular = coffee_counts.idxmax()
most_popular_count = coffee_counts.max()

print(f"""
📌 KEY FINDINGS:

1. Total Orders Analyzed: {len(df):,}
2. Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
3. Most Popular Coffee: {most_popular} ({most_popular_count} orders, {(most_popular_count/len(df))*100:.1f}%)
4. Average Daily Orders: {avg_daily:.1f}
5. Model Prediction Accuracy: {accuracy*100:.1f}%

🎯 RECOMMENDATION FOR NEXT 10 ORDERS:
   You should prepare MORE of these coffees:
   1. {coffee_counts.index[0]} (Highest demand)
   2. {coffee_counts.index[1]}
   3. {coffee_counts.index[2]}

💡 STOCKING STRATEGY:
   - Focus on {most_popular} as it represents {((most_popular_count/len(df))*100):.1f}% of orders
   - Ensure sufficient stock during peak hours ({peak_hours.index[0]}:00-{peak_hours.index[0]+2}:00)
   - Weekend days ({dow_names[5]}, {dow_names[6]}) may need extra preparation
""")

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions)
predictions_df.to_csv('next_10_predictions.csv', index=False)
print("\n✅ Predictions saved to: next_10_predictions.csv")
print("✅ Analysis complete!")

