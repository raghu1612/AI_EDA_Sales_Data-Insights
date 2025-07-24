# AI_EDA_Sales_Data-Insights#

1.What is Exploratory Data Analysis (EDA)?

**Definition:**
EDA stands for Exploratory Data Analysis. In simple terms, EDA is about understanding a
dataset in detail.

**Purpose of EDA:**
Data Exploration: Understanding the dataset before diving into formal analysis or
modelling.
Visualisation: Creating charts, graphs, and dashboards to summarize and present
data.
Identifying Patterns and Anomalies: Spotting interesting trends, patterns, and any
unusual data points (outliers).
Testing Initial Hypotheses: Checking assumptions about the dataset through
visualisation and basic statistics.

**EDA** is all about exploring data to understand its main characteristics before any formal modeling. In simple terms, we use EDA to summarize what our data looks like, often using stats and visualizations.

Why EDA Matters
Spot patterns
Find anomalies
Test initial hypotheses
Check assumptions

#Introduction to EDA & Basic Statistics in Python

Welcome everyone! Today we’ll dive into **Exploratory Data Analysis (EDA)** – a fundamental step in any data project.

*Last week, we learnt how to use Google Colab and Gemini AI for data segregation. Today, we’re moving to a more advanced concept – analyzing data to prepare dashboards and graphs.*

EDA is all about exploring data to understand its main characteristics before any formal modeling.
In simple terms, we use EDA to **summarize what our data looks like**, often using stats and visualizations.

### Why EDA Matters

* Spot patterns
* Find anomalies
* Test initial hypotheses
* Check assumptions

---

### 🔍 Starting With A New Dataset

1. **Preview data**
   Use `df.head()` in Pandas to look at a few rows.

2. **Understand Data Types**
   Check for numbers, dates, text columns.

3. **Check for Missing or Odd Values**

---

### 📀 Basic Statistics

* **Count** – How many data points?
* **Mean** – Average value (e.g., average age of users)
* **Median** – Middle value when sorted (better for skewed data)
* **Standard Deviation** – Spread of values from the mean
* **Min/Max** – Range of values (spot outliers)

 Use `df.describe()` to get all of this at once.

---
### Basic Statistics Example

```python
data = [1, 2, 2, 3, 14]
mean = sum(data)/len(data)  # 4.4
```

* **Mean** = 4.4 seconds (pulled up by the outlier 14)
* **Median** = 2 seconds (better "typical" value)
* **Standard Deviation** = Larger due to outlier

💡 **Use median for skewed data** like income, web traffic, etc.

---

### 🧮 Understanding Distributions

* Is the data **normal** (bell-shaped)?
* Is it **skewed** (left or right)?
* Are there **multiple peaks** (bi-modal)?

Use visual tools:

* **Histograms**
* **Box plots**

> **Histogram** groups values into bins and shows how frequent each bin is.
> E.g., most website sessions are under 200 seconds, but there’s a long tail of longer ones → right-skewed.

---

###  Key Python Libraries for EDA

#### 1. Pandas

* Load data: `pd.read_csv('data.csv')`
* Clean data: handle missing values, convert types
* Compute stats: `df.mean()`, `df['column'].median()`
* Slice and dice: filter and aggregate

>  Pandas is the backbone of data analysis in Python!

#### 2. Matplotlib

* Core plotting library in Python
* Create line plots, bar charts, pie charts, histograms
* Import as: `import matplotlib.pyplot as plt`

> It may not be pretty by default, but it's powerful and flexible!

#### 3. Seaborn

* Built on top of Matplotlib
* Easier to make **beautiful** plots
* Integrates well with Pandas

Examples:

* Correlation heatmaps
* Box plots
* Violin plots

> Seaborn provides visually appealing defaults and statistical overlays (like confidence intervals).

---

### ⚖️ Typical EDA Workflow

1. **Load Data**

```python
df = pd.read_csv('data.csv')
```

2. **Understand Structure**

```python
df.shape
df.columns
df.dtypes
```

3. **Clean Data**

```python
df.dropna()
# or
df.fillna(value)
df['date'] = pd.to_datetime(df['date'])
```

4. **Compute Statistics**

```python
df.describe()
df['sales'].mean()
```

5. **Visualize Distributions**

```python
import seaborn as sns
sns.histplot(df['sales'])
sns.boxplot(y=df['sales'])
```

6. **Explore Relationships**

```python
sns.scatterplot(x='ad_spend', y='sales', data=df)
```

7. **Iterate**
   Ask new questions, explore deeper, create new features.

> EDA is a **playful, iterative** process. It's like detective work with data! 
