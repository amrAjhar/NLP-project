# Dataset Column Name Guide

## Problem: `KeyError: 'label'`

This error means your dataset doesn't have a column named `'label'`. This is common because different datasets use different column naming conventions.

## Solution: Identify Your Column Names

### Step 1: Find Your Column Names

Run this code in a new Colab cell BEFORE running the full pipeline:

```python
import pandas as pd

# Load your dataset
df = pd.read_csv('/path/to/your/data.csv')

# Show column names
print("Column names in your dataset:")
print(df.columns.tolist())

# Show first row
print("\nFirst row:")
print(df.iloc[0])

# Show data types
print("\nData types:")
print(df.dtypes)
```

### Step 2: Identify Which Columns to Use

Look at the output and find:
1. **TEXT column**: Which column contains the text/content?
   - Common names: `text`, `content`, `body`, `sentence`, `document`, `generated_text`, `message`
   
2. **LABEL column**: Which column contains the label (0 or 1)?
   - Common names: `label`, `class`, `target`, `is_ai`, `is_generated`, `category`, `source`, `human_generated`

### Step 3: Set Columns in Notebook

In the **"Identify text and label columns"** cell, uncomment and modify:

```python
# If auto-detection didn't work, MANUALLY SET these:
TEXT_COLUMN = 'your_text_column_name'      # Replace with actual column name
LABEL_COLUMN = 'your_label_column_name'    # Replace with actual column name
```

**Example 1:**
```python
TEXT_COLUMN = 'generated_text'
LABEL_COLUMN = 'is_ai'
```

**Example 2:**
```python
TEXT_COLUMN = 'sentence'
LABEL_COLUMN = 'source'  # 0 = human, 1 = AI
```

## Common Dataset Formats

### Kaggle "AI vs Human Text" Dataset
```python
TEXT_COLUMN = 'text'
LABEL_COLUMN = 'label'  # 0 = human, 1 = AI
```

### Alternative Dataset Columns
| Format | TEXT | LABEL |
|--------|------|-------|
| Format A | `text` | `label` |
| Format B | `content` | `class` |
| Format C | `message` | `is_ai` |
| Format D | `sentence` | `target` |
| Format E | `generated_text` | `source` |

## Verification Checklist

After setting column names, run this verification code:

```python
# Verify columns exist
assert TEXT_COLUMN in df.columns, f"❌ Column '{TEXT_COLUMN}' not found!"
assert LABEL_COLUMN in df.columns, f"❌ Column '{LABEL_COLUMN}' not found!"

# Verify label values (should be 0 and 1 or similar)
print(f"Unique labels: {df[LABEL_COLUMN].unique()}")
print(f"Label value counts:\n{df[LABEL_COLUMN].value_counts()}")

# Verify text column has content
print(f"\nSample text (first 100 chars):")
print(df[TEXT_COLUMN].iloc[0][:100])

print("\n✅ All checks passed!")
```

## Troubleshooting

### Issue: Text column is empty or null
```python
# Check for missing values
print(df[TEXT_COLUMN].isnull().sum())

# Remove rows with missing text
df = df[df[TEXT_COLUMN].notna()].reset_index(drop=True)
```

### Issue: Label column has unexpected values
```python
# Check unique values
print(df[LABEL_COLUMN].unique())

# If labels are not 0/1, they might be strings like 'human'/'ai'
# Map them to 0/1:
df[LABEL_COLUMN] = df[LABEL_COLUMN].map({'human': 0, 'ai': 1})
```

### Issue: Multiple text columns
```python
# Combine them
df[TEXT_COLUMN] = df['col1'].astype(str) + ' ' + df['col2'].astype(str)
```

## Quick Test Code

Add this to a cell after loading data to verify everything works:

```python
print("="*60)
print("DATASET VERIFICATION")
print("="*60)

print(f"\n✅ Text column: {TEXT_COLUMN}")
print(f"   Shape: {df[TEXT_COLUMN].shape}")
print(f"   Sample: {str(df[TEXT_COLUMN].iloc[0])[:100]}...")

print(f"\n✅ Label column: {LABEL_COLUMN}")
print(f"   Unique values: {df[LABEL_COLUMN].unique()}")
print(f"   Distribution:\n{df[LABEL_COLUMN].value_counts()}")

print("\n✅ Ready to run pipeline!")
```

## Contact/Help

If you still have issues:
1. Check the column names printout from the first code block
2. Make sure TEXT_COLUMN and LABEL_COLUMN are spelled exactly as they appear
3. Verify they're strings (in quotes): `TEXT_COLUMN = 'text'` not `TEXT_COLUMN = text`
