# Three evidence-based observations (associations only)

1. Class imbalance (Figure 1A, eda_facts.json). 249 of 5110 rows have stroke = 1 (4.87%), against 4861 with stroke = 0. A model that always predicted 0 would score about 95% accuracy, so accuracy alone says little here. Training is oversampled and the test metrics focus on the positive class.
2. Missing and unavailable data (Figure 1B, missing_values.csv). bmi is blank in 201 rows (3.93%) and smoking_status is "Unknown" in 1544 rows (30.22%). Missing bmi is not evenly spread: 19.90% of rows with blank bmi have stroke = 1, against 4.26% of rows with bmi recorded. Unknown smoking stays its own category and is not merged with "never smoked".
3. Age pattern (Figure 1C, 1F). Median age is 71 in the stroke = 1 group and 43 in the stroke = 0 group. 72.69% of stroke = 1 rows are aged 60 or over, against 24.58% of stroke = 0 rows. Hypertension and heart disease show the same kind of association (stroke share 13.25% vs 3.97%, and 17.03% vs 4.18%, Figure 1H). These are retrospective associations, not causes.

Unusual values kept in the data and flagged: 43 ages under 1 year (non-integer ages exist, 115 rows), 856 rows under 18, bmi range 10.3 to 97.6 (3 rows under 12, 13 rows over 60), 1 row with gender = Other.
