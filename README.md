# SC5002 Lab Assignment 1: Search Algorithms and Regression

**Author:** Ambrose Chan Wye Yang  
**GitHub:** [Ambrose3112](https://github.com/Ambrose3112)  
**Module:** SC5002

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ambrose3112/SC5002-Lab-Assignment-1/blob/main/SC5002_Lab_Assignment_1_Part_2.ipynb)

## Overview

This assignment combines classical search problems with a small machine-learning experiment. It also explores how generative AI can support learning when its suggestions are checked against calculations, task definitions and experimental results.

- **Part 1:** Classical Search & Problem Solving
- **Part 2:** Applied Machine Learning with AI Copilot
- **Part 3:** Final Synthesis, Video & Submission

The Python notebook covers **Part 2**. The manual search working, AI prompts and reflections are documented in the separate report.

## Repository contents

| File | Description |
|---|---|
| README.md | Project overview, methods and findings. |
| [SC5002_Lab_Assignment_1_Part_2.ipynb](SC5002_Lab_Assignment_1_Part_2.ipynb) | Colab notebook containing the dataset setup, model evaluation and comparison chart. |

**Repository:** https://github.com/Ambrose3112/SC5002-Lab-Assignment-1

## Completed notebook

All cells in the Google Colab notebook have been run. The dataset preview, model scores and comparison chart are already displayed in the saved outputs.

[View the completed Google Colab notebook](https://colab.research.google.com/drive/1H6rNGPJKZdstX19bc371AssdIkdUSSv5)

## Part 1: Classical Search & Problem Solving

### Task 1.1: Weighted graph

| Algorithm | Returned route | Total cost |
|---|---|---:|
| Breadth-first search | S -> F -> G | 9 |
| A* search | S -> B -> D -> G | 7 |

BFS prioritises the number of edges. Because the edge costs are unequal, its shortest route in edge count is not necessarily the cheapest route.

A* prioritises f(n) = g(n) + h(n), where g(n) is the actual cost from the start and h(n) estimates the remaining cost. In this trace:

- Equal f-values are resolved alphabetically.
- The selected-node order is S, A, B, D, C, F, G.
- D is initially reached through A with g(D) = 5.
- The route through B reduces g(D) to 3, so D's cost and parent are updated.
- The final route costs 1 + 2 + 4 = 7.

An admissible heuristic never overestimates the cheapest remaining cost. At B, h(B) = 5 is admissible because the true remaining cost is 6.

### Task 1.2: Eight-puzzle

Using A* search and selecting the state with the lowest f(n) = g(n) + h(n), the puzzle is solved in **five moves: Up, Up, Left, Down, Right**. These directions refer to the movement of the blank. At the goal, g(n) = 5 and h(n) = 0, so f(n) = 5. The report shows the possible moves considered and the calculations used to select each step.

## Part 2: Applied Machine Learning with AI Copilot

### Task 2.1: Dataset setup

The notebook loads the California Housing dataset with fetch_california_housing(as_frame=True), then selects **1,000 rows** using sample(n=1000, random_state=42).

The continuous target is **MedHouseVal**, the median house value for a census block group, measured in units of **$100,000**.

| Input feature | Meaning |
|---|---|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average rooms per household |
| AveBedrms | Average bedrooms per household |
| Population | Block-group population |
| AveOccup | Average household occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |

StandardScaler is used to place the numerical features on comparable scales. This is particularly relevant to Ridge, whose coefficient penalty depends on feature scaling.

Scikit-learn's standard LinearRegression uses an ordinary least-squares solver. Scaling does not automatically improve its prediction accuracy.

### Actual notebook configuration

| Setting | Value |
|---|---|
| Sample size | 1,000 rows |
| Sampling random_state | 42 |
| Input features | 8 |
| Train/test split | 80% / 20% |
| Split random_state | 42 |
| Training subset | 800 rows |
| Reserved test subset | 200 rows |
| Cross-validation | cv=5 on the 800-row training subset |
| Scoring | r2 |
| Ridge alpha values | 0.1, 10.0, 1000.0 |

The current order is: load and sample the data, separate X and y, fit StandardScaler on X, split the scaled data, then cross-validate each estimator on X_train and y_train.

### Task 2.2: Model evaluation

The saved notebook output reports:

| Model | Alpha | Mean five-fold validation R-squared |
|---|---:|---:|
| Linear Regression | N/A: unregularised | 0.6457 |
| Ridge Regression | 0.1 | 0.6457 |
| Ridge Regression | 10.0 | 0.6443 |
| Ridge Regression | 1000.0 | 0.3574 |

Linear Regression and Ridge with alpha 0.1 tie at the displayed four-decimal precision. Ridge with alpha 10.0 is lower by 0.0014. The much lower score at alpha 1000.0 is consistent with excessive regularisation restricting the model and causing underfitting.

These are **cross-validation scores on the training subset**, not scores on the separate 200-row test subset. The notebook does not calculate a final test-set score.

Five-fold cross-validation repeatedly fits an estimator on four folds and evaluates it on the remaining fold. Averaging the five scores makes the result less dependent on one train/validation partition. A small difference between mean scores should be interpreted alongside fold-to-fold variation.

R-squared compares predictions with a mean-prediction baseline. A score of 0.6457 does not mean that 64.57% of individual predictions are correct.

### Task 2.3: Visual comparison

The notebook displays a bar chart comparing the four mean validation scores. Its bar labels use **three decimal places**, while the printed score table uses **four decimal places**.

The results illustrate the balance between overfitting and underfitting. A weak penalty may leave a model more sensitive to noise, while an excessive penalty can shrink coefficients enough to miss useful relationships. The best tested penalty should be judged using validation results rather than assuming a larger alpha is better.

## GenAI use and verification

**Gemini Enterprise Notebook** helped review manual search calculations and explain admissible heuristics, feature scaling, cross-validation and Ridge regularisation.

**ChatGPT** helped check explanations, organise report wording and prepare the presentation and repository documentation.

Examples of checking AI feedback:

| AI feedback | Decision and reason |
|---|---|
| Correct the puzzle goal calculation from 5 + 0 = 1 to 5. | Accepted: direct arithmetic confirms f(n) = 5. |
| Increase the initial misplaced-tile heuristic from 4 to 5 by including the blank. | Rejected: the required heuristic excludes the blank; only tiles 1, 2, 6 and 8 are misplaced. |

The documented examples show calculation checking and conceptual clarification. They are not presented as Python bug fixes.

## Part 3: Final Synthesis, Video & Submission

- PDF report: manual calculations, AI prompt logs, summaries and score tables, within the six-page limit.
- Public GitHub repository: completed Part 2 notebook and this README.
- Five-minute presentation: findings from both parts, a webcam overlay and an explanation of GenAI use. For a group submission, each member presents their actual contribution.

## References

- [Assignment Google Colab notebook](https://colab.research.google.com/drive/1H6rNGPJKZdstX19bc371AssdIkdUSSv5)
- [Assignment GitHub repository](https://github.com/Ambrose3112/SC5002-Lab-Assignment-1)
