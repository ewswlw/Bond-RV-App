# Portfolio123 Machine Learning and AI Capabilities Guide

## When to Use

- Use this guide when evaluating Portfolio123's AI Factor capabilities, selecting algorithms, or planning ML workflows within the platform.
- Apply it before configuring new AI Factors so you understand feature requirements, preprocessing options, and deployment steps.
- Reference it during model reviews to compare algorithm strengths, limitations, and tuning considerations specific to Portfolio123.
- Consult it when mentoring team members; it summarizes end-to-end workflow and supported tools without requiring code.
- If you are coding against the API instead, pair this with the automation guide; otherwise treat it as the authoritative resource for no-code ML features.

## Overview

Portfolio123's AI Factor feature allows users to create machine learning models for stock prediction and ranking without writing code. The system automates the entire ML pipeline from data preprocessing to model training and deployment.

## AI Factor Workflow

The AI Factor creation process follows these steps:

1. **Define Universe**: Select which stocks to train on
2. **Select Features**: Choose factors/variables for prediction
3. **Configure Preprocessing**: Normalize and clean data
4. **Choose Algorithm**: Select ML algorithm and hyperparameters
5. **Train Model**: System trains model on historical data
6. **Validate Performance**: Review backtest results
7. **Deploy**: Use AI Factor in ranking systems and strategies

## Supported Machine Learning Algorithms

### 1. XGBoost (Extreme Gradient Boosting)

**Type**: Non-parametric gradient boosting

**Description**: Forms a strong predictor by training a sequence of weak predictors, each improving on previous results.

**Strengths**:
- Excellent performance on tabular data
- Handles non-linear relationships well
- Built-in feature importance
- Robust to overfitting with proper tuning

**Limitations**:
- Memory intensive for extremely large datasets
- Requires careful hyperparameter tuning

**Best For**: General-purpose stock prediction, complex feature interactions

**Documentation**: https://xgboost.readthedocs.io/en/latest/parameter.html

---

### 2. LightGBM

**Type**: Non-parametric gradient boosting

**Description**: Gradient boosting framework using tree-based learning algorithms, optimized for efficiency.

**Strengths**:
- Faster training speed than XGBoost
- Lower memory usage
- Better accuracy on large datasets
- Supports parallel, distributed, and GPU learning
- Capable of handling large-scale data

**Limitations**:
- Can overfit on small datasets
- Requires tuning for optimal performance

**Best For**: Large universes, high-frequency rebalancing, production systems

**Documentation**: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRegressor.html

---

### 3. Random Forest

**Type**: Non-parametric ensemble

**Description**: Ensemble algorithm averaging outputs of multiple decision trees.

**Strengths**:
- Robust to overfitting
- Handles missing data well
- Provides feature importance
- No assumptions about data distribution

**Limitations**:
- Memory-intensive with many trees or features
- Slower than gradient boosting methods
- Can be biased toward features with more categories

**Best For**: Robust baseline models, feature selection, interpretability

**Documentation**: https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html

---

### 4. Extra Trees (Extremely Randomized Trees)

**Type**: Non-parametric ensemble

**Description**: Predicts numerical values using ensemble of decision trees with extra randomness in splits.

**Strengths**:
- More robust to overfitting than Random Forest
- Faster training than Random Forest
- Good generalization
- No assumptions about data distribution

**Limitations**:
- Similar memory requirements to Random Forest
- May underfit compared to gradient boosting

**Best For**: Quick prototyping, reducing overfitting, feature selection

**Documentation**: https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html

---

### 5. Linear Regression

**Type**: Parametric statistical model

**Description**: Fits linear relationship between target and features.

**Strengths**:
- Fast training and prediction
- Highly interpretable
- Low memory requirements
- Scales well to large datasets

**Limitations**:
- Assumes linear relationships
- Cannot capture complex interactions
- Sensitive to outliers

**Best For**: Simple factor combinations, baseline models, Z-Score preprocessor

**Recommended Preprocessor**: Z-Score normalization

**Documentation**: https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html

---

### 6. Keras Neural Networks

**Type**: Non-parametric deep learning

**Description**: Multi-layer neural networks that learn non-linear transformations.

**Strengths**:
- Can learn extremely complex patterns
- Flexible architecture
- Handles high-dimensional data
- No assumptions about data distribution

**Limitations**:
- Requires large datasets
- Prone to overfitting without regularization
- Longer training time
- Requires careful architecture design

**Best For**: Large datasets, complex non-linear relationships, deep feature learning

**Documentation**: https://keras.io/about/

---

### 7. Support Vector Machines (SVM)

**Type**: Primarily parametric, can handle non-parametric data

**Description**: Finds optimal hyperplane for regression/classification.

**Strengths**:
- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Versatile kernel functions

**Limitations**:
- **Does not scale well** with large datasets
- Requires feature scaling
- Sensitive to hyperparameter choices
- Slow training on large datasets

**Best For**: Small to medium datasets, high-dimensional feature spaces

**Preprocessing Required**: Feature scaling recommended

**Documentation**: https://scikit-learn.org/1.5/modules/generated/sklearn.svm.SVR.html

---

### 8. Generalized Additive Models (GAM)

**Type**: Flexible parametric/non-parametric hybrid

**Description**: Models that accommodate both parametric and non-parametric relationships.

**Strengths**:
- Handles non-linear relationships
- Interpretable components
- Flexible for various data types

**Limitations**:
- **Does not scale well** with large datasets
- Computationally expensive
- Requires careful smoothing parameter selection

**Best For**: Small to medium datasets, interpretability, non-linear relationships

**Documentation**: https://www.statsmodels.org/stable/gam.html

---

### 9. DeepTables

**Type**: Non-parametric deep learning for tabular data

**Description**: Automated deep learning tool specifically designed for tabular data.

**Strengths**:
- Automates preprocessing and feature engineering
- Handles both parametric and non-parametric variables
- Automated model selection
- Automated hyperparameter tuning
- Ensemble learning built-in

**Limitations**:
- Requires substantial computational resources
- May be overkill for simple patterns

**Best For**: Complex tabular data, automated ML pipelines, large feature sets

**Documentation**: https://deeptables.readthedocs.io/en/latest/

---

## Scalability Guide

### Algorithms That Scale Well
✅ **LightGBM** - Best for very large datasets  
✅ **XGBoost** - Good for large datasets  
✅ **Random Forest** - Scales reasonably well  
✅ **Extra Trees** - Scales reasonably well  
✅ **Linear Regression** - Excellent scalability  
✅ **Keras Neural Networks** - Scales well with GPU  
✅ **DeepTables** - Scales well with proper resources  

### Algorithms That Do NOT Scale Well
❌ **Support Vector Machines (SVM)** - Struggles with millions of rows  
❌ **Generalized Additive Models (GAM)** - Struggles with millions of rows  

### Optimization Strategies for Large Datasets

When using algorithms that don't scale well, reduce computational load by:

1. **Use smaller training universe** - Reduce number of stocks
2. **Shorten dataset period** - Use fewer historical years
3. **Lengthen dataset frequency** - Use monthly instead of weekly
4. **Reduce number of features** - Select most important factors
5. **Switch to scalable algorithm** - Use LightGBM or XGBoost instead

## Data Preprocessing Options

### Scaling Methods

#### 1. MinMax Scaling
- **Range**: Scales features to [0, 1] range
- **Formula**: (X - min) / (max - min)
- **Best For**: Neural networks, algorithms sensitive to scale
- **Use Case**: When you want bounded values

#### 2. Rank Scaling
- **Range**: Converts to percentile ranks [0, 100]
- **Formula**: Percentile rank of value
- **Best For**: Handling outliers, non-normal distributions
- **Use Case**: Stock ranking, reducing outlier impact

#### 3. Normal (Z-Score) Scaling
- **Range**: Mean=0, StdDev=1
- **Formula**: (X - mean) / std
- **Best For**: Linear regression, algorithms assuming normality
- **Use Case**: When features should have similar variance

### Preprocessing Scope

#### Dataset Scope
- Scaling parameters computed across entire training period
- Prevents look-ahead bias
- **Parameter**: `mlTrainingEnd` - End date for computing scaling parameters
- **Best For**: Production models, avoiding data leakage

#### Date Scope
- Scaling parameters computed separately for each date
- More adaptive to changing market conditions
- **Best For**: Capturing time-varying relationships

### Outlier Handling

**Outlier Clipping**: When enabled, extreme values are clipped to reduce impact

**Parameters**:
- `outliers: True` - Enable outlier clipping
- `outlierLimit: 5` - Number of standard deviations for clipping (used with normal scaling)
- `trimPct: 5.0` - Percentage to trim from each end

### Missing Value Handling

**NA Fill**: When enabled, missing values are replaced with middle values

**Parameter**: `naFill: False` or `True`

**Best Practice**: 
- Set to `True` for algorithms that don't handle NAs (e.g., Linear Regression)
- Set to `False` for tree-based algorithms that handle NAs natively

### Excluded Formulas

**Purpose**: Exclude certain factors from preprocessing (typically technical indicators)

**Parameter**: `excludedFormulas: ["Close(0)/close(5)"]`

**Use Case**: 
- Technical factors already normalized
- Price ratios don't need scaling
- **Note**: Data license required for non-technical factors

## Feature Selection

### Choosing Features

Portfolio123 allows selection from 1,000+ factors including:

- **Fundamental Data**: P/E, P/B, ROE, debt ratios, margins
- **Growth Metrics**: Revenue growth, earnings growth, cash flow growth
- **Technical Indicators**: Price momentum, volume, moving averages
- **Analyst Data**: Estimates, revisions, consensus ratings
- **Macro Factors**: Interest rates, economic indicators
- **Custom Formulas**: User-defined combinations

### Best Practices

1. **Start with 10-30 features** - Avoid overfitting
2. **Use domain knowledge** - Include known predictive factors
3. **Mix factor types** - Combine value, growth, momentum, quality
4. **Avoid highly correlated features** - Reduces redundancy
5. **Use feature importance** - Review which features matter most
6. **Iterate and refine** - Add/remove based on performance

### Feature Engineering

Create custom formulas combining base factors:
```
// Example custom features
PEExclXorTTM / Industry(PEExclXorTTM)  // Relative P/E
(ROE - Industry(ROE)) / StdDev(ROE,20)  // ROE Z-score vs industry
SalesGr%TTM * GrMarginTTM               // Quality-adjusted growth
```

## Training Configuration

### Universe Selection

**Training Universe**: Stocks used to train the model

**Options**:
- Standard universes (SP500, Russell2000, etc.)
- Custom universes
- API Universe (dynamic, updatable via API)

**Best Practice**: Train on universe similar to deployment universe

### Time Period

**Training Period**: Historical date range for training

**Considerations**:
- **Longer periods**: More data, captures different market regimes
- **Shorter periods**: More recent patterns, less computational load
- **Typical range**: 5-15 years

### Frequency

**Data Frequency**: How often to sample data

**Options**:
- Weekly (most common)
- Every N weeks (2, 3, 4, 6, 8, 13, 26, 52)
- Monthly
- Quarterly

**Trade-off**:
- Higher frequency = More data points, more computation
- Lower frequency = Less overfitting, faster training

### PIT Method (Point-in-Time)

**Prelim**: Uses preliminary financial data (available sooner)  
**Complete**: Uses only complete/restated financial data (more accurate)

**Default**: Prelim (as of recent update)

**Best Practice**: Use Complete for final models to avoid restatement bias

## Model Validation

### Backtest Performance

After training, Portfolio123 provides:

1. **Rank Performance Test**: Bucketized backtest showing performance by rank decile
2. **Factor Value Chart**: Distribution of AI Factor scores
3. **Correlation Analysis**: How AI Factor correlates with returns
4. **Feature Importance**: Which features contribute most

### Evaluation Metrics

Review these metrics to assess model quality:

- **Sharpe Ratio**: Risk-adjusted returns
- **Information Coefficient (IC)**: Correlation between predictions and returns
- **Turnover**: How often rankings change
- **Decile Spread**: Performance difference between top and bottom deciles

### Avoiding Overfitting

1. **Use holdout period** - Test on recent out-of-sample data
2. **Limit feature count** - Fewer features = less overfitting
3. **Cross-validation** - Train on different time periods
4. **Regularization** - Use algorithm regularization parameters
5. **Ensemble methods** - Combine multiple models

## Deployment and Usage

### Using AI Factors

Once trained, AI Factors can be used like any other factor:

**In Ranking Systems**:
```
AIFactor("MyMLModel")  // Returns ML prediction score
```

**In Screens**:
```
AIFactor("MyMLModel") > 50  // Filter by AI score
```

**In Formulas**:
```
AIFactor("MyMLModel") * PEExclXorTTM  // Combine with other factors
```

### Retraining

**When to Retrain**:
- Market regime changes
- Model performance degrades
- New data becomes available
- Adding/removing features

**Frequency**: Typically every 6-12 months, or as needed

## API Access to AI Factors

### Training via API

Use the `p123api` Python package to train AI Factors programmatically:

```python
from p123api import Client

client = Client(api_id='YOUR_ID', api_key='YOUR_KEY')

# AI Factor training endpoint available
# See API documentation for parameters
```

### Retrieving AI Factor Values

AI Factor scores can be retrieved via:
- `/data_universe` endpoint
- `/rank/ranks` endpoint
- Screen reports
- Factor downloads

## Advanced Techniques

### Ensemble Models

Combine multiple AI Factors for better performance:

```
(AIFactor("Model1") + AIFactor("Model2") + AIFactor("Model3")) / 3
```

### Sector-Specific Models

Train separate models for different sectors:
- Technology stocks
- Financial stocks
- Energy stocks
- Healthcare stocks

### Multi-Timeframe Models

Combine predictions from different rebalancing frequencies:
- Weekly momentum model
- Monthly mean reversion model
- Quarterly fundamental model

### Target Variable Engineering

Instead of predicting raw returns, predict:
- Risk-adjusted returns (Sharpe)
- Relative returns vs. sector
- Probability of outperformance
- Drawdown risk

## Common Pitfalls

### 1. Look-Ahead Bias
❌ **Problem**: Using future data in training  
✅ **Solution**: Use point-in-time data, set `mlTrainingEnd` properly

### 2. Survivorship Bias
❌ **Problem**: Training only on currently listed stocks  
✅ **Solution**: Include delisted stocks in training universe

### 3. Overfitting
❌ **Problem**: Model memorizes training data  
✅ **Solution**: Limit features, use regularization, validate out-of-sample

### 4. Data Snooping
❌ **Problem**: Testing multiple models on same data  
✅ **Solution**: Use separate validation period, avoid excessive iteration

### 5. Regime Changes
❌ **Problem**: Model trained in one market regime fails in another  
✅ **Solution**: Train on multiple regimes, monitor performance, retrain regularly

## Best Practices Summary

1. **Start Simple**: Begin with Linear Regression or XGBoost
2. **Use Appropriate Scaling**: Match scaling method to algorithm
3. **Validate Thoroughly**: Always test out-of-sample
4. **Monitor Performance**: Track live performance vs. backtest
5. **Retrain Regularly**: Update models as market conditions change
6. **Combine with Rules**: Use ML alongside traditional factor models
7. **Document Everything**: Track model versions, features, parameters
8. **Avoid Overfitting**: Prefer simpler models that generalize well

## Resources

- **AI Factor User Guide**: https://portfolio123.customerly.help/en/articles/34576-ai-factors-user-guide
- **AI Algorithms Documentation**: https://portfolio123.customerly.help/en/articles/33967-ai-algorithms
- **API Documentation**: https://portfolio123.customerly.help/en/articles/13765-the-api-wrapper-p123api
- **Community Forum**: https://community.portfolio123.com/

## Example Workflow

### Step-by-Step: Creating Your First AI Factor

1. **Navigate to Research → AI Factors**
2. **Click "Create New AI Factor"**
3. **Select Universe**: e.g., "Russell 2000"
4. **Choose Features**: Select 15-20 factors (value, growth, momentum mix)
5. **Configure Preprocessing**:
   - Scaling: Rank
   - Scope: Dataset
   - Trim: 5%
   - Outliers: True
   - NA Fill: False
6. **Select Algorithm**: XGBoost (good default)
7. **Set Training Period**: 2010-2023
8. **Set Frequency**: Weekly
9. **Click "Train Model"**
10. **Review Performance**: Check rank performance test
11. **Save and Deploy**: Use in ranking system

---

**Last Updated**: October 2025  
**Version**: 1.0

