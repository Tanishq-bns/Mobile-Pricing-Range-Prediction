# Mobile Pricing Range Prediction

## Overview
This project uses machine learning to predict the price range of mobile phones based on various hardware and software features. The target variable `price_range` categorizes phones into four classes:  
- 0: Cheap  
- 1: Mid-range  
- 2: High mid-range  
- 3: Expensive  

The dataset includes features like battery power, RAM, camera specs, screen dimensions, and more. The goal is to build and evaluate classification models to accurately predict these price ranges.

## Dataset
- **Source**: `dataset.csv` (included in the project).  
- **Rows**: 2000 samples.  
- **Features**: 20 input features (e.g., `battery_power`, `ram`, `px_height`, `px_width`, `mobile_wt`, etc.).  
- **Target**: `price_range` (integer from 0 to 3).  
- **Feature Engineering**: New columns added, such as `Pixels Dimension` (px_height * px_width) and `Screen Dimension` (sc_h * sc_w).  
- **Preprocessing**: Outlier detection with Isolation Forest, feature scaling with StandardScaler, and train-test split (80/20).

For a full list of features and their descriptions, refer to the notebook.

## Requirements
- Python 3.x  
- Libraries:  
  - pandas  
  - numpy  
  - scikit-learn (for models, preprocessing, and evaluation)  
  - xgboost  
  - matplotlib  
  - seaborn  

Install dependencies:  
```
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

## Usage
1. Clone the repository or download the files.  
2. Ensure `dataset.csv` is in the same directory as the notebook.  
3. Open and run the Jupyter notebook: `Mobile_Pricing_Range_Prediction.ipynb`.  
   - The notebook handles data loading, exploration, preprocessing, model training, and evaluation.  

Example: Predict price range for new data using the trained model (see the notebook for implementation).

## Methodology
- **Exploratory Data Analysis (EDA)**: Summary statistics, visualizations (e.g., correlations, distributions).  
- **Preprocessing**: Handling outliers, feature scaling, dimensionality reduction (PCA if needed).  
- **Models Trained**:  
  - Decision Tree Classifier  
  - Random Forest Classifier  
  - K-Nearest Neighbors (KNN)  
  - Gradient Boosting Classifier  
  - Logistic Regression  
  - XGBoost Classifier  
  - Gaussian Naive Bayes  
  - Support Vector Classifier (SVC)  
  - Ensemble methods: Stacking and Bagging.  
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV.  
- **Evaluation Metrics**: Accuracy, confusion matrix, classification report.  

## Results
- **Best Model**: Random Forest Classifier.  
- **Training Accuracy**: 98.6%  
- **Test Accuracy**: 93.8%  
- Overfitting checked via learning curves.  
- The model shows strong performance but may benefit from more data or advanced tuning.

## Limitations
- Dataset is synthetic/simulated and may not reflect real-world mobile pricing perfectly.  
- No external validation dataset used.  
- Potential for further improvements (e.g., deep learning models).

## Contributing
Feel free to fork the repository, submit issues, or pull requests for improvements.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details (if applicable).
