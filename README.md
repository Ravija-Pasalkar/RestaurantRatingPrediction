# RestaurantRatingPrediction
This project involves building a machine learning model to predict a restaurant's aggregate rating based on various features such as location, cost, services, and cuisine.

## Objective
Predict the **aggregate rating** of a restaurant using machine learning techniques and deploy an interactive **Streamlit web app** to make predictions based on user input.

## ML Pipeline
### 1. Data Preprocessing
- Handled missing values
- Encoded categorical features (`LabelEncoder`, `MultiLabelBinarizer`)
- Feature engineering (`Cuisine Count`, `Log Cost`)
- Train-test split
### 2. Model Training
- Feature selection for optimal performance
- Trained 3 regression model (Linear Regression, Decision Tree Regressor, and Random Forest Regressor)
- Evaluated using:
  - R² Score
  - Root Mean Squared Error (RMSE)
- Saved the best model (Random Forest Regressor)
### 3. Deployment
- Built a **Streamlit app** to allow real-time prediction from user inputs
- Displayed predicted rating, category (e.g., "Good", "Excellent"), and color-coded visual cues
- Showed top 10 most influential features

