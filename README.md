
# Monthly Sales Prediction

## Project Overview

This project involves predicting monthly sales using a Linear Regression model. The model is trained on past sales data, and the goal is to predict future sales based on historical trends.

## Dataset

The dataset consists of monthly sales data with the following structure:

- `date`: The date of the sales record.
- `sales`: The sales amount for the given date.

## Project Structure

- `monthly_sales.csv`: The dataset file.
- `sales_prediction.py`: The main script containing data preprocessing, model training, and evaluation.
- `linear_regression_model.pkl`: The trained Linear Regression model.

## Methodology

1. **Data Loading**: The dataset is loaded and parsed with the date as the index.
2. **Feature Engineering**: 
   - Calculating the sales difference to create a supervised learning problem.
   - Creating lag features to incorporate historical sales data.
3. **Data Splitting**:
   - Splitting the data into training and testing sets.
4. **Scaling**: Scaling the features using MinMaxScaler.
5. **Model Training**: Training a Linear Regression model on the training set.
6. **Prediction and Evaluation**:
   - Making predictions on the test set.
   - Evaluating the model performance using Mean Squared Error (MSE), Mean Absolute Error (MAE), and R2 Score.
7. **Visualization**: Plotting actual vs. predicted sales.
8. **Model Saving**: Saving the trained model using `joblib`.

## Instructions

### Prerequisites

Make sure you have the following libraries installed:

- numpy
- pandas
- scikit-learn
- matplotlib
- joblib

You can install them using:

```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

### Running the Model

1. **Load the Dataset**: Place the `monthly_sales.csv` file in the same directory as the script.
2. **Run the Script**: Execute the `sales_prediction.py` script to preprocess the data, train the model, and evaluate its performance.

```bash
python sales_prediction.py
```

### Expected Output

The script will output the following:

- The shape of the training and testing data.
- The shape of the input and output data for training and testing.
- Model evaluation metrics (MSE, MAE, R2 Score).
- A plot showing actual vs. predicted sales.
- A saved model file named `linear_regression_model.pkl`.

## Model Performance

- **Mean Squared Error (MSE)**: [Your MSE value here]
- **Mean Absolute Error (MAE)**: [Your MAE value here]
- **R2 Score**: [Your R2 Score value here]

## Future Improvements

- Explore more complex models like Decision Trees, Random Forests, or Neural Networks.
- Include additional features that might affect sales, such as promotions, holidays, economic indicators, etc.
- Periodically retrain the model to capture new trends in the data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by various machine learning and time series forecasting tutorials.
