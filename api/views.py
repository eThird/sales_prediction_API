import pandas as pd
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
from .serializers import SalesDataSerializer

# Load the trained model and scaler
lr_model = joblib.load('models/linear_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

class SalesPredictionView(APIView):
    def post(self, request):
        serializer = SalesDataSerializer(data=request.data)
        if serializer.is_valid():
            dates = serializer.validated_data['date']
            sales = serializer.validated_data['sales']

            # Convert input data to DataFrame
            data = pd.DataFrame({'date': pd.to_datetime(dates), 'sales': sales})
            data.set_index('date', inplace=True)

            # Feature engineering
            data['sales_diff'] = data['sales'].diff()
            data.dropna(inplace=True)

            # Create supervised data
            def create_supervised(data, lag=1):
                df = pd.DataFrame(data)
                columns = [df.shift(i) for i in range(1, lag+1)]
                columns.append(df)
                df = pd.concat(columns, axis=1)
                df.fillna(0, inplace=True)
                return df

            supervised_data = create_supervised(data['sales_diff'], 12)
            test_data = supervised_data[-12:]

            # Scaling features
            test_data = scaler.transform(test_data)
            x_test = test_data[:, 1:]
            y_test = test_data[:, 0]

            # Predict
            lr_predict = lr_model.predict(x_test)

            # Inverse transform to original scale
            lr_predict = scaler.inverse_transform(np.concatenate((lr_predict.reshape(-1, 1), x_test), axis=1))[:, 0]

            # Add the last actual sales value to the predictions to get the cumulative sales
            if len(data['sales']) >= 13:
                last_actual_sales = data['sales'].values[-13]
            else:
                last_actual_sales = data['sales'].values[0]

            lr_predict_cumulative = np.cumsum(np.insert(lr_predict, 0, last_actual_sales))[1:]

            # Actual sales for the last 12 months
            actual_sales = data['sales'].values[-12:]  # Adjusted to get the last 12 actual sales values

            response_data = {
                'actual_sales': actual_sales.tolist(),
                'predicted_sales': lr_predict_cumulative.tolist()
            }
            return Response(response_data, status=status.HTTP_200_OK)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
 