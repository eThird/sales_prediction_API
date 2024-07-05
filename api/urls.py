from django.urls import path
from . import views

urlpatterns = [
    path('sales/', views.SalesPredictionView.as_view(), name='sales_prediction'),
]
