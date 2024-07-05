# api/serializers.py
from rest_framework import serializers

class SalesDataSerializer(serializers.Serializer):
    date = serializers.ListField(child=serializers.DateField())
    sales = serializers.ListField(child=serializers.IntegerField())
