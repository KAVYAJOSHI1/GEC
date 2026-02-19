from rest_framework import serializers
from .models import POSLog, AnomalyAlert

class POSLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = POSLog
        fields = '__all__'

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnomalyAlert
        fields = '__all__'