from rest_framework import serializers
from .models import POSLog, AnomalyAlert, WorkShift, SafetyLog

class POSLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = POSLog
        fields = '__all__'

class WorkShiftSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkShift
        fields = '__all__'

class SafetyLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = SafetyLog
        fields = '__all__'

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnomalyAlert
        fields = '__all__'