from django.db import models

class POSLog(models.Model):
    """Digital record from the Cashier's POS UI [cite: 13, 21]"""
    transaction_id = models.CharField(max_length=100, unique=True)
    event_type = models.CharField(max_length=50) # SALE, SHIFT_END, NO_SALE
    amount = models.FloatField(default=0.0)
    payment_mode = models.CharField(max_length=20, default='CASH') # CASH, UPI, CARD
    timestamp = models.DateTimeField(auto_now_add=True)

class AnomalyAlert(models.Model):
    """The Flag from the AI Engine [cite: 14, 15]"""
    type_choices = [('UNAUTH', 'Unauthorized Access'), ('POCKET', 'Hand-to-Pocket')]
    anomaly_type = models.CharField(max_length=10, choices=type_choices)
    confidence = models.FloatField()
    video_clip = models.CharField(max_length=500, blank=True, null=True) # Stored in S3/Local or URL
    timestamp = models.DateTimeField(auto_now_add=True)
    is_verified = models.BooleanField(default=False)
    blockchain_tx = models.CharField(max_length=100, blank=True)
    
    # New fields for Rules
    rule_violated = models.CharField(max_length=100, blank=True, null=True) # e.g. "Drawer Open > 10s"
    safety_mode_active = models.BooleanField(default=False)

class WorkShift(models.Model):
    cashier_name = models.CharField(max_length=100)
    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    initial_cash = models.FloatField(default=0.0)
    expected_cash = models.FloatField(default=0.0)
    actual_cash = models.FloatField(default=0.0) # User Input
    discrepancy = models.FloatField(default=0.0)
    is_active = models.BooleanField(default=True)

class SafetyLog(models.Model):
    shift = models.ForeignKey(WorkShift, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    action = models.CharField(max_length=10) # ON / OFF
    linked_transaction_id = models.CharField(max_length=100, blank=True, null=True)