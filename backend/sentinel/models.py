from django.db import models

class POSLog(models.Model):
    """Digital record from the Cashier's POS UI [cite: 13, 21]"""
    transaction_id = models.CharField(max_length=100, unique=True)
    event_type = models.CharField(max_length=50) # SALE, SHIFT_END, NO_SALE
    amount = models.FloatField(default=0.0)
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