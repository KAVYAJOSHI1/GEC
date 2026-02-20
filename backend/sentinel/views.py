import json
from rest_framework import viewsets, status
from rest_framework.response import Response
from web3 import Web3
from decouple import config
from .models import POSLog, AnomalyAlert, WorkShift, SafetyLog
from .serializers import POSLogSerializer, AlertSerializer, WorkShiftSerializer, SafetyLogSerializer
from rest_framework.decorators import action
from django.utils import timezone

class POSLogViewSet(viewsets.ModelViewSet):
    queryset = POSLog.objects.all().order_by('-timestamp')
    serializer_class = POSLogSerializer

class WorkShiftViewSet(viewsets.ModelViewSet):
    queryset = WorkShift.objects.all().order_by('-start_time')
    serializer_class = WorkShiftSerializer

    @action(detail=False, methods=['post'])
    def start(self, request):
        # Close any existing active shifts
        WorkShift.objects.filter(is_active=True).update(is_active=False, end_time=timezone.now())
        return super().create(request)

    @action(detail=False, methods=['post'])
    def end(self, request):
        try:
            shift = WorkShift.objects.get(is_active=True)
            shift.end_time = timezone.now()
            shift.actual_cash = request.data.get('actual_cash', 0.0)
            shift.is_active = False
            
            # Calculate Expected Cash
            # Simple logic: Initial + Sum of Sales
            total_sales = 0
            logs = POSLog.objects.filter(timestamp__gte=shift.start_time, timestamp__lte=shift.end_time)
            for log in logs:
                if log.event_type == 'SALE':
                    total_sales += log.amount
                elif log.event_type == 'REFUND': # Assuming REFUND exists or negative amount
                    total_sales -= abs(log.amount)
            
            shift.expected_cash = shift.initial_cash + total_sales
            shift.discrepancy = shift.actual_cash - shift.expected_cash
            shift.save()
            
            return Response(WorkShiftSerializer(shift).data)
        except WorkShift.DoesNotExist:
            return Response({"error": "No active shift found"}, status=404)

    @action(detail=False, methods=['get'])
    def current(self, request):
        try:
            shift = WorkShift.objects.get(is_active=True)
            
            # Calculate Real-Time Expected Cash
            total_sales = 0
            logs = POSLog.objects.filter(timestamp__gte=shift.start_time)
            for log in logs:
                if log.event_type == 'SALE' and log.payment_mode == 'CASH':
                    total_sales += log.amount
                elif log.event_type == 'REFUND' and log.payment_mode == 'CASH':
                    total_sales -= abs(log.amount)
            
            current_balance = shift.initial_cash + total_sales
            
            data = WorkShiftSerializer(shift).data
            data['expected_cash'] = current_balance # Override with real-time calc
            return Response(data)
        except WorkShift.DoesNotExist:
            return Response({"active": False})

class SafetyLogViewSet(viewsets.ModelViewSet):
    queryset = SafetyLog.objects.all().order_by('-timestamp')
    serializer_class = SafetyLogSerializer

class AlertViewSet(viewsets.ModelViewSet):
    queryset = AnomalyAlert.objects.all().order_by('-timestamp')
    serializer_class = AlertSerializer

    def create(self, request, *args, **kwargs):
        # Check Safety Mode Status
        is_safe = False
        try:
            # Check if latest Safety Log for current active shift is ON
            active_shift = WorkShift.objects.filter(is_active=True).first()
            if active_shift:
                last_log = SafetyLog.objects.filter(shift=active_shift).order_by('-timestamp').first()
                if last_log and last_log.action == 'ON':
                    is_safe = True
        except Exception:
            pass

        # If Safety Mode is ON, we might tag it or suppress it
        # For now, we will add it to the request data so it saves to the model
        if is_safe:
            request.data['safety_mode_active'] = True
            # Optional: Auto-verify if safe?
            # request.data['is_verified'] = True 
        
        # 1. Save locally first
        response = super().create(request, *args, **kwargs)
        alert_data = response.data
        
        # 2. Anchor to Sepolia Testnet
        try:
            w3 = Web3(Web3.HTTPProvider(config('SEPOLIA_RPC_URL')))
            account = w3.eth.account.from_key(config('PRIVATE_KEY'))
            contract = w3.eth.contract(
                address=config('CONTRACT_ADDRESS'), 
                abi=json.loads(config('CONTRACT_ABI'))
            )

            # Create evidence fingerprint
            evidence_hash = w3.keccak(text=f"{alert_data['anomaly_type']}_{alert_data['timestamp']}").hex()

            # Build transaction
            nonce = w3.eth.get_transaction_count(account.address, 'pending')
            # Dynamic gas price to prevent underpriced errors
            base_fee = w3.eth.get_block('latest')['baseFeePerGas']
            max_priority = w3.to_wei('2', 'gwei')
            max_fee = base_fee + max_priority

            tx = contract.functions.anchorEvent(alert_data['anomaly_type'], evidence_hash).build_transaction({
                'chainId': 11155111,
                'gas': 200000,
                'maxFeePerGas': max_fee,
                'maxPriorityFeePerGas': max_priority,
                'nonce': nonce,
            })

            signed_tx = w3.eth.account.sign_transaction(tx, private_key=config('PRIVATE_KEY'))
            tx_hash = w3.to_hex(w3.eth.send_raw_transaction(signed_tx.raw_transaction))

            # 3. Update the alert with the TX Hash
            alert = AnomalyAlert.objects.get(id=alert_data['id'])
            alert.blockchain_tx = tx_hash
            alert.is_verified = True
            alert.save()
            
            print(f"✅ Alert anchored to Sepolia: {tx_hash}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Blockchain Error: {e}")

        return response