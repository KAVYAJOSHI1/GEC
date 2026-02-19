import json
from rest_framework import viewsets, status
from rest_framework.response import Response
from web3 import Web3
from decouple import config
from .models import POSLog, AnomalyAlert
from .serializers import POSLogSerializer, AlertSerializer

class POSLogViewSet(viewsets.ModelViewSet):
    queryset = POSLog.objects.all().order_by('-timestamp')
    serializer_class = POSLogSerializer

class AlertViewSet(viewsets.ModelViewSet):
    queryset = AnomalyAlert.objects.all().order_by('-timestamp')
    serializer_class = AlertSerializer

    def create(self, request, *args, **kwargs):
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