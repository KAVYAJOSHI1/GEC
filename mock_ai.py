import requests
import time
import sys
import random
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Configuration
BACKEND_URL = "http://127.0.0.1:8000/api/alerts/"
MOCK_STREAM_PORT = 8001

class MockStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<html><body><h1>Mock AI Video Stream</h1><p>Simulating MJPEG...</p></body></html>")
    
    def log_message(self, format, *args):
         return # Silence logs

def start_mock_stream():
    server = HTTPServer(('localhost', MOCK_STREAM_PORT), MockStreamHandler)
    print(f"📷 Mock Video Stream running at http://localhost:{MOCK_STREAM_PORT}")
    server.serve_forever()

def send_alert(alert_type):
    print(f"\n🚀 Sending {alert_type} alert to Sentinel Backend...")
    
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    
    payload = {
        "anomaly_type": alert_type,
        "confidence": round(random.uniform(0.85, 0.99), 2),
        "video_clip": f"http://localhost:{MOCK_STREAM_PORT}/evidence_{int(time.time())}.mp4",
        "is_verified": False
    }
    
    try:
        response = requests.post(BACKEND_URL, json=payload)
        if response.status_code in [200, 201]:
            print(f"✅ Success! Backend Response: {response.json()}")
        else:
            print(f"❌ Failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        print("   (Ensure Django server is running: python manage.py runserver)")

def print_menu():
    print("\n" + "="*40)
    print(" 🤖 SENTINEL GEC - MOCK AI ENGINE")
    print("="*40)
    print(" 1. Simulate 'Unauthorized Drawer Open' (UNAUTH)")
    print(" 2. Simulate 'Hand-to-Pocket' Theft (POCKET)")
    print(" 3. Exit")
    print("-" * 40)

if __name__ == "__main__":
    # Start stream in background
    stream_thread = threading.Thread(target=start_mock_stream, daemon=True)
    stream_thread.start()
    
    time.sleep(1)
    
    while True:
        print_menu()
        choice = input("Select Action: ")
        
        if choice == '1':
            send_alert('UNAUTH')
        elif choice == '2':
            send_alert('POCKET')
        elif choice == '3':
            print("Exiting...")
            break
        else:
            print("Invalid choice.")
        
        time.sleep(0.5)
