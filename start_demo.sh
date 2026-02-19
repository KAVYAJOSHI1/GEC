#!/bin/bash

# Sentinel GEC - One-Click Demo Starter

echo "🚀 Starting Sentinel GEC System..."

# 1. Kill any existing server on port 8000
echo "Stopping old servers..."
fuser -k 8000/tcp > /dev/null 2>&1

# 2. Start Backend Server
echo "✅ Starting Django Backend (Port 8000)..."
./backend/venv/bin/python backend/manage.py runserver 0.0.0.0:8000 > backend.log 2>&1 &
BACKEND_PID=$!
echo "   PID: $BACKEND_PID"

# 3. Wait for Server to be ready
echo "⏳ Waiting for backend to initialize..."
sleep 5

# 4. Open Frontend in Browser
echo "✅ Opening Cashier POS and Admin Dashboard..."
xdg-open "file://$(pwd)/cashier-frontend/cashier-pos.html"
xdg-open "file://$(pwd)/admin-frontend/admin-dashboard.html"

# 5. Instructions for Mock AI
echo ""
echo "=================================================="
echo "🎉 SYSTEM IS RUNNING!"
echo "=================================================="
echo "To trigger alerts (Theft Simulation):"
echo "Run this command in a NEW TERMINAL:"
echo ""
echo "   ./backend/venv/bin/python mock_ai.py"
echo ""
echo "=================================================="
echo "Press Ctrl+C to stop the backend server"
wait $BACKEND_PID
