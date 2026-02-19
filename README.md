# Sentinel GEC (Guard. Evidence. Chain.) 🛡️🔗

**Winner of Hack the Spring '26 (Hopefully!)**

Sentinel GEC is a next-generation retail security system that bridges the gap between **Physical AI Surveillance** and **Blockchain Immutability**. It detects anomalies at the Point of Sale (POS) and anchors critical evidence to the **Sepolia Testnet** for tamper-proof auditing.

## 🚀 Key Features
- **Real-Time Anomaly Detection**: Uses Computer Vision (simulated) to detect "Unauthorized Drawer Opens" and "Hand-to-Pocket" theft events.
- **Blockchain Evidence Anchoring**: Automatically hashes and anchors alert metadata to the Ethereum Sepolia Testnet.
- **Live Admin Dashboard**: A futuristic, dark-mode UI for monitoring multiple cashier stations and verifying blockchain proofs.
- **Smart Alerts**: Intelligent toast notifications that alert security personnel only when new critical events occur.

## 🛠️ Tech Stack
- **Backend**: Django & Django REST Framework (Python)
- **Blockchain**: Web3.py & Solidity (Sepolia Testnet)
- **Frontend**: HTML5, CSS3, Vanilla JS (No heavy framework overhead)
- **AI Simulation**: Custom Python Script (`mock_ai.py`) simulating MJPEG streams and inference events.

## 📦 Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/KAVYAJOSHI1/GEC.git
    cd GEC
    ```

2.  **Setup Virtual Environment**
    ```bash
    python3 -m venv backend/venv
    source backend/venv/bin/activate
    pip install -r backend/requirements.txt
    ```

3.  **Configure Environment**
    Create a `.env` file in `backend/` with your credentials:
    ```env
    SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_KEY
    PRIVATE_KEY=0xYOUR_PRIVATE_KEY
    CONTRACT_ADDRESS=0xYOUR_CONTRACT_ADDRESS
    ```

## ⚡ Quick Start (Demo Mode)

We have included a **One-Click Script** for hackathon demos:

```bash
./start_demo.sh
```
This will:
1.  Start the Django Backend on Port 8000.
2.  Open the **Cashier POS** and **Admin Dashboard** in your browser.

## 🤖 Simulating AI Alerts

To trigger security events (since we don't have live cameras connected):

1.  Open a **new terminal**.
2.  Run the Mock AI Engine:
    ```bash
    ./backend/venv/bin/python mock_ai.py
    ```
3.  Select an option:
    -   `[1] Unauthorized Drawer Open` -> Triggers Critical Alert + Blockchain Transaction.
    -   `[2] Hand-to-Pocket` -> Triggers Warning Alert.

## 🔗 Verification via Blockchain

1.  Go to the **Admin Dashboard**.
2.  Click **"⛓ Verify on Chain"** on any Critical Alert.
3.  Click the Etherscan link to view the immutable transaction on Sepolia.

---
*Built with ❤️ by Kavya Joshi & Team*
