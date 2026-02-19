# Sentinel GEC - Technical Documentation

## 1. System Architecture

The system follows a modular architecture designed for low latency and high integrity.

### A. The "Sentinel" Backend (Django)
-   **Role**: Central nervous system.
-   **Responsibilities**:
    -   Receives POS transaction logs via API (`/api/pos-logs/`).
    -   Receives AI anomaly alerts via API (`/api/alerts/`).
    -   **Logic Engine**: Compares POS logs with AI alerts. If a "Drawer Open" event has no matching POS transaction within ±3 seconds, it is flagged as **UNAUTH**.
    -   **Blockchain Anchor**: Uses `Web3.py` to sign and broadcast a transaction to the AuditShield Smart Contract on Sepolia.

### B. The AuditShield Contract (Smart Contract)
-   **Network**: Sepolia Testnet
-   **Function**: `anchorEvent(string memory _eventType, string memory _evidenceHash)`
-   **Purpose**: Stores a permanent, tamper-proof record of every security incident. Even if the local database is wiped, the blockchain record remains.

### C. The Frontend (Dashboard)
-   **Admin Dashboard**:
    -   Polls `/api/alerts/` every 2 seconds.
    -   Visualizes camera feeds (mocked via Canvas/MJPEG).
    -   Displays a live "Digital Twin" of the store status.
    -   Provides direct links to Etherscan for verification.

## 2. Data Flow

### Scenario: Unauthorized Cash Drawer Opening

1.  **Physical Event**: Cashier opens the drawer without a sale.
2.  **Detection**:
    -   AI Camera detects `drawer_open` state.
    -   POS System logs `status: IDLE`.
3.  **Processing**:
    -   Backend receives `drawer_open` event.
    -   Queries DB for recent `POSLog`. Result: `None`.
    -   **Verdict**: `UNAUTHORIZED_ACCESS`.
4.  **Anchoring**:
    -   Backend generates `evidence_hash = keccak256(event_type + timestamp)`.
    -   Sends transaction to Sepolia.
5.  **Notification**:
    -   Admin Dashboard polls new data.
    -   Shows "CRITICAL ALERT" popup.
    -   "Verify" button appears with the TX Hash.

## 3. API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/pos-logs/` | Fetch recent POS transaction logs. |
| `POST` | `/api/pos-logs/` | Log a new sale (Cash/UPI). |
| `GET` | `/api/alerts/` | Fetch security alerts (includes TX Hashes). |
| `POST` | `/api/alerts/` | Trigger a new alert (used by AI Engine). |

## 4. Setup for Judges/Reviewers

If you are reviewing the code:
-   **`backend/sentinel/views.py`**: Contains the core logic for Blockchain anchoring (Lines 40-60).
-   **`mock_ai.py`**: A standalone script that simulates the Computer Vision camera inputs.
-   **`admin-dashboard.html`**: The frontend code that visualizes the data.
