# Sentinel GEC (Hack the Spring '26)

## 📌 Project Overview
Sentinel GEC is an intelligent monitoring system designed for retail billing stations. It bridges the gap between digital transaction logs (POS) and physical reality (Video Feed) to prevent "internal theft" and unauthorized access.

## 🚀 Core Features
* **Anomaly Detection:** Uses YOLOv8/Pose models to detect if a cash drawer is opened without a corresponding POS command.
* **Gesture Recognition:** Identifies suspicious hand-to-pocket movements during active transactions.
* **Temporal Sync:** Aligning millisecond-accurate POS logs with video frames for precise audit trails.
* **Blockchain Verification:** High-value alerts are hashed and stored on the **Sepolia Testnet** to prevent log tampering by store managers.
* **Digital Twin Dashboard:** A real-time schematic representation of the physical billing station status.

## 🛠️ Tech Stack
* **Backend:** Django (Python)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript
* **Database:** PostgreSQL (RDS ready)
* **Blockchain:** Solidity, Web3.py (Sepolia Testnet)
* **AI/ML:** OpenCV, Ultralytics YOLOv8
* **Cloud:** AWS (EC2, S3, RDS)

## 📁 System Architecture
1. **POS Simulator:** A simple UI for cashiers to log sales.
2. **AI Engine:** Processes video streams and detects physical state changes.
3. **Audit Dashboard:** Admin view showing live video, digital twin state, and blockchain verification badges.
