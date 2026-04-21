#  Robo-Advisor for Treasury Management

**Robo-Advisor for Treasury** is an advanced financial tool designed for optimizing corporate and individual cash surpluses. This project utilizes mathematical optimization to recommend the best allocation among Egyptian treasury bills, mutual funds, and certificates of deposit (CDs).

##  Key Features
* **Portfolio Optimization:** Uses `Scipy.optimize` to find the efficient frontier and maximize the Sharpe Ratio.
* **Live Treasury Yields:** Simulates and calculates real-time yields for various financial instruments.
* **Risk Management:** Allows users to set emergency cash reserves and select risk profiles (Low, Mid, High).
* **Scenario Analysis:** Visualizes how portfolio returns might react to interest rate shocks.
* **Interactive Dashboard:** A professional financial interface built with HTML5 and Chart.js.

## Tech Stack
* **Backend:** Python 3.x, Flask, NumPy.
* **Mathematical Engine:** SciPy (Optimization & Statistics).
* **Frontend:** JavaScript (ES6+), CSS3 (Modern UI), Chart.js.

##  How to Run
1. **Start the API:**
   ```bash
   python robo_advisor_local.py
The server will run on http://localhost:5001.

Launch the Dashboard:
Open RoboAdvisor_local.html in any modern web browser.

Connect:
Ensure the API URL is set correctly in the dashboard to fetch live calculations.

📊 Project Structure
robo_advisor_local.py: The backend logic for portfolio optimization and risk assessment.

RoboAdvisor_local.html: The interactive front-end advisory dashboard.
