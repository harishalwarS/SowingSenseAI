# SowingSense AI — VS Code Prototype

A Python + Streamlit prototype for the Climate Tech Hackathon proposal. It ranks candidate 7-day sowing windows using a transparent baseline suitability score, shows rainfall/temperature information, explains risk flags, and exports a CSV report.

## Requirements
- Python 3.10 or newer
- VS Code
- Internet connection only for attempting the live Open-Meteo forecast; demo mode works offline after installing dependencies.

## Run in VS Code (Windows)
1. Extract `SowingSenseAI.zip`.
2. Open VS Code.
3. Choose **File → Open Folder** and select the extracted `SowingSenseAI` folder.
4. Open **Terminal → New Terminal**.
5. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```
6. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, use Command Prompt terminal and run:
   ```bat
   .venv\Scripts\activate.bat
   ```
7. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
8. Start the app:
   ```powershell
   python -m streamlit run app.py
   ```
9. A browser tab should open, usually at `http://localhost:8501`.

## Demo
- Select a district and crop.
- Select **Live forecast (if available)** or **Demo data (offline)**.
- Click **Generate recommendation**.
- Review the primary and alternative candidate windows, score, risk explanation and chart.
- Download the CSV report.

## Important limitations
This is a hackathon prototype, not a validated farm advisory. The crop thresholds and scoring weights are simplified demonstration values. Live forecast availability varies; if retrieval fails, the app uses synthetic data and labels it clearly. Validate crop-specific thresholds and recommendations with local agricultural experts and reliable regional datasets before practical use.
