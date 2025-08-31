# Fitkofer Budžet (Streamlit)

Mini aplikacija za kućni budžet (2 korisnika). Praćenje **troškova/priliva/transfera**, mesečnog **budžeta** (očekivano vs ostvareno) i **stanja po računima**.

## Instalacija (Windows)

```powershell
cd C:\PythonProjects\fitkofer-budget-app
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app_budget.py
