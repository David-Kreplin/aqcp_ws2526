# Angewandtes Quantencomputing Projektlabor (WS 2025/26)

Dieses Repository enthält **Aufgaben und Musterlösungen** für die Lehrveranstaltung  
**Angewandtes Quantencomputing Projektlabor (173593)**.

---

## Installation

### Voraussetzungen

- **Python 3.10**  oder höher
- Alle weiteren Abhängigkeiten sind in der Datei `requirements.txt` spezifiziert.

---

### Schritt-für-Schritt-Anleitung

1. **Virtuelle Umgebung erstellen (empfohlen):**
   ```bash
   python3.10 -m venv .venv
   ```

2. **Virtuelle Umgebung aktivieren:**
   - **Linux/macOS:**  
     ```bash
     source .venv/bin/activate
     ```
   - **Windows (PowerShell):**  
     ```powershell
     .venv\Scripts\Activate.ps1
     ```

3. **Benötigte Pakete installieren:**
   ```bash
   pip install -r requirements.txt
   ```

---

### Hinweise

- Alternativ kann auch eine Conda-Umgebung verwendet werden.  
  Beispiel (nicht getestet):
  ```bash
  conda create -n aqc python=3.10
  conda activate aqc
  pip install -r requirements.txt
  ```
