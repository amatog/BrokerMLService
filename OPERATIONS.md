# Servicebetrieb nach dem Review

Abhängigkeiten: `python -m pip install -r requirements.txt`; lokaler Start: `python run_local.py`.

- `POST /risk`: JSON mit `positions` (maximal 1000) und optional `portfolio_id`. Jede Position benötigt `symbol`, positive endliche `quantity` und `last_price`. Der vorhandene heuristische Algorithmus ist angebunden; er ist keine Inferenz des trainierten Random Forest. Datenbankbasierte Inferenz und Short-Positionen sind nicht aktiviert.
- `GET /trend`: HTTP 501, solange kein Trendmodell implementiert ist.
- `GET /health`: Status und tatsächlich verfügbare Funktionen.
- `POST /train-risk-model`: nur mit `Authorization: Bearer <ML_ADMIN_TOKEN>`. Ohne konfigurierten Token ist Training abgeschaltet. Nur lokal oder hinter einem HTTPS-Proxy betreiben.

Training erlaubt 100–10000 Portfolios, 1–126 Horizonttage, 1–10 Jahre und 3–30 unterschiedliche Ticker. Der Request ist auf 64 KiB begrenzt. Es läuft synchron, ein weiterer Job erhält HTTP 409. Die gemeinsame Datei `ML_ARTIFACT_DIR/training.lock` schützt mehrere Worker. Nach einem Prozessabsturz darf sie erst entfernt werden, wenn sicher kein Training mehr läuft. Standardverzeichnis: `artifacts`.

Trainingsfeatures verwenden ausschließlich die letzten 63 Handelstage bis zum Beobachtungsdatum. Labels stammen aus dem nachfolgenden Horizont. Trainings- und Testperioden werden chronologisch mit nicht überlappenden Labelintervallen getrennt. Zu wenig Historie führt zu einem Fehler. Die Ergebnisse sind weiterhin von Universum, Datenqualität und Marktperioden abhängig.

Modelle und Featurelisten werden als unveränderliches Paar in einem neuen `run_*`-Verzeichnis geschrieben. Erst danach ersetzt der Dienst `current.json` atomar. Konsumenten müssen diesem Manifest folgen; alte feste Dateinamen im Wurzelverzeichnis werden nicht mehr überschrieben. Alte Versionen bleiben erhalten.

Tests: `python -m unittest discover -s tests -v` (6 Tests mit synthetischen Kursen, ohne Finnhub-Zugriff).
