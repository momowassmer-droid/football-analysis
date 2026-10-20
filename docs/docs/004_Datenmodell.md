# 004 – Datenmodell

## Zweck

Dieses Dokument definiert die zentralen Datenobjekte der Software.

Alle Module kommunizieren ausschließlich über diese Datenobjekte.

Dadurch bleiben die einzelnen Komponenten unabhängig voneinander und können jederzeit erweitert oder ersetzt werden.

Das Datenmodell orientiert sich an modernen Python-Standards (`dataclasses`, Type Hints) und an etablierten Datenformaten aus der Fußballanalyse (insbesondere StatsBomb).

---

# Grundprinzipien

Das Datenmodell folgt den folgenden Regeln:

- Ein Objekt repräsentiert genau einen fachlichen Sachverhalt.
- Keine verschachtelten Dictionaries als Hauptdatenstruktur.
- Alle Klassen sind typisiert.
- Datenobjekte enthalten möglichst keine Logik.
- Verarbeitung erfolgt in den jeweiligen Modulen.

---

# Zentrale Datenobjekte

Die Software basiert auf folgenden Kernobjekten.

```
Match
│
├── Frames
│
├── Tracks
│
├── Events
│
└── Metadata
```

---

# Match

Repräsentiert ein vollständiges Fußballspiel.

Enthält unter anderem:

- Match-ID
- Teams
- Datum
- Spielzeit
- Videodateien
- Spielfeldinformationen
- Metadaten

---

# Frame

Repräsentiert genau ein Videobild.

Enthält:

- Bild
- Frame-ID
- Zeitstempel
- Detektionen

Ein Frame kennt keine späteren Analyseergebnisse.

---

# Detection

Ergebnis der Objekterkennung.

Beispiele:

- Spieler
- Ball
- Schiedsrichter

Enthält:

- Klasse
- Bounding Box
- Konfidenz
- Frame-ID

---

# Track

Beschreibt ein über mehrere Frames verfolgtes Objekt.

Enthält:

- Track-ID
- Objektklasse
- Position
- Geschwindigkeit (optional)
- Historie

---

# Player

Repräsentiert einen identifizierten Spieler.

Enthält beispielsweise:

- Track-ID
- Team
- Trikotfarbe
- Rückennummer (optional)
- Positionen im Spiel

---

# Ball

Repräsentiert den Ball.

Enthält:

- Position
- Geschwindigkeit
- Track-ID

---

# Event

Beschreibt ein Fußballereignis.

Beispiele:

- PASS
- SHOT
- DRIBBLE
- DUEL
- INTERCEPTION
- FOUL

Ein Event besitzt zusätzlich strukturierte Attribute.

Beispiele:

- Spieler
- Startposition
- Endposition
- Zeit
- Erfolg

---

# Team

Repräsentiert eine Mannschaft.

Enthält:

- Name
- Team-ID
- Trikotfarbe
- Spieler

---

# Metadata

Beschreibt allgemeine Informationen zum Spiel.

Beispiele:

- Kamera
- FPS
- Auflösung
- Spielfeldgröße
- Wetter (optional)
- Liga (optional)

---

# Koordinatensystem

Die Software verwendet intern ausschließlich Spielfeldkoordinaten.

Standard:

StatsBomb

Spielfeldgröße

120 × 80

Pixelkoordinaten werden ausschließlich innerhalb der Bildverarbeitung verwendet.

---

# Zeitmodell

Alle zeitabhängigen Objekte besitzen mindestens:

- Frame-ID
- Timestamp

Dadurch können Video, Tracking und Events jederzeit synchronisiert werden.

---

# Objektbeziehungen

```
Match

├── Metadata

├── Team

│      └── Player

├── Frame

│      └── Detection

├── Track

└── Event
```

---

# Technische Umsetzung

Alle Datenobjekte werden als Python-Dataclasses implementiert.

Beispiel:

- Match
- Frame
- Detection
- Track
- Player
- Ball
- Event

Durch Type Hints und Dataclasses entsteht eine klar definierte Schnittstelle zwischen sämtlichen Modulen.

---

# Ziel

Das Datenmodell bildet die gemeinsame Sprache aller Komponenten.

Kein Modul kommuniziert über lose Dictionaries oder nicht dokumentierte Datenstrukturen.

Dadurch bleibt die Software nachvollziehbar, testbar und langfristig wartbar.