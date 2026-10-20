# 003 – Projektarchitektur

## Zweck

Dieses Dokument beschreibt die technische Architektur des Projekts und definiert die Verantwortlichkeiten der einzelnen Module.

Die Architektur orientiert sich an bewährten Prinzipien der Softwareentwicklung und Computer Vision. Ziel ist eine modulare, erweiterbare und langfristig wartbare Software zur automatisierten Analyse von Fußballspielen.

---

# Architekturprinzipien

Für das gesamte Projekt gelten folgende Grundsätze:

- Jedes Modul besitzt genau eine klar definierte Aufgabe (Single Responsibility Principle).
- Module kommunizieren ausschließlich über definierte Datenstrukturen.
- Module kennen keine internen Implementierungsdetails anderer Module.
- Komponenten sollen jederzeit austauschbar sein.
- Die Software soll unabhängig von der verwendeten Videoquelle arbeiten.

---

# Gesamtübersicht

```
                 Fußballvideo
                       │
                       ▼
                Video Reader
                       │
                       ▼
               Frame Generator
                       │
                       ▼
            Objekterkennung (YOLO)
                       │
                       ▼
           Multi Object Tracking
                       │
                       ▼
      Spielfeldkalibrierung (Homographie)
                       │
                       ▼
          Transformation in Spielfeld-
               koordinaten (120 × 80)
                       │
                       ▼
             Ereigniserkennung
                       │
                       ▼
            Statistische Auswertung
                       │
                       ▼
          StatsBomb-kompatibler Export
```

---

# Module

## Video

Verantwortung

- Videos laden
- Videos öffnen
- Frames lesen
- Zeitstempel erzeugen
- Metadaten auslesen
- Mehrere Videoclips als ein Spiel behandeln

Input

- MP4-Dateien

Output

- Frame
- Frame-ID
- Timestamp

---

## Detection

Verantwortung

- Spieler erkennen
- Ball erkennen
- Schiedsrichter erkennen

Geplante Bibliothek

- Ultralytics YOLO

Output

- Detection-Objekte

---

## Tracking

Verantwortung

- Track-IDs vergeben
- Spieler über mehrere Frames verfolgen
- Ball verfolgen

Geplante Bibliothek

- ByteTrack (oder gleichwertiger Standard)

Output

- Track-Objekte

---

## Calibration

Verantwortung

- Spielfeld erkennen
- Homographie berechnen
- Pixelkoordinaten in Spielfeldkoordinaten umrechnen

Output

- Positionen im StatsBomb-Koordinatensystem

---

## Event Detection

Verantwortung

- Pässe erkennen
- Schüsse erkennen
- Dribblings erkennen
- Zweikämpfe erkennen
- Ballgewinne erkennen
- weitere Ereignisse

Output

- standardisierte Eventobjekte

---

## Analytics

Verantwortung

Berechnung sämtlicher Statistiken.

Beispiele

- Laufstrecken
- Heatmaps
- Ballbesitz
- Passnetzwerke
- Pressing
- xG (zukünftig)

---

## Export

Verantwortung

Export der Analyseergebnisse.

Geplante Formate

- StatsBomb JSON
- CSV
- Parquet (optional)

---

# Datenfluss

Die Verarbeitung erfolgt ausschließlich in Vorwärtsrichtung.

Video

↓

Frames

↓

Detections

↓

Tracks

↓

Spielfeldkoordinaten

↓

Events

↓

Statistiken

↓

Export

Ein Modul greift niemals direkt auf spätere Verarbeitungsschritte zu.

---

# Entwurf der Projektstruktur

```
football-analysis/

config/
data/
docs/
models/
notebooks/
src/
    video/
    detection/
    tracking/
    calibration/
    events/
    analytics/
    export/
    visualization/
    utils/
tests/
```

---

# Erweiterbarkeit

Die Architektur ist so aufgebaut, dass einzelne Komponenten später problemlos ersetzt werden können.

Beispiele:

- Austausch des YOLO-Modells
- Austausch des Tracking-Algorithmus
- Unterstützung weiterer Kamerasysteme
- zusätzliche Exportformate
- neue Eventtypen
- neue Statistikmodule

Dadurch bleibt die Software langfristig wartbar und flexibel.

---

# Architekturziel

Die Software soll kein Sammlung einzelner Skripte sein.

Sie soll als modulare Analyseplattform entwickelt werden, deren Komponenten unabhängig voneinander getestet, erweitert und ausgetauscht werden können.

Die gesamte Architektur orientiert sich an etablierten Standards der Python-Softwareentwicklung, Computer Vision und Fußballanalyse.