# 001 – Projektvision und Entwicklungsrichtlinien

## Projektziel

Ziel dieses Projekts ist die Entwicklung einer modularen Software zur automatisierten Analyse von Fußballspielen anhand von Videodaten.

Die Software soll unabhängig von der Videoquelle arbeiten und aus beliebigen Fußballvideos strukturierte Tracking-, Event- und Statistikdaten erzeugen.

Primäres Ziel ist die Erzeugung von Daten, die möglichst kompatibel zu etablierten Fußball-Datenanbietern (insbesondere StatsBomb) sind und dadurch mit bestehenden Analysewerkzeugen weiterverarbeitet werden können.

---

# Grundprinzipien

Das Projekt orientiert sich an professionellen Softwareentwicklungs- und Data-Science-Standards.

Dabei gelten folgende Grundsätze:

1. Standards vor Eigenlösungen
2. Modulare Architektur
3. Nach jedem Entwicklungsschritt lauffähige Software
4. Vollständige Dokumentation wichtiger Entscheidungen
5. Wiederverwendbare Komponenten
6. Erweiterbarkeit für zukünftige Entwicklungen

---

# Langfristige Vision

Die Software soll später in der Lage sein, aus einem Fußballvideo automatisch

- Spieler zu erkennen
- den Ball zu erkennen
- Spieler und Ball über das gesamte Spiel zu verfolgen
- Positionen in Spielfeldkoordinaten umzurechnen
- Fußballereignisse zu erkennen
- Statistiken zu berechnen
- Daten im StatsBomb-Format zu exportieren

Dabei soll die Analyse unabhängig von der verwendeten Kamera funktionieren.

Unterstützt werden sollen langfristig beispielsweise

- Veo
- Handyaufnahmen
- Stativkameras
- Hudl
- Spiideo
- weitere MP4-Quellen

Die Videoquelle ist lediglich der Eingang der Verarbeitungskette und soll keinen Einfluss auf die eigentliche Analyse haben.

---

# Entwicklungsphilosophie

Das Projekt ist kein reines KI-Projekt.

Die künstliche Intelligenz ist lediglich ein Bestandteil einer vollständigen Softwarearchitektur.

Der geplante Datenfluss lautet:

Video

↓

Video Reader

↓

Objekterkennung (YOLO)

↓

Multi Object Tracking

↓

Spielfeldkalibrierung (Homography)

↓

Transformation in Spielfeldkoordinaten

↓

Ereigniserkennung

↓

Statistische Auswertung

↓

StatsBomb-kompatibler Export

Alle Module besitzen klar definierte Ein- und Ausgaben.

Die Module sollen möglichst lose gekoppelt sein, sodass einzelne Komponenten später ersetzt oder verbessert werden können.

---

# Datenstandard

Für sämtliche internen Berechnungen werden etablierte Standards verwendet.

## Spielfeld

Koordinatensystem:

StatsBomb

Größe:

120 × 80

Pixelkoordinaten werden ausschließlich während der Bildverarbeitung verwendet.

Nach der Kalibrierung werden sämtliche Positionen in Spielfeldkoordinaten gespeichert.

---

## Zeit

Alle Daten besitzen

- Frame-Nummer
- Zeitstempel

Dadurch können Video-, Tracking- und Eventdaten jederzeit synchronisiert werden.

---

## Tracking

Jeder Spieler erhält eine eindeutige Track-ID.

Diese ID soll möglichst während des gesamten Spiels erhalten bleiben.

---

## Events

Fußballereignisse werden nicht als Freitext gespeichert.

Stattdessen erhält jedes Event einen standardisierten Typ.

Beispiele:

- PASS
- SHOT
- TACKLE
- DUEL
- DRIBBLE
- INTERCEPTION

Zusätzliche Eigenschaften (Startposition, Endposition, Spieler, Erfolg usw.) werden als strukturierte Attribute gespeichert.

---

# Verwendete Standards

Wann immer möglich werden etablierte Open-Source-Lösungen verwendet.

Geplant sind insbesondere

- Python
- OpenCV
- Ultralytics YOLO
- ByteTrack
- NumPy
- Pandas
- Matplotlib
- mplsoccer
- StatsBomb-Datenmodell
- Git

Eigene Implementierungen sollen nur entstehen, wenn keine geeignete etablierte Lösung existiert.

---

# Projektstruktur

Das Projekt wird modular aufgebaut.
