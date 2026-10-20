# ARCHITEKTUR — football-analysis

> **Lebendiges Dokument.** Immer aktuell halten. Es beschreibt jedes Programm/Modul
> und was es tut, den Datenfluss und wie man alles startet.
> Für Prioritäten/Roadmap siehe [PLAN.md](PLAN.md). Letzte Aktualisierung: 2026-07-11

---

## 0. Kurz gesagt

Das System macht aus **Veo-Kameramaterial** (drei feststehende Kameraansichten
desselben Spiels: Mitte/Links/Rechts) **Tracking-Daten im StatsBomb-Format**:
jeder Spieler in Feldkoordinaten (120×80), pro Frame, mit möglichst stabiler ID,
plus daraus abgeleitete Auswertungen (Laufwege, Ballbesitz, Events). Läuft
CPU-only auf einem Laptop; GPU ist geplant.

Der Weg eines Bildes bis zur Zahl:

```
Video ──► Detektion (Boxen) ──► Tracking (IDs) ──► Kalibrierung (Feldkoords)
      ──► Bereinigung (Dedup/Filter/Stitch/Glättung) ──► Teams ──► Analytik/Export
      ──► (mehrere Kameras) Zusammenbau zu globalen IDs
```

---

## 1. Zwei Pipelines — wichtig zu verstehen

Historisch gibt es **zwei** Ketten. Das erklärt, warum manche Programme doppelt
zu existieren scheinen.

### A) Produktions-Pipeline (aktuell, „x-Modell")
Das ist die Kette, an der wir arbeiten. Detektion offline mit dem starken
Fußballmodell, dann alles Weitere auf diesen Boxen.

- Orchestriert von **`tools/xmodel_pipeline.py`**.
- Detektionsquelle: **`tools/autolabel.py`** (läuft das x-Modell offline über die
  Clips, schreibt YOLO-Labels).
- Tracker: **`src/tracking/bytetrack_boxes.py`** (ByteTrack auf fertigen Boxen).
- Ausgaben: `data/output/<clip>/detections/x_*.csv`.

### B) Legacy-Pipeline (nano, „workflow")
Die ältere, **vollständige** Ende-zu-Ende-Kette bis StatsBomb — aber vom
schwächeren nano-Detektor gespeist (trackt Zuschauer, fragmentiert ferne Spieler).

- Orchestriert von **`src/workflow.py`** / **`src/main.py`** / **`src/match.py`**.
- Enthält die **Analytik/Export-Endstücke** (Ballbesitz, Events, StatsBomb,
  physische Metriken, Roster, Video-Render), die die x-Modell-Kette (A) noch
  **nicht** erreicht.

> **Die zentrale Lücke** (= PLAN.md Nr. 2): die guten x-Modell-Tracks (A) sind noch
> nicht an die Analytik/Export-Endstücke (B) angedockt. Beide sind CSV-getrieben,
> also andockbar.

---

## 2. Datenfluss der Produktions-Pipeline (Schritt für Schritt)

Alle Schritte für einen Clip, wie `tools/xmodel_pipeline.rebuild_clip` sie fährt:

| # | Schritt | Modul / Funktion | Ein → Aus |
|---|---------|------------------|-----------|
| 1 | Detektion (offline, über Nacht) | `tools/autolabel.py` + `models/football_x_ref.pt` | Video → `data/autolabel/<clip>/labels/*.txt` |
| 2 | Labels laden | `xmodel_pipeline.load_label_boxes` | Labels → Boxen pro Frame |
| 3 | Duplikate entfernen (Kopf/Fuß-Fragmente, Doppelklassen) | `src/detection/dedup.py` | Boxen → bereinigte Boxen |
| 4 | Tracking (ID-Vergabe, im stabilisierten Rahmen) | `src/tracking/bytetrack_boxes.py` (+ `src/motion/camera_motion.py`) | Boxen → Tracks (`track_id`) |
| 5 | Ball/Artefakte raus (als „Spieler" gelabelter Ball) | `xmodel_pipeline._drop_ball_shaped` | Tracks → Tracks ohne Ball |
| 6 | Feldkoordinaten + Off-Pitch-Filter | `src/calibration/homography.filter_to_pitch_csv` (+ camera_motion) | Tracks → `x_pitch.csv` |
| 7 | Fragmente zusammenführen | `src/tracking/enhanced_stitcher.py` (+ `src/tracking/appearance.py`) | → gestitcht |
| 8 | Glätten (Ausreißer, Lücken) | `src/tracking/smoothing.py` | → `x_tracks.csv` |
| 9 | Teamzuordnung (Trikotfarbe) | `src/analytics/team_classifier.py` | → `x_teams.csv` |
| 10 | (mehrere Kameras) globale IDs | `tools/assemble_x.py` → `src/analytics/match_assembler.py` | 3×`x_teams.csv` → `review_assembled/x_match_tracks.csv` + `x_idmap.csv` |
| 11 | Ansehen/Bewerten | `tools/review/` (lokale Browser-App) | CSVs + Videos → Review-Tool |

Noch **nicht** in Kette A (nur in B vorhanden): Ballbesitz (6), Events, StatsBomb,
physische Metriken, Roster, Video-Render.

---

## 3. Modul-Referenz

Gruppiert nach Verantwortung. `[A]` = in Produktions-Pipeline aktiv, `[B]` = nur
Legacy/vollständige Kette, `[A+B]` = beide, `[Tool]` = eigenständiges Werkzeug.

### 3.1 Grundlagen
- **`src/core/detection.py`** `[A+B]` — Datenmodell einer einzelnen Detektion (Box, Klasse, Konfidenz).
- **`src/core/frame.py`**, **`src/core/video_metadata.py`** `[A+B]` — Frame- und Video-Metadaten-Hilfen.
- **`src/config/config_loader.py`** `[A+B]` — lädt YAML-Konfigs in typisierte Dataclasses (Video, Detection, Tracking, Field, Venue …).
- **`src/video/video_reader.py`** `[A+B]` — Frames aus einem Video lesen.
- **`src/video/video_player.py`** `[Tool]` — einfacher Player (Debug).
- **`src/video/stabilizer.py`** `[B]` — Bild-Stabilisierung für die Legacy-Tracking-Stufe (ORB-Homographie, `unwarp_boxes`). In Kette A macht das die Motion-Kompensation stattdessen.

### 3.2 Kamerabewegung
- **`src/motion/camera_motion.py`** `[A]` — **Kernmodul.** Schätzt pro Frame die
  Homographie auf einen Referenzrahmen (Schwenk/Zoom/Rotation). Grasmasken-Features
  (ORB), Direct-to-Reference-Matching, zeitliche Median-Glättung, photometrischer
  Verifier. `MotionTrack` (speicher-/ladbar als `.npz`), `to_reference(frame)`,
  `camera_state(frame)`, `camera_motion_for(video, cache)`. Wiederverwendbar von
  allen Tracking-Programmen.

### 3.3 Kalibrierung (Bild → Feld)
- **`src/calibration/pitch_landmarks.py`** `[A+B]` — echte Feldgeometrie aus der
  Venue-Größe (FIFA-Maße skaliert), Landmarken, Ellipsen für Kreise, Snap-Ziele.
- **`src/calibration/homography.py`** `[A+B]` — Homographie Bild↔Feld.
  `compute_manual_calibration` (Least-Squares über alle geklickten Punkte),
  `PitchCalibration.load`, **`filter_to_pitch_csv`** (mappt Boxen in Feldkoords,
  wirft Off-Pitch raus, nutzt camera_motion zur Schwenk-Kompensation),
  `suggest_worst_point` (Berater).
- **`src/calibration/field_region.py`** `[B]` — Feld-Polygon in Bildkoordinaten,
  um Personen außerhalb des Felds zu verwerfen.

### 3.4 Detektion
- **`models/football_x_ref.pt`** — das aktuelle Detektionsmodell (YOLOv8x,
  fußballspezifisch). Klassen: 0=ball, 1=goalkeeper, 2=player, 3=referee. **Kein
  Kopf/Beine** — jede Box ist eine ganze Person oder der Ball.
- **`src/detection/yolo_detector.py`** `[B]` — YOLO-Wrapper.
- **`src/detection/result_parser.py`** `[B]` — YOLO-Ergebnis → `Detection`-Objekte.
- **`src/detection/dedup.py`** `[A]` — **entfernt Duplikate auf derselben Person.**
  Stufe 1 `dedup_frame_boxes` (pro Frame): Fragment = kleinere Box stark enthalten
  (zwei Bänder: IoMin≥0,85 & Höhe≤0,80, ODER IoMin≥0,70 & Höhe≤0,60 für klare
  Kopf-/Fuß-Fragmente); near-identical IoU≥0,80. Stufe 2 `glued_pair_frame_drops`
  (Track-Paare, die episodisch verkleben).
- **`src/detection/ball_lowconf.py`**, **`ball_search_tracker.py`**, **`ball_tracker.py`**, **`ball_extractor.py`** `[B]` — separate Ball-Erkennung/-Verfolgung (Suchfenster, niedrige Konfidenz). Aktuell auf Wunsch **nicht** genutzt; in Kette A wird der Ball sogar herausgefiltert (siehe `_drop_ball_shaped`).

### 3.5 Tracking (ID-Vergabe & Track-Bereinigung)
- **`src/tracking/bytetrack_boxes.py`** `[A]` — fährt ultralytics-**ByteTrack**
  direkt mit den fertigen Boxen (nicht über `model.track`, spart die 9 s/Frame).
  Assoziation im **stabilisierten** Referenzrahmen (Boxen via `MotionTrack`
  gewarpt, gespeichert werden die Rohboxen). Kalman + Hungarian → stabilere IDs
  als der alte Greedy-Tracker.
- **`src/tracking/appearance.py`** `[A+B]` — Aussehens-Fingerprint pro Track
  (grass-maskierte HS-Histogramme, Ober-/Unterkörper). `appearance_distance`
  (Bhattacharyya). Wird vom Stitcher und vom ID-Split genutzt.
- **`src/tracking/enhanced_stitcher.py`** `[A+B]` — führt fragmentierte Tracks
  desselben Spielers zusammen: Gates = Feld-Distanz + plausible Geschwindigkeit +
  Trikotfarbe. Zusatz `motion_stitch_small`: Bewegungs-Merge für ferne kleine
  Fragmente, die zu klein für einen Fingerprint sind (enge Gates + gleiche Farbe).
- **`src/tracking/smoothing.py`** `[A+B]` — gap-aware Glättung: teilt Tracks an
  großen Lücken, Speed-Gate gegen Ausreißer, Interpolation, Rolling-Median/Mean.
- **`src/tracking/interpolation.py`** `[B]` — füllt Okklusions-Lücken innerhalb
  eines Tracks.
- **`src/tracking/track_filter.py`** `[B]` — Track-Level-Gültigkeitsfilter (braucht
  Feldkoords): verwirft Tracks, die zu selten im Feld sind.
- **`src/tracking/offfield_filter.py`** `[B]` — **entfernt Zuschauer/Trainer, die
  den Pitch-Filter überleben** (stehen hinter der fernen Auslinie → mappen fast
  auf die Linie). *Hinweis:* hängt aktuell nur in Kette B; Kette A könnte davon
  profitieren.
- **`src/tracking/track_stitcher.py`** `[B]` — älterer Stitcher (Vorläufer des
  enhanced_stitcher).
- **`src/tracking/id_switch.py`** `[A, hinter Flag]` — teilt einen Track, wenn sich
  sein Aussehen stark und anhaltend ändert (ID auf anderen Spieler gesprungen).
  Konservativ; auf diesem Material nur bedingt zuverlässig (kleine, ähnliche
  Spieler) → derzeit **nicht** standardmäßig aktiv.

### 3.6 Analytik
- **`src/analytics/team_classifier.py`** `[A+B]` — Team pro Track über
  Trikotfarben-Clustering (`classify_teams`).
- **`src/analytics/match_assembler.py`** `[A]` — **fügt die Kameras zusammen.**
  Tracks, die räumlich+zeitlich in der Überlappung koinzidieren, werden per
  Union-Find zu einer **global_id** verschmolzen (Auto-Sync der Frame-Offsets).
  Optional `idmap_csv` → schreibt `(view, local_id) → global_id` (fürs Review-Tool).
- **`src/analytics/possession.py`** `[B]` — Ballbesitz: pro Frame der Spieler am
  nächsten zum Ball.
- **`src/analytics/events.py`** `[B]` — Events aus der Ballbesitz-Folge (Ballwechsel …).
- **`src/analytics/physical_metrics.py`** `[B]` — ball-unabhängige Kennzahlen:
  Laufdistanzen, Team-Kompaktheit, Durchschnittspositionen (in echten Metern).
- **`src/analytics/pitch_plots.py`** `[A+B]` — Visualisierungen: **`run_paths`**
  (Laufwegkarte, bricht an Lücken), `team_heatmap`, `formation_map`.
- **`src/analytics/roster.py`** `[B]` — schlägt Team+Name pro Track vor (Nutzer
  korrigiert), schneidet Spielerbilder.

### 3.7 Export & Visualisierung
- **`src/export/statsbomb_exporter.py`** `[B]` — Events → StatsBomb-JSON (Koordinaten passen schon).
- **`src/export/csv_exporter.py`** `[A+B]` — Detektionen als CSV.
- **`src/visualization/pitch_overlay.py`** `[A+B]` — zeichnet Feldlinien aus der Homographie ins Bild.
- **`src/visualization/team_ball_renderer.py`** `[B]` — rendert team-gefärbte Boxen + Ball ins Video (Validierung).

### 3.8 Orchestrierung / Pipelines
- **`src/pipeline/analysis_pipeline.py`** `[B]` — Legacy-Kern: Video → Detektion/Tracking (model.track) → Stitching → Export.
- **`src/workflow.py`** `[B]` — vollständige Ende-zu-Ende-Kette pro Clip +
  `run_match`/`reassemble_match` (Mehr-Kamera). Enthält die Analytik/Export-Endstücke.
- **`src/main.py`** `[B]` — Einstieg: `python -m src.main <config.yaml>`.
- **`src/match.py`** `[B]` — Ein-Befehl-Match über einen Ordner voller Clips.
- **`tools/xmodel_pipeline.py`** `[A]` — **Orchestrator der Produktions-Pipeline**
  (Schritte 2–9 oben) + CLI-Flags: Standard = ByteTrack; `--greedy` (alter Tracker),
  `--stitch` (Bewegungs-Stitch-Variante `xs_`).

### 3.9 Werkzeuge (`tools/`)
- **`tools/autolabel.py`** `[Tool/A]` — robustes, wiederaufsetzbares Offline-Labeln
  mit dem x-Modell (Keep-Awake, atomarer Fortschritt, `ALL_DONE`). Für Über-Nacht-Läufe.
- **`tools/assemble_x.py`** `[Tool/A]` — assembliert die 3 ByteTrack-Läufe zu
  globalen IDs (nutzt `match_assembler`), schreibt idmap fürs Review-Tool.
- **`tools/analytics_x.py`** `[Tool/A]` — dockt die **ball-freie Analytik** an die
  globalen x-Tracks an (Laufwege, Heatmaps, Distanzen, Kompaktheit, Formation →
  `review_assembled/analytics/`). Erster Teil des „Lücke schließen" (PLAN Nr. 2).
- **`tools/nightly.py`** + **`run_nightly.bat`** `[Tool/A]` — **Nacht-Runner**:
  baut das Produktions-Tracking neu, assembliert global, legt einen **datierten
  Schnappschuss** unter `data/output/review_runs/<ts>/` ab. Robust (keep-awake,
  `ALL_DONE`, prune). Das Review-Tool zeigt Schnappschüsse als 🌙-Läufe → A/B
  „letzte Nacht vs. heute". (`--snapshot-only` = nur ablegen ohne Rebuild.)
- **`tools/calibrate_field.py`** `[Tool]` — **manuelle Kalibrierungs-UI** (Zoom/Pan,
  Snap auf Linien/Kreise, Berater). Der einzige manuelle Schritt.
- **`tools/mark_field.py`** `[Tool]` — Feld-Polygon markieren (für `field_region`).
- **`tools/benchmark_tracking.py`**, **`tools/compare_tracking.py`** `[Tool]` —
  Tracking-Experimente/Vergleiche (metrikbasiert; Achtung: Metriken täuschen, siehe PLAN).
- **`tools/render_field_stabilization_test.py`** `[Tool]` — rendert Frames zur
  Sichtprüfung von Kalibrierung + Stabilisierung.

### 3.10 Review-Tool (`tools/review/`) — das interaktive Auswertungswerkzeug
- **`server.py`** — lokaler HTTP-Server (Standardbibliothek, kein Flask; HTTP-Range
  für Video-Scrubbing). Endpunkte: `/api/runs` (Läufe, inkl. globalem),
  `/api/tracks` (Boxen pro Lauf+Clip, im globalen Modus auf global_id gemappt),
  `/video/<clip>.mp4`, `/api/feedback` (speichern/lesen).
- **`index.html` / `app.js` / `styles.css`** — Browser-App: 3 Videos synchron mit
  Box/ID-Overlay, Play/Pause/Scrub, IDs ein-/ausblenden (pro Kamera **oder** global),
  A/B-Laufvergleich, gekoppelte Laufwegkarte, Feedback pro Track.
- Start: `python -m tools.review.server` → http://127.0.0.1:8731/

### 3.11 Evaluation
- **`src/evaluation/tracking_metrics.py`** `[Tool]` — Tracking-Kennzahlen
  (Fragmentierung, ID-Switches …). **Nur Kontext, nicht Entscheider** — die
  Projektregel ist: das Bild (Laufwegkarte/Crops) entscheidet.

---

## 4. Daten- & Verzeichnislayout

```
data/
  raw/test1/{match_center,match_left,match_right}.mp4   # die 3 Kameraansichten (30 fps, 20 s Testclips)
  autolabel/<clip>/labels/*.txt                         # x-Modell-Labels (offline erzeugt)
  autolabel/<clip>/images/*.jpg                         # zugehörige Frames
  output/<clip>/detections/                             # pro Clip:
      x_raw.csv         # getrackte Rohboxen (nach Dedup + Ball-Filter)
      x_pitch.csv       # + Feldkoordinaten (nach Off-Pitch-Filter)
      x_tracks.csv      # nach Stitch + Glättung
      x_teams.csv       # + Teamzuordnung  ← das zeigt das Review-Tool
      xs_*.csv          # Variante mit Bewegungs-Stitch (A/B-Vergleich)
      camera_motion.npz # gecachte Kamerabewegung
      teams.csv         # (Legacy nano-Lauf, im Tester ausgeblendet)
  output/review_assembled/
      x_match_tracks.csv  # globale IDs (assembliert)
      x_idmap.csv         # (Kamera, lokale ID) → global_id
  output/review_feedback/<lauf>.json                   # Feedback aus dem Review-Tool
config/
  test1_match_{center,left,right}.yaml                  # pro Clip: Modell, Kalibrierung, Venue
  fields/*.json                                        # Kalibrierungen (geklickte Punkte + Homographie)
  trackers/*.yaml                                      # ByteTrack/BoT-SORT-Konfigs
models/
  football_x_ref.pt                                    # aktives Detektionsmodell (YOLOv8x Fußball)
  pitch_keypoints.pt                                   # verworfenes Feldlinien-Modell (inaktiv)
```

---

## 5. Startbefehle (Cheatsheet)

```bash
# Produktions-Pipeline (ByteTrack) für alle 3 Clips neu bauen
python -m tools.xmodel_pipeline
python -m tools.xmodel_pipeline --stitch     # Variante xs_ (Bewegungs-Stitch)
python -m tools.xmodel_pipeline --greedy     # alter Tracker (Vergleich)

# Globale IDs (Kameras zusammenführen)
python -m tools.assemble_x

# Review-Tool starten
python -m tools.review.server                # -> http://127.0.0.1:8731/

# Offline neu labeln (über Nacht, x-Modell)
python -m tools.autolabel --batch

# Manuelle Kalibrierung
python -m tools.calibrate_field <FRAME_IDX> <config.yaml>

# Legacy-Vollkette (nano, mit Analytik/Export)
python -m src.main <config.yaml>
python -m src.match data/raw/test1
```

---

## 6. Grenzen & offene Punkte (Kurzfassung)

- **Detektion offline:** x-Modell 9 s/Frame CPU → nur Etikettierer, bis GPU da ist.
  Deshalb nur 20-s-Testclips.
- **Analytik/Export** (Ballbesitz→Events→StatsBomb) hängt noch an der Legacy-Kette,
  nicht an den guten x-Tracks → PLAN.md Nr. 2.
- **ID-Wechsel** bei Gleich-Team-Kreuzungen sind mit den schwachen Farbmerkmalen
  kleiner Spieler nicht sicher trennbar → braucht Re-ID/GPU.
- **Ferne Feldhälfte** dünn (Detektionsdichte), viele Kurzfragmente ferner Spieler.

Prioritäten und Fortschritt: **[PLAN.md](PLAN.md).**
