# football-analysis — Plan & Arbeitsweise

> **Lebendiges Dokument.** Immer aktuell halten, immer daran halten.
> Ganz oben steht die aktuelle **Nr. 1**. Erledigtes wandert ins Änderungslog.
> Letzte Aktualisierung: 2026-07-10

---

## Nordstern

Aus Veo-Kameramaterial (3 Ansichten desselben Spiels) **verlässliche Tracking-Daten
im StatsBomb-Format** erzeugen: jeder Spieler durchgehend mit stabiler Identität in
Feldkoordinaten → daraus Ballbesitz, Events, Laufwege. CPU-only bis GPU verfügbar ist;
bis dahin wird nur auf 20-Sekunden-Testclips gearbeitet.

---

## Architektur heute (Stand 2026-07-10)

**Das Fundament trägt** (visuell verifiziert): Video-I/O, Kamerabewegung
(`src/motion/camera_motion.py`), Kalibrierung (`tools/calibrate_field.py`,
echte Feldgröße 99×64 m).

**Die zentrale Lücke:** Es gibt zwei Pipelines, die nicht verbunden sind.
- `src/workflow.py` (+ `src/match.py`): vollständige Kette bis StatsBomb-Export
  und Multi-Kamera-Zusammenbau — aber vom **schwachen** nano-Detektor gespeist.
- `tools/xmodel_pipeline.py`: der **starke** Strang
  (x-Modell offline → Dedup → ByteTrack → Teams) — **endet aber bei `teams`**.
  Ballbesitz, Events, Export, Zusammenbau erreicht er nicht.

→ Das Produkt wird heute aus den schlechteren Tracks gebaut. Die guten Tracks
laufen ins Leere. Diese Lücke zu schließen ist Phase 1.

**Entscheidungs-Grundregel (teuer gelernt):** Metriken (Track-Zahl, ID-Switches)
täuschen auf diesem Material. **Das Bild entscheidet** — Laufwegkarten und
Frame-Crops sind der Schiedsrichter, nicht Zahlentabellen.

---

## Prioritätenliste

### ▶ Nr. 1 (AKTUELL) — Interaktives Auswertungs- & Review-Tool + Nacht/Tag-Rahmen

Ein echtes Werkzeug (kein statischer Bildersatz), mit dem wir Läufe bewerten und
Feedback pro Track geben. Muss-Anforderungen:

1. **Zwei Läufe vergleichen (A/B).** Zwei Ergebnis-Sätze nebeneinander laden
   (z. B. Greedy vs ByteTrack, oder Lauf von gestern vs heute).
2. **Die 3 Videos parallel & synchron** abspielen (match_center/left/right),
   mit Box-Overlay + Track-IDs + Teamfarben aus den CSVs.
3. **Pause & Scrubbing** — an jeder Stelle anhalten, frameweise vor/zurück.
4. **IDs interaktiv ein-/ausblenden** — nur eine oder mehrere ausgewählte
   Track-IDs anzeigen, Rest ausblenden. So folgt man einem Spieler sauber.
5. **Laufwegkarte dazu**, gekoppelt an die Auswahl (gewählte IDs → deren Pfade).
6. **Feedback pro Track direkt im Tool** erfassen und speichern: wo ein Track
   abbricht, ob er zwei Spieler verwechselt (ID-Swap), Doppelungen, etc.
   → als strukturierte Datei (JSON), damit Verbesserungen gezielt darauf zielen.

**Technik (entschieden 2026-07-10):** lokale Browser-App. Server mit Python-
Standardbibliothek (`http.server`, HTTP-Range für Video-Scrubbing) — **kein Flask**
(fehlt, keine neue Abhängigkeit). Videos 30 fps / 600 Frames / 1920×1080.
Ein „Lauf" = ein Datei-Präfix in `data/output/<clip>/detections/` (z. B. `x_` =
ByteTrack), sodass A/B einfach zwei Präfixe vergleicht. Dateien unter `tools/review/`.

**Milestones (alle ✅ erledigt 2026-07-10, je einzeln im Browser verifiziert):**
- ✅ A: Server (`tools/review/server.py`, Range-Video) + 3 synchrone Videos +
  Box/ID-Overlay + Play/Pause/Scrub/Framezähler.
- ✅ B: IDs interaktiv ein-/ausblenden (Textliste + Klick auf Box).
- ✅ C: A/B — zwei Läufe (Präfixe) umschalten oder überlagern (A durchgezogen,
  B gestrichelt). nano-Lauf liegt schon vor → echtes A/B ohne Extra-Rechnung.
- ✅ D: Laufwegkarte (StatsBomb-Feld), Pfade der gewählten IDs, farbig nach
  Kamera, Positions-Punkt am aktuellen Frame.
- ✅ E: Feedback pro Track (Kamera/ID/Problemtyp/Notiz/Frame) → gespeichert unter
  `data/output/review_feedback/<lauf>.json`.

**Starten:** `python -m tools.review.server` → http://127.0.0.1:8731/
(Standardbibliothek, keine Abhängigkeiten. `.claude/launch.json` fürs Preview vorhanden.)

**Korrektur 2026-07-10:** Track-IDs sind **pro Kamera lokal** — ID 5 in Mitte ≠
ID 5 in Links (verschiedene Personen). Tool zeigt Auswahl jetzt kamera-spezifisch
(Tokens M5/L12/R3, Klick auf Box wählt genau diese Kamera-ID). Damit zeichnet die
Laufwegkarte keine drei fremden Personen mehr als eine.

**Echte globale Identität (ein Spieler = eine ID über alle Kameras) = Phase 2/Nr. 2:**
`match_tracks.csv` (assembliert) hat `global_id` + `views`(A/B/C) + Feldkoords, aber
KEINE Bildboxen und ist noch aus der alten Greedy-Assemblierung (stale). Für den
aktuellen ByteTrack-Lauf muss neu assembliert werden; dann kann das Tool einen
„global"-Modus bekommen (global_id overlay via (view,local_id)→global_id-Mapping).

**✅ Nacht-Runner (2026-07-11):** `tools/nightly.py` + `run_nightly.bat` bauen die
Produktions-Tracking neu, assemblieren global und legen einen **datierten
Schnappschuss** unter `data/output/review_runs/<ts>/` ab. Der Server bietet ihn als
🌙-Lauf (`snap:<ts>:`, auch global) → im Tester A/B „letzte Nacht vs. heute".
Robust (keep-awake, ALL_DONE, prune auf 10). Damit ist Nr. 1 **komplett**.

**Noch offen am Tool (klein, später):** Feedback-Einträge im Video anspringen;
Mindest-Track-Länge-Umschalter.

**Nacht/Tag-Rahmen drumherum** (sobald das Tool steht):
- 🌙 **Nacht:** robuster Loop-Runner (`run_nightly.bat`, Muster von
  `run_autolabel_overnight.bat`: keep-awake, resumable, `ALL_DONE`) fährt die
  Pipeline auf den 3 Testclips und legt einen Lauf-Ordner mit Zeitstempel an,
  den das Tool morgens laden kann.
- ☀️ **Morgen:** Lauf im Tool öffnen, gegen den Vorlauf vergleichen, Feedback geben.
- 🔧 **Tag:** **eine** Änderung, schnell auf kurzem Clip validiert, committet,
  nächsten Nacht-Lauf einreihen. `PLAN.md` aktualisieren.

### Nr. 2 — Phase 1: Die Lücke schließen  *(begonnen)*
x-Modell-Tracks an die vorhandene Auswertung andocken. **Erledigt 2026-07-11:**
ball-freier Teil via `tools/analytics_x.py` — läuft physische/taktische Analytik +
Plots auf den **globalen** Tracks (`review_assembled/x_match_tracks.csv`, Venue 99×64)
→ `review_assembled/analytics/` (Laufwege, Heatmaps, Distanzen, Kompaktheit,
Formation). Laufwegkarte gut; Formation/Distanzen noch fragmentierungsbegrenzt
(172 global_ids statt ~22 → Phase 3). **Offen:** Ballbesitz → Events → StatsBomb-
Events + Video-Render — brauchen den Ball (bewusst zurückgestellt).

### Nr. 3 — Phase 2: Multi-Kamera-Identität  *(Mechanik ✅, Qualität offen)*
`assemble_match` auf den ByteTrack-Tracks laufen lassen → globale IDs. **Erledigt
2026-07-10:** Assembler schreibt jetzt optional das `(view,local_id)→global_id`-
Mapping (`idmap_csv`); `tools/assemble_x.py` erzeugt `data/output/review_assembled/
x_match_tracks.csv` + `x_idmap.csv` (A=Mitte, B=Links, C=Rechts); das Review-Tool
hat einen **globalen Modus** (`x_@global`), in dem eine ID = ein Spieler über alle
Kameras ist (verifiziert: global 373 gleichzeitig in Mitte+Links, ein Feldpfad).
**Ehrlich offen:** 413 lokale → nur 379 globale (nur 32 kameraübergreifend). Grund
ist NICHT der Assembler, sondern die **Fragmentierung pro Kamera** (~140 Tracks/
Kamera für ~15 sichtbare Spieler) + dass Spieler außerhalb der Mittel-Überlappung
korrekt nur in einer Kamera sind. → echte Verbesserung kommt über Phase 3
(Kurzfragmente/Tracking), nicht über die Assemblierung.

### Nr. 4 — Phase 3: Feinschliff  *(in Arbeit)*
**Diagnose 2026-07-10:** Kurzfragmente (<20f) sind 69–81 % der Tracks, aber nur
5–13 % der Boxen — ferne kleine Spieler (Boxhöhe ~34px vs 54px). ~50 % sind
stitchbar (Nachbar-Track in ≤4 Einheiten/≤20f), ~50 % isoliert (2–3f-Blips ferner
Spieler, nicht rettbar — Detektionsdichte-Limit). Der `enhanced_stitcher` merged
sie bisher nicht, weil 74–91 % zu klein für einen Appearance-Fingerprint sind und
er ohne Fingerprint ablehnte.
**Fix (gebaut):** `stitch_csv_enhanced(motion_stitch_small=True)` — Bewegungs-Merge
wenn Appearance fehlt, mit engen Gates (Distanz ≤3 Einheiten, gleiche Teamfarbe,
eine Seite <20f). Nicht-destruktive Variante `xs_` (`python -m tools.xmodel_pipeline
--stitch`) → im Tool A/B gegen `x_` prüfbar.
- **Erledigt 2026-07-11:** Motion-Stitch ist jetzt **Standard** im Produktions-Lauf
  `x_`; zusätzlich `_drop_short_far` (Tracks <10f UND <45px median = isolierte ferne
  Blips raus, nahe/große bleiben). Ergebnis: Tracks pro Kamera 74/117/13 → **53/67/10**,
  globale Spieler **172 → 98**, Laufwege sauberer, Abdeckung intakt (visuell bestätigt).
  Im Tester direkt vergleichbar: `x_` (Phase 3) vs 🌙-Schnappschuss (vorher).
- Offen: ferne Feldhälfte (Detektionsdichte); Rest-Fragmentierung sind Gleich-Team-
  Wechsel + nicht-gemergte Kamera-Duplikate → Trikotnummern / Re-ID (GPU).

### Nr. 5 — Phase 4: Skalierung aufs ganze Spiel
Wartet auf GPU (x-Modell 9 s/Frame auf CPU). Bis dahin Testset = kurze Clips.

---

## Arbeitsweise (Vereinbarung)

- **Ein Plan, gepflegt.** Diese Datei bleibt aktuell; wir halten uns an die
  Reihenfolge. Neue Erkenntnisse ändern den Plan, nicht die Gewohnheit.
- **Nacht rechnet, Tag urteilt.** Teure Läufe über Nacht, menschliche Bewertung
  am Tag. Brücke sind visuelle/interaktive Artefakte, keine Metriken.
- **Eine Änderung pro Tag-Zyklus**, kurz validiert, dann committen.
- **Visueller Beweis vor Metrik.** Immer.

---

## Änderungslog

- **2026-07-11 (11)** — Phase 3: Motion-Stitch als Standard in `x_` +
  `_drop_short_far` (kurze+kleine ferne Blips). Tracks 74/117/13→53/67/10, global
  172→98, Laufwege sauberer, Abdeckung intakt. `x_` im Tester vs 🌙-Snapshot (vorher).
  Formation lesbarer, aber noch nicht ~22 (Rest = Gleich-Team-Wechsel/Kamera-Dupes).
- **2026-07-11 (10)** — Nr. 2 begonnen: `tools/analytics_x.py` dockt die ball-freie
  Analytik an die globalen x-Tracks an (→ `review_assembled/analytics/`). Laufwegkarte
  des ganzen Spiels gut; Formation/Distanzen zeigen die Fragmentierungsgrenze
  (172 global_ids). Nächster Hebel dafür: Phase 3 (Fragmente reduzieren).
- **2026-07-11 (9)** — Nacht-Runner gebaut (`tools/nightly.py`, `run_nightly.bat`):
  Rebuild + Assembly + datierter Schnappschuss (`review_runs/<ts>/`). Server zeigt
  Snapshots als 🌙-Läufe (`snap:<ts>:`, `_run_paths`-Resolver, path-basierte idmap).
  Getestet: Snapshot erscheint + lädt. **Nr. 1 (Review-Tool + Nacht/Tag-Rahmen) komplett.**
  Nächstes: Nr. 2 (x-Tracks → Analytik/Export andocken).
- **2026-07-11 (8)** — Off-Field-Filter in x-Pipeline eingebaut
  (`filter_offfield_tracks_csv`, nach Feldkoords, vor Stitch). Neuer Param
  `far_static_only=True`: ferne Auslinie nur wenn statisch → bewegte ferne Spieler
  bleiben (Regel A hätte sonst echte Spieler nahe der fernen Linie gedroppt, gemessen
  Bewegung 4–6 vs Zuschauer ~0). Entfernt statische Randpersonen (Trainer/Auswechsel:
  0/4/3) + 1–2f-Blips. Rebuild+Reassembly. Globale Spieler 379→352→**172** über die
  Bereinigungen. ARCHITECTURE.md (Repo, + Artifact-Ansicht) als lesbare Systemübersicht.
- **2026-07-11 (7)** — Ball nicht mehr tracken (Nutzerwunsch): x-Modell labelt den
  Ball zeitweise als „player", rutscht durch den Klassenfilter. Fix in
  `tools/xmodel_pipeline._drop_ball_shaped`: Track-Level, entfernt Tracks mit
  median h/w<1,3 UND median Höhe<25px (Ball ist klein+breit; echte Spieler immer
  hoch, kleinste ar≥1,5). Entfernt 7/47/1 Ball/Artefakt-Tracks (Links: Ball mit
  89/48/42… Frames), behält alle kleinen echten Spieler. Rebuild x_/xs_ + Reassembly.
- **2026-07-10 (6)** — Nutzer-Feedback aus Tester, 3 Probleme untersucht:
  **P1 Zuschauer**: NICHT in `x_`-Daten (0 dauerhafte Nicht-Spieler-Tracks); Quelle
  war der nano-Vergleichslauf → im Tester ausgeblendet (Präfix "" raus).
  **P2 Kopf+Körper**: real (242/319/24 überlappende Paare überlebten Dedup, IoMin
  0,7–0,85). Fix: zweites Dedup-Band (IoMin≥0,70 & Höhe<0,60); 12/12 Stichproben
  waren echte Körperteil-Fragmente. Rebuild x_/xs_.
  **P3 gleiche ID→versch. Spieler**: appearance-Split gebaut (`src/tracking/id_switch.py`),
  aber isoliert getestet nur 4 mehrdeutige Splits → NICHT eingebaut. Grobe
  Team-Wechsel selten (gut); Rest sind Gleich-Team-Kreuzungen, mit den schwachen
  Histogrammen kleiner Spieler nicht trennbar. Ehrlicher harter Grenzfall → braucht
  Re-ID/GPU oder trainiertes Modell. Modul bleibt hinter Flag verfügbar.
- **2026-07-10 (5)** — Phase 3 Diagnose + Fix: Kurzfragmente = ferne kleine
  Spieler, ~50 % stitchbar; Stitcher lehnte sie ab (zu klein für Fingerprint).
  Bewegungs-Stitch-Fallback gebaut (`motion_stitch_small`), Variante `xs_`.
  Merges Mitte 3→39, Links 6→86, Rechts 0→9; substanzielle Tracks (Karte) stabil
  (45→46, 59→57). Laufwege x_ vs xs_ visuell geprüft: sauberer, KEINE neuen
  Teleport-Artefakte. Im Tool als 3. Lauf A/B prüfbar.
- **2026-07-10 (4)** — Phase 2 Mechanik gebaut: `assemble_match` schreibt idmap;
  `tools/assemble_x.py`; Review-Tool globaler Modus (`x_@global`). Verifiziert
  (global 373 in Mitte+Links, ein Pfad). Merge-Rate niedrig (32 kameraübergreifend)
  wegen Fragmentierung pro Kamera → Hebel ist Phase 3, nicht der Assembler.
- **2026-07-10 (3)** — Review-Tool: IDs auf **pro-Kamera** korrigiert (Tokens
  M/L/R), nachdem auffiel dass lokale IDs verschiedene Personen konflatierten.
  Verifiziert (M5 → nur Mitte, Links 0). Echte globale IDs bleiben Phase 2.
- **2026-07-10 (2)** — Review-Tool gebaut: `tools/review/` (server.py + index.html
  + app.js + styles.css, ~750 Zeilen, nur Standardbibliothek). Milestones A–E alle
  im Browser verifiziert (synchrone Wiedergabe, ID-Filter, A/B nano vs ByteTrack,
  Laufwegkarte, Feedback-Speicherung). Nächster Schritt der Nr.-1-Achse: Nacht-Runner.
- **2026-07-10 (1)** — Plan angelegt. Architektur kartiert (zwei Pipelines, Lücke
  identifiziert). Erledigt davor: Detektions-Dedup (2 Stufen, 0 Restduplikate,
  kein echter Spieler verloren), ByteTrack ersetzt Greedy (saubere Laufwege,
  visuell bestätigt). Nr. 1 festgelegt: interaktives Review-Tool + Nacht/Tag-Rahmen.
