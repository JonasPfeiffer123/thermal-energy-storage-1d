# Backlog: Vollständigkeit & Härtung

Ergebnis einer Vollständigkeitsanalyse von `thermal_energy_storage_model/` (2026-09-21).
Der numerische Kern (Zylinder-Geometrie, Standard-Verlustmodell, Standard-Fluid,
Zweikreis-Ports, Upwind/TVD, explizit/implizit) ist gut getestet und gegen zwei
reale PTES-Standorte validiert. Die seither hinzugekommenen Erweiterungen
(Wärmetauscher-Port, Headspace, erweiterte Geometrien/Verlustmodelle,
Diffusor-Verteilung) sind implementiert, aber unterproportional getestet, nicht
in der UI verfügbar und teils dokumentativ inkonsistent. Dieses Backlog trackt
die Behebung, priorisiert nach Aufwand/Nutzen.

Arbeitsweise: ein Item = ein Commit, Checkbox wird im selben Commit abgehakt.

## P0 – Dokumentation & Code-Hygiene

- [ ] **architecture.md ist veraltet.** Beschreibt das Paket noch als
      Einzeldatei (`docs/architecture.md:20,30`), tatsächlich ist es seit der
      Modularisierung auf 10 Dateien unter `thermal_energy_storage_model/`
      aufgeteilt. Modul-Übersicht und Datei-Beschreibung aktualisieren.
- [ ] **physics.md Codebeispiel ist nicht lauffähig.** `docs/physics.md:650-651`
      zeigt `Port(..., diffusor=UniformDiffusor(H_zone=1.0))` – `Port`
      (`thermal_energy_storage_model/ports.py:12-60`) hat kein `diffusor`-Feld;
      das Diffusor-Modell wird tankweit über `StorageConfig.diffusor_model`
      gesetzt. Beispiel korrigieren und explizit dokumentieren, dass der
      Diffusor pro Tank (nicht pro Port) gilt.
- [x] **Toter Code entfernen.** `_port_to_node()`
      (`thermal_energy_storage_model/solver.py:1107`) wird nirgends mehr
      aufgerufen (durch `DiffusorModel.node_weights()` ersetzt), nur noch in
      einem Kommentar in `diffusors.py:74` erwähnt. Entfernen.
      *(nebenbei gefunden: `PointDiffusor`-Docstring erzeugte eine
      `SyntaxWarning: invalid escape sequence '\D'` durch LaTeX in einem
      Nicht-Raw-String – auf `r"""..."""` umgestellt.)*

## P1 – Testabdeckung (Hauptbefund)

Coverage-Lauf (`pytest --cov`, 2026-09-21): Gesamt 66 %. Aufschlüsselung:

| Modul | Abdeckung | Ursache |
|---|---|---|
| `geometry.py` | 42 % | `TruncatedConeGeometry`, `TruncatedPyramidGeometry` = 0 % |
| `losses.py` | 27 % | `SplitAmbientLoss`, `GroundTemperatureLoss`, `TransientGroundLoss` fast 0 % |
| `diffusors.py` | 52 % | `UniformDiffusor` ungetestet |
| `solver.py` | 75 % | `_compute_hx_source_terms` (lumped+segmented) und `_compute_headspace_exchange` = 0 % |
| `presets.py` | 73 % | `steel_tank_buried`, `ptes` kaum geprüft |

Alle PTES-relevanten Pfade (genau das, was gegen Dronninglund/Høje Taastrup
validiert wird) laufen im CI-sicheren Teil der Suite nie durch – nur die
optionalen, standardmäßig übersprungenen Validierungsskripte berühren sie.

- [ ] Tests für `TruncatedConeGeometry`: Volumenformel, `V_nodes`-Summe =
      Gesamtvolumen, `A_wall_nodes`, Entartungsfall `r_top == r_bottom` ≙
      Zylinder, Validierungsfehler bei `r <= 0`/`height <= 0`.
- [ ] Tests für `TruncatedPyramidGeometry`: Volumenformel (analytische
      Integration), `V_nodes`-Summe = Gesamtvolumen, `A_wall_nodes`,
      `from_slope()`-Konstruktor, Validierungsfehler.
- [ ] Tests für `SplitAmbientLoss`: Lid-Knoten nutzt `U_lid`/`T_ambient_lid`,
      übrige Knoten `U_wall`/`T_ambient`, Fallback wenn `T_ambient_lid=None`.
- [ ] Tests für `GroundTemperatureLoss`: `T_ground_at_depth`-Formel
      (Exponentialansatz), Tiefenabhängigkeit pro Knoten, Validierungsfehler.
- [ ] Tests für `TransientGroundLoss`: RC-Kette konvergiert im Steady State
      gegen den erwarteten Wert, Lid bleibt stationär (nicht Teil der
      Zeitintegration), `advance()`-Energiebilanz, Re-Init bei Knotenzahl-Wechsel.
- [ ] Tests für `UniformDiffusor`: Gewichte summieren zu 1, Gleichverteilung
      über `H_zone`, Fallback auf nächsten Knoten bei zu grobem Grid.
- [ ] Tests für `HeatExchangerPort` lumped-Modus: ε-NTU-Formel, Energiebilanz
      (Wärmeeintrag in Tank = Enthalpieänderung des externen Kreises),
      mehrere gleichzeitige HX-Ports.
- [ ] Tests für `HeatExchangerPort` segmented-Modus: knotenweise NTU-Aufteilung,
      `flow_direction="downward"` vs. `"upward"`, Konsistenz mit lumped-Modus
      im homogenen Temperaturfeld (beide Modi müssen dort gleiches Ergebnis liefern).
- [ ] Tests für das Headspace-Modell: Energiebilanz
      `C_hs dT_hs/dt = -Q_roof - Q_hs_water`, Wärmeeintrag in obersten
      Wasserknoten, Konsistenzcheck mit `state.T_headspace`.
- [ ] Tests für `StoragePresets.steel_tank_buried` und `.ptes` (bisher nur
      `steel_tank_aboveground` indirekt geprüft).
- [ ] Tests für bislang unbenutzte Public API: `check_cfl()`, `get_soc()`,
      `max_stable_dt()` (aktuell 0 Aufrufe in `tests/`).

## P2 – Validierung / Robustheit

- [ ] **Massenbilanz-Warnung.** `StorageInputs`-Docstring
      (`thermal_energy_storage_model/state.py:119-121`) verlangt
      `Σ m_dot ≈ 0`, wird aber nirgends geprüft. `RuntimeWarning` analog zum
      bestehenden CFL-Check in `_step_single` ergänzen, wenn die Bilanz relativ
      zum größten Portfluss signifikant von 0 abweicht. Plus Test.

## P3 – UI-Parität

- [ ] `HeatExchangerPort` in `ui/config_panel.py` / `ui/main_window.py`
      verfügbar machen (aktuell 0 Treffer – nur über Python-API nutzbar).
- [ ] Headspace-Modell in der UI konfigurierbar machen (aktuell 0 Treffer).
- [ ] `auto_substep`-Toggle in der UI ergänzen.

## P4 – Engineering-Prozess

- [ ] Lint/Type-Check-Schritt in CI (`.github/workflows/tests.yml` hat nur
      `pytest`, kein ruff/mypy trotz `py.typed`-Marker in
      `thermal_energy_storage_model/py.typed`).

## Bewusst zurückgestellt

- **FMI-2.0-Schnittstelle**: im README als "planned" markiert, im Code nicht
  begonnen (0 Treffer für `pythonfmu`/`fmi`/`fmu`). Eigenständiges Feature,
  kein Bugfix/Härtung – nicht Teil dieses Backlogs.
- Physikalische Modellgrenzen aus `docs/physics.md` (Inlet-Jet-Mischung nicht
  abgebildet, λ(T)/cp(T) profilgemittelt statt knotenweise) sind bewusste,
  literaturgestützte Näherungen – kein Vollständigkeitsdefizit.
