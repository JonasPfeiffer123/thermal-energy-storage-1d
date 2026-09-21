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

- [x] Tests für `TruncatedConeGeometry`: Volumenformel, `V_nodes`-Summe =
      Gesamtvolumen, `A_wall_nodes`, Entartungsfall `r_top == r_bottom` ≙
      Zylinder, Validierungsfehler bei `r <= 0`/`height <= 0`.
      (`tests/test_geometry.py`, dabei auch `CylinderGeometry`-Konstruktor-
      /Validierungspfade mitgetestet, die vorher ebenfalls ungetestet waren.
      `geometry.py`-Coverage: 42 % → 97 %.)
- [x] Tests für `TruncatedPyramidGeometry`: Volumenformel (analytische
      Integration), `V_nodes`-Summe = Gesamtvolumen, `A_wall_nodes`,
      `from_slope()`-Konstruktor, Validierungsfehler.
      (`tests/test_geometry.py`; Volumenformel zusätzlich unabhängig per
      Simpson-Integration von A(z)=a(z)·b(z) verifiziert.)
- [x] Tests für `SplitAmbientLoss`: Lid-Knoten nutzt `U_lid`/`T_ambient_lid`,
      übrige Knoten `U_wall`/`T_ambient`, Fallback wenn `T_ambient_lid=None`.
- [x] Tests für `GroundTemperatureLoss`: `T_ground_at_depth`-Formel
      (Exponentialansatz), Tiefenabhängigkeit pro Knoten, Validierungsfehler.
- [x] Tests für `TransientGroundLoss`: RC-Kette konvergiert im Steady State
      gegen den erwarteten Wert, Lid bleibt stationär (nicht Teil der
      Zeitintegration), `advance()`-Energiebilanz, Re-Init bei Knotenzahl-Wechsel.
      (`tests/test_losses.py`; Steady-State-Fixpunkt analytisch hergeleitet
      und empirisch gegen einen konvergierten Langzeitlauf verifiziert, bevor
      er als Assertion verwendet wurde. `losses.py`-Coverage: 27 % → 99 %.)
- [x] Tests für `UniformDiffusor`: Gewichte summieren zu 1, Gleichverteilung
      über `H_zone`, Fallback auf nächsten Knoten bei zu grobem Grid.
      (`tests/test_diffusors.py`, inkl. Solver-Integrationstest: Inflow
      verteilt sich nachweislich auf mehrere Knoten, Energieerhaltung bleibt
      erhalten. `diffusors.py`-Coverage: 52 % → 96 %.)
- [x] Tests für `HeatExchangerPort` lumped-Modus: ε-NTU-Formel, Energiebilanz
      (Wärmeeintrag in Tank = Enthalpieänderung des externen Kreises),
      mehrere gleichzeitige HX-Ports. (`tests/test_heat_exchanger.py`)
- [x] Tests für `HeatExchangerPort` segmented-Modus: knotenweise NTU-Aufteilung,
      `flow_direction="downward"` vs. `"upward"`, Konsistenz mit lumped-Modus
      im homogenen Temperaturfeld (beide Modi müssen dort gleiches Ergebnis liefern).
      (`tests/test_heat_exchanger.py`)

      **Dabei echten Bug gefunden und gefixt:** `flow_direction` war in
      `solver.py::_compute_hx_source_terms` (segmented-Zweig) invertiert.
      Für den **Standardwert** `"downward"` (laut Docstring/physics.md:
      "Entry at top, exit at bottom") verarbeitete der Code den Knoten am
      **unteren** Zonenrand zuerst statt am oberen – exakt umgekehrt zur
      dokumentierten und physikalisch beabsichtigten Semantik. Betraf jeden
      Nutzer des segmentierten Modus über eine Thermokline hinweg (bei
      homogener Zonentemperatur ist der Fehler unsichtbar, da dort Summe und
      `T_ext_out` unabhängig von der Reihenfolge sind – vermutlich deshalb
      bei Implementierung nicht aufgefallen). Fix: `reverse`-Bedingung in
      der Sortierung umgedreht. Da vorher **keine** Tests für den
      segmented-Modus existierten (0 % Coverage, siehe Analyse oben), war
      der Bug unsichtbar. Regressionstest pinnt jetzt explizit fest, welcher
      Zonenknoten den vollen `T_ext_in`-Antrieb sieht (der zuerst
      durchströmte), statt nur ein Aggregat zu prüfen.
- [x] Tests für das Headspace-Modell: Energiebilanz
      `C_hs dT_hs/dt = -Q_roof - Q_hs_water`, Wärmeeintrag in obersten
      Wasserknoten, Konsistenzcheck mit `state.T_headspace`.
      (`tests/test_headspace.py`; zusätzlich kombinierte Energiebilanz
      Wasser+Headspace über mehrere Schritte gegen die aufintegrierte
      Dachverlustleistung verifiziert, `dE_water + dE_headspace ==
      -Σ Q_roof·dt`, für expliziten und impliziten Solver.)
- [x] Tests für `StoragePresets.steel_tank_buried` und `.ptes` (bisher nur
      `steel_tank_aboveground` indirekt geprüft). (`tests/test_presets.py`,
      inkl. `ThermalStorage1D.from_preset()`, das ebenfalls ungetestet war.
      `presets.py`-Coverage: 73 % → 100 %.)
- [x] Tests für bislang unbenutzte Public API: `check_cfl()`, `get_soc()`,
      `max_stable_dt()` (aktuell 0 Aufrufe in `tests/`). (`tests/test_public_api.py`)

**P1 abgeschlossen.** Gesamt-Coverage: 66 % → 94 %
(`geometry.py` 97 %, `losses.py` 99 %, `diffusors.py` 96 %, `solver.py` 91 %,
`presets.py` 100 %). Verbleibende Lücken sind überwiegend abstrakte
`NotImplementedError`-Stubs und einzelne `_validate_config`-Fehlerzweige;
nicht weiter verfolgt, da geringer Grenznutzen.

## P2 – Validierung / Robustheit

- [x] **Massenbilanz-Warnung.** `StorageInputs`-Docstring
      (`thermal_energy_storage_model/state.py:119-121`) verlangt
      `Σ m_dot ≈ 0`, wird aber nirgends geprüft. `RuntimeWarning` analog zum
      bestehenden CFL-Check in `_step_single` ergänzt, wenn die Bilanz relativ
      zum größten Portfluss signifikant von 0 abweicht (Schwelle: 1e-6
      relativ, toleriert Fließkomma-Rauschen, greift aber bei vergessenen
      Gegen-Ports). `tests/test_mass_balance.py`.
      *(Geprüft: `benchmark/dronninglund_validation.py` und
      `hoje_taastrup_validation.py` berechnen den dritten Port bereits
      explizit als `-(f_top+f_mid)` „mass balance" – dort greift die neue
      Warnung nicht ungewollt. Kleine Restimbalancen durch deren
      Rausch-Filterung (`abs(f) > 0.01`) könnten die Warnung theoretisch
      triggern, aber nur als Info-Ausgabe, kein Testbruch, da diese Skripte
      ohne `filterwarnings=error` laufen.)*

## P3 – UI-Parität

- [ ] `HeatExchangerPort` in `ui/config_panel.py` / `ui/main_window.py`
      verfügbar machen (aktuell 0 Treffer – nur über Python-API nutzbar).
- [ ] Headspace-Modell in der UI konfigurierbar machen (aktuell 0 Treffer).
- [ ] `auto_substep`-Toggle in der UI ergänzen.

## P4 – Engineering-Prozess

- [x] Lint/Type-Check-Schritt in CI (`.github/workflows/tests.yml` hat nur
      `pytest`, kein ruff/mypy trotz `py.typed`-Marker in
      `thermal_energy_storage_model/py.typed`).

      `ruff` (Default-Regelsatz: E4/E7/E9/F/UP/I/SIM/RUF u. a.) und `mypy`
      als neue `[project.optional-dependencies].lint`-Gruppe ergänzt,
      repoweit sauber gezogen (173 Befunde: 102 automatisch, Rest manuell
      behoben oder mit begründetem Per-File-Ignore versehen – z. B.
      `BLE001`/`S110` in `ui/*` und `benchmark/*` für bewusst breite
      Exception-Behandlung in Live-Preview- bzw. Best-Effort-Codepfaden,
      `C408` in `tests/*` für den etablierten `dict(...)`-Stil). `mypy`
      auf `thermal_energy_storage_model/` beschränkt (6 Typfehler gefunden
      und behoben, u. a. `_ensure_init()` in `TransientGroundLoss` gibt das
      Array jetzt zurück statt `Optional`-Attribut erneut zu lesen). Neuer
      `lint`-Job in `.github/workflows/tests.yml` (Python 3.12, separat von
      der Test-Matrix, da mypy neuere numpy-Stubs mit Py3.12-only-Syntax
      nicht unter einem erzwungenen `python_version=3.10` parsen kann).

## Bewusst zurückgestellt

- **FMI-2.0-Schnittstelle**: im README als "planned" markiert, im Code nicht
  begonnen (0 Treffer für `pythonfmu`/`fmi`/`fmu`). Eigenständiges Feature,
  kein Bugfix/Härtung – nicht Teil dieses Backlogs.
- Physikalische Modellgrenzen aus `docs/physics.md` (Inlet-Jet-Mischung nicht
  abgebildet, λ(T)/cp(T) profilgemittelt statt knotenweise) sind bewusste,
  literaturgestützte Näherungen – kein Vollständigkeitsdefizit.
