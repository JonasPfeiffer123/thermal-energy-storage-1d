# Third-party notices

thermal-energy-storage-1d is licensed under the MIT License (see
[LICENSE](LICENSE)). It contains material derived from the third-party work
listed below, which remains under its own license.

## FreeTTES

- Source: <https://github.com/gewv-tu-dresden/FreeTTES>
- Copyright (c) 2026, FreeTTES, TU Dresden – Gebäudeenergietechnik und
  Wärmeversorgung (developers: Andreas Herwig, Bogdan Narusavicius)
- License: BSD 3-Clause (full text below)

Derived material:

- `thermal_energy_storage_model/fluids.py`: the polynomial fits of
  `WaterProperties` (density, specific heat capacity and thermal conductivity
  of liquid water, 0–130 °C) are those of FreeTTES
  (`__Temperatur_Abhaengige_Stoffwerte` in `src/FreeTTES_model.py`).
- `benchmark/shared_utils.py`: the water density and enthalpy polynomials
  (same FreeTTES function) and the usable-energy calculation (after
  `__energie_nutz`).
- `benchmark/config_benchmark.py`: the benchmark parameters and the initial
  temperature profile follow the FreeTTES reference configuration
  (`src/config.json`).

FreeTTES itself is not part of this repository and not a dependency of the
package; the benchmark scripts can optionally run it from a separate checkout
as a reference model.

### FreeTTES license

```text
BSD 3-Clause License

Copyright (c) 2026, FreeTTES
TU Dresden – Gebäudeenergietechnik und Wärmeversorgung
Developers: Andreas Herwig, Bogdan Narusavicius
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
