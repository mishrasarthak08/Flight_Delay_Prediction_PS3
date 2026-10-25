# Dataset card

- **Source:** [Matin Sajadi Flights, Kaggle](https://www.kaggle.com/datasets/matinsajadi/flights), representing nycflights13/BTS flight records.
- **Pinned CSV SHA-256:** `8bdc6abbf07ac0cbb1036bc14a787cab3eea616be46c31899d2d8048963b08aa`.
- **Raw coverage:** 336,776 scheduled 2013 NYC departures, 21 columns, origins EWR/JFK/LGA. This is not a national sample of all US departures.
- **Supervised population:** 327,346 rows with observed arrival delay. The 9,430 missing arrival outcomes are excluded from supervised scoring. Without explicit status fields, their individual reasons are not all verified as cancellations or diversions.
- **Target:** arrival delay at least 15 minutes. Overall positive prevalence is approximately 24.47%; partition prevalences differ.
- **Timetable population:** all 336,776 raw rows, including missing arrival outcomes, contribute scheduled-demand counts.
- **Raw destinations versus supervised destinations:** 105 in the raw roster and 104 with observed arrival outcomes. Always identify the population when quoting coverage.
- **Quality controls:** duplicate flight-key rejection, valid HHMM clocks/dates, nonempty categorical identifiers, positive finite prediction distances, missingness counts and UTC arrival consistency audit.
- **Historical availability:** reconstructed actual arrival plus a declared 24-hour reporting lag. True historical publication times and schedule revisions are unavailable. Inconsistent timestamp records cannot supply historical labels.
- **Timezone metadata:** the tracked subset of the original nycflights13 airport metadata plus explicitly recorded territory timezone additions. Review `data/metadata/airport_timezones.json`.
- **Reproducibility:** `scripts/download_data.py` checks the complete CSV hash and refuses changed snapshots. The raw CSV is intentionally ignored by Git. The saved run profile records the hash and audit counts.
- **Use limits:** education and retrospective analysis. Current operational forecasting, other origins, cancellations/diversions and causal effects of rescheduling are not validated.

The project code license does not replace the dataset provider's terms. Obtain data through the linked source rather than committing the raw CSV to the repository.
