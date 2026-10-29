# Data

The model input files for CAMELS-US, CAMELS-AUS and CAMELS-GB are published as assets of the GitHub release (five `.npz` files,
511 MB in total). They are derived from the sources below. The US and AUS files are distributed under CC BY 4.0 and the GB file
under the Open Government Licence v3.0, with the original attributions; please cite the sources when you use them.

```
python scripts/data/fetch_data.py    # downloads into data/ and checks the SHA256 of each file
```

The files can also be downloaded from the release page and placed in `data/`.

## Sources

| source | used for | reference | licence |
|---|---|---|---|
| CAMELS-US v1.2 | basin polygons, discharge 1980-2014, elevation bands, static attributes | Newman et al. (2015), HESS 19, 209-223, doi:10.5194/hess-19-209-2015; Addor et al. (2017), HESS 21, 5293-5313, doi:10.5194/hess-21-5293-2017; data: Zenodo record 15529996 | CC BY 4.0 |
| Daymet v4, daily, North America | US forcing 1980-2019 (prcp, tmax, tmin, srad, vp, dayl) | Thornton et al. (2020), Daymet Version 4, ORNL DAAC, doi:10.3334/ORNLDAAC/1840; Thornton et al. (2021), Sci. Data 8, 190 | NASA EOSDIS data, no restrictions; citation requested |
| USGS NWIS daily values | US discharge 2015-2019 (parameter 00060, statistic 00003) | U.S. Geological Survey, National Water Information System, https://waterdata.usgs.gov/nwis | public domain |
| CAMELS-AUS v2 (release 2.03) | AUS forcing (AGCD precipitation and temperature, SILO Morton wet-environment PET), discharge, attributes, boundaries | Fowler, Zhang and Hou (2025), ESSD 17, 4079-4095, doi:10.5194/essd-17-4079-2025; data: Zenodo record 14289037 | CC BY 4.0 |
| CAMELS-GB | GB PET (`peti`) and discharge (`discharge_spec`) to 2015-09-30, static attributes, elevation quantiles, boundaries | Coxon et al. (2020), ESSD 12, 2459-2483, doi:10.5194/essd-12-2459-2020; data: NERC EIDC, doi:10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9 | Open Government Licence |
| CAMELS-GB v2 | GB air temperature (`temperature_chess`) over the whole period; PET (`peti_chess`) and discharge (`discharge_spec`) 2015-10-01 .. 2019-12-31 | Coxon et al. (2026), ESSD 18, 4345-4371, doi:10.5194/essd-18-4345-2026; data: NERC EIDC, doi:10.5285/9a46d428-958f-4ac1-86eb-94eee70c0955 | Open Government Licence |
| HadUK-Grid, daily rainfall, 5 km | GB precipitation | Hollis et al. (2019), Geosci. Data J. 6, 151-159, doi:10.1002/gdj3.78; Met Office, distributed by CEDA | Open Government Licence v3.0 |
| Natural Earth 1:50m, version 5.1.1 | map outlines (`scripts/eval/prep_map_geometry.py`) | https://www.naturalearthdata.com | public domain |

The CAMELS-GB data carry the statements "© UK Centre for Ecology & Hydrology", "© University of Bristol", "Contains Environment
Agency information © Environment Agency and/or database right" and "Derived from UK bedrock hydrogeological mapping and UK
superficial productivity mapping, BGS © UKRI". HadUK-Grid contains public sector information licensed under the Open Government
Licence v3.0 (© Crown copyright, Met Office). Open Government Licence v3.0:
https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/

## Variables

Forcing and discharge are daily [days, catchments] float32 arrays in the catchment order of `data_lists/`; the attribute names
are in `static_names`. Missing discharge is NaN.

| | CAMELS-US | CAMELS-AUS | CAMELS-GB |
|---|---|---|---|
| period | 1980-2019 | 1980-2019 | 1988-12-03 .. 2019 |
| precipitation `P` (mm/d) | Daymet v4 | AGCD | HadUK-Grid |
| air temperature | `Tmax`, `Tmin` (deg C), Daymet v4 | `Tmax`, `Tmin` (deg C), AGCD | `Tm` (K), daily mean, CAMELS-GB v2 |
| PET `PET` (mm/d) | Priestley-Taylor (alpha 1.26) from Daymet v4 | SILO Morton wet-environment areal PET | CAMELS-GB `peti`, from 2015-10-01 `peti_chess` |
| discharge `q` (mm/d) | CAMELS v1.2 to 2014, NWIS 2015-2019 | `streamflow_mmd` | CAMELS-GB `discharge_spec`, from 2015-10-01 CAMELS-GB v2 |
| attributes | 27 (Kratzert et al. 2019) | 27, named as the US attributes where the quantity is the same | 21 |

The GB `Tm` is taken from CAMELS-GB v2. It differs slightly from the datesets being used in the paper, which calculates catchment mean from Chess-met using 5km grid.

## Held-out catchments

| dataset | gauged | held out |
|---|---|---|
| CAMELS-US | 603 | 68 (10 %), drawn at random within the 9 strata of terciles of snow fraction x aridity, round(10 %) per stratum, `numpy.random.default_rng(0)` |
| CAMELS-AUS | 505 | 56 (10 %), drawn the same way within terciles of aridity x mean elevation; 406213, 406214 and 407230 are never drawn |
| CAMELS-GB | 602 | 67, most of which have short discharge records before 2009 (64 with test data) |

The lists are in `data_lists/`. `scripts/data/README.md` describes how to rebuild the US and AUS files.
