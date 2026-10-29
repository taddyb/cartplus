# Rebuilding the released input files

Use the data environment (`environment/requirements-dataext.txt`) and run each script with `python -I` from a directory outside the
downloads.

**CAMELS-US.** Set the data root with `CAMELS_US_ROOT` (default `./camels_us`): the CAMELS v1.2 files
(`basin_timeseries_v1p2_metForcing_obsFlow.zip`, `basin_set_full_res.zip`, `camels_*.txt`) in `raw/`, outputs in `processed/`.

1. `scripts/data/us/camels_us_prep671.py` writes the layout template `processed/camels_us_daymet671.npz` (CAMELS v1.2 Daymet basin
   means 1980-2014, discharge in mm/d, Priestley-Taylor PET, the 27 attributes) and `processed/elev_bands671.npz`.
2. `scripts/data/us/dl_nwis.sh <ids.txt> <nwis_dir>` downloads the NWIS daily discharge 2010-10-01 .. 2019-12-31 (`dl_nwis_one.sh`
   retries one site).
3. `scripts/data/us/daymet_basin_means.py <raw_dir> <work_dir> [y0 y1]` computes the Daymet v4 basin means from the Microsoft
   Planetary Computer Zarr copy of Daymet v4 (`<raw_dir>` holds its `zmetadata.json`); run it for the default years 2010-2019 and
   for 1980-2009.
4. `scripts/data/us/build_ext2019.py <nwis_dir> <work_dir> <log_dir>` parses the NWIS files, writes the 2015-2019 discharge
   (`processed/camels_us_ext2019_q_qualifiers.npz`) and writes its log and overlap tables to `<log_dir>`.
5. `scripts/data/us/build_v4full.py <work_dir> scripts/data/us` writes `processed/camels_us_daymet671_v4full.npz` (it imports helpers
   from `build_v4ovl.py` in the folder given as second argument and writes its log and comparison tables there).
6. `scripts/data/us/draw_ungauged_10pct.py processed/camels_us_daymet671.npz <out.txt>` draws the held-out list.

**CAMELS-AUS.** Download `03_streamflow.zip`, `05_hydrometeorology.zip` and `CAMELS_AUS_Attributes&Indices_MasterTable.csv` of Zenodo
record 14289037, then

```
unzip 03_streamflow.zip -d <ex>; unzip 05_hydrometeorology.zip -d <ex>
python -I scripts/data/aus/build_camels_aus_v2.py <ex> "<dir>/CAMELS_AUS_Attributes&Indices_MasterTable.csv" <out_dir>
```

writes `camels_aus_v2.npz`, `elev_bands_aus.npz` and `aus_ungauged_10pct.txt` (about 15 s, 0.9 GB of memory).

A rebuilt `.npz` has the same arrays as the released file; its file checksum can differ through the timestamps inside the archive.
