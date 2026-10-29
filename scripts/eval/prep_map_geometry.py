"""Project and simplify the map geometry of Figures fig:map_gb / fig:map_us / fig:map_aus once (needs pyproj + shapely + pyshp: the dataext
environment, environment/requirements-dataext.txt; run with python -I). GB: CAMELS-GB catchment boundaries (British National Grid, kept)
and the Natural Earth 1:50m outlines of Great Britain, Ireland and the Isle of Man projected to BNG. US: CAMELS (HCDN) basin polygons and
the Natural Earth lower-48 state outlines, projected to CONUS Albers equal area (EPSG:5070). AUS: CAMELS-AUS v2 adopted boundaries (WGS84)
and the Natural Earth 1:50m Australian state outlines (mainland, Tasmania and near-shore islands; remote islands dropped), projected to
Australian Albers (EPSG:3577); cat_id = the running npz id ("9xxxxxxx" as int) of the station, so fig_maps.py matches polygons to the
camels_aus_v2.npz columns. Output: <out>/geom_{gb,us,aus}.npz with rings as float32 arrays plus ring -> id maps (default
<repository>/results/data/maps, where fig_maps.py looks for them). Inputs (DATA.md):
  --gb_boundaries  CAMELS-GB data/boundaries/CAMELS_GB_catchment_boundaries.shp (EIDC download of CAMELS-GB)
  --us_basins      CAMELS-US basin_set_full_res/HCDN_nhru_final_671.shp (basin_set_full_res.zip of CAMELS v1.2)
  --ne_countries   Natural Earth 1:50m Admin 0 countries, ne_50m_admin_0_countries.shp (version 5.1.1)
  --ne_states      Natural Earth 1:50m Admin 1 states/provinces with large lakes, ne_50m_admin_1_states_provinces_lakes.shp (5.1.1)
  --aus_boundaries CAMELS-AUS v2 02_location_boundary_area/shp/CAMELS_AUS_v2_Boundaries_adopted.shp (02_location_boundary_area.zip)
  --aus_npz        camels_aus_v2.npz (scripts/data/aus/build_camels_aus_v2.py; station_ids -> npz ids)
usage: python -I scripts/eval/prep_map_geometry.py [all] --gb_boundaries ... --us_basins ... --ne_countries ... --ne_states ... [--out DIR]
       python -I scripts/eval/prep_map_geometry.py aus --aus_boundaries ... --aus_npz ... --ne_states ... [--out DIR]
(all = gb and us; aus only when asked)
"""
import argparse, os, sys, numpy as np, shapefile
from shapely.geometry import shape, mapping, MultiPolygon
from shapely.ops import transform
from pyproj import Transformer
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("what", nargs="?", default="all", choices=["all", "aus"])
for _k in ("gb_boundaries", "us_basins", "ne_countries", "ne_states", "aus_boundaries", "aus_npz"):
    _ap.add_argument("--" + _k, default="")
_ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "data", "maps"))
A = _ap.parse_args()
_need = ("gb_boundaries", "us_basins", "ne_countries", "ne_states") if A.what == "all" else ("aus_boundaries", "aus_npz", "ne_states")
_miss = [k for k in _need if not getattr(A, k)]
if _miss: _ap.error(f"{A.what} needs " + ", ".join("--" + k for k in _miss))
OUT = A.out.rstrip("/") + "/"
os.makedirs(OUT, exist_ok=True)
WHAT = "aus" if A.what == "aus" else "gb+us"


def rings(geom):
    polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
    return [np.asarray(p.exterior.coords, np.float32) for p in polys if p.geom_type == "Polygon" and not p.is_empty]


def pack(items):
    """items: list of (key, [ring arrays]) -> flat coordinates, ring offsets, ring keys"""
    xy, off, key = [], [0], []
    for k, rs in items:
        for r in rs: xy.append(r); off.append(off[-1] + len(r)); key.append(k)
    return np.concatenate(xy), np.asarray(off, np.int64), np.asarray(key)


# ---- GB
if WHAT != "aus":
    t_gb = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True)
    r = shapefile.Reader(A.gb_boundaries)
    fn = [f[0] for f in r.fields[1:]]; ci = fn.index("ID_STRING")
    cat = []
    for sr in r.iterShapeRecords():
        g = shape(sr.shape.__geo_interface__).buffer(0).simplify(100); cat.append((int(float(str(sr.record[ci]).strip())), rings(g)))
    c = shapefile.Reader(A.ne_countries); cfn = [f[0] for f in c.fields[1:]]; ia = cfn.index("ADM0_A3")
    land = []
    for sr in c.iterShapeRecords():
        if sr.record[ia] in ("GBR", "IRL", "IMN"):
            g = transform(t_gb.transform, shape(sr.shape.__geo_interface__)).simplify(200)
            land.append((sr.record[ia], [x for x in rings(g) if len(x) > 3]))
    a, o, k = pack(cat); b, p, q = pack(land)
    np.savez_compressed(OUT + "geom_gb.npz", cat_xy=a, cat_off=o, cat_id=k, land_xy=b, land_off=p, land_key=q)
    print("gb", len(cat), "catchments,", len(o) - 1, "rings; land rings", len(p) - 1)
# ---- US
if WHAT != "aus":
    t_us = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True); t_ll = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    r = shapefile.Reader(A.us_basins)
    fn = [f[0] for f in r.fields[1:]]; ci = fn.index("hru_id")
    bas = []
    for sr in r.iterShapeRecords():
        g = transform(t_us.transform, shape(sr.shape.__geo_interface__).buffer(0)).simplify(500); bas.append((int(sr.record[ci]), rings(g)))
    s = shapefile.Reader(A.ne_states); sfn = [f[0] for f in s.fields[1:]]
    ia, ic = sfn.index("adm0_a3"), sfn.index("postal")
    st = []
    for sr in s.iterShapeRecords():
        if sr.record[ia] == "USA" and sr.record[ic] not in ("AK", "HI"):
            g = transform(t_ll.transform, shape(sr.shape.__geo_interface__)).simplify(1000); st.append((sr.record[ic], rings(g)))
    a, o, k = pack(bas); b, p, q = pack(st)
    np.savez_compressed(OUT + "geom_us.npz", cat_xy=a, cat_off=o, cat_id=k, land_xy=b, land_off=p, land_key=q)
    print("us", len(bas), "basins,", len(o) - 1, "rings; state rings", len(p) - 1)
# ---- AUS (only when asked: python -I prep_map_geometry.py aus --aus_boundaries ... --aus_npz ... --ne_states ...)
if WHAT == "aus":
    t_au = Transformer.from_crs("EPSG:4326", "EPSG:3577", always_xy=True)
    z = np.load(A.aus_npz, allow_pickle=False); sid = {str(s): int(i) for s, i in zip(z["station_ids"], z["ids"])}
    r = shapefile.Reader(A.aus_boundaries)
    fn = [f[0] for f in r.fields[1:]]; ci = fn.index("CatchID")
    cat, miss = [], []
    for sr in r.iterShapeRecords():
        k = str(sr.record[ci]).strip()
        if k not in sid: miss.append(k); continue
        g = transform(t_au.transform, shape(sr.shape.__geo_interface__).buffer(0)).simplify(500)
        if g.geom_type == "MultiPolygon": g = MultiPolygon([x for x in g.geoms if x.area >= 0.005 * max(y.area for y in g.geoms)])   # drop digitising slivers
        cat.append((sid[k], rings(g)))
    s = shapefile.Reader(A.ne_states); sfn = [f[0] for f in s.fields[1:]]
    ia, ic = sfn.index("adm0_a3"), sfn.index("name")
    st = []
    for sr in s.iterShapeRecords():
        if sr.record[ia] != "AUS": continue
        g = shape(sr.shape.__geo_interface__)
        ps = [p for p in ([g] if g.geom_type == "Polygon" else list(g.geoms))
              if 112.5 <= p.centroid.x <= 154.0 and -44.0 <= p.centroid.y <= -10.0]          # drops Lord Howe, Macquarie and the Coral Sea islets
        for p in ps:
            pp = transform(t_au.transform, p).simplify(1000); st.append((sr.record[ic], [x for x in rings(pp) if len(x) > 3]))
    a, o, k = pack(cat); b, p, q = pack(st)
    np.savez_compressed(OUT + "geom_aus.npz", cat_xy=a, cat_off=o, cat_id=k, land_xy=b, land_off=p, land_key=q)
    print("aus", len(cat), "catchments,", len(o) - 1, "rings; state rings", len(p) - 1, "; boundary ids not in npz:", miss,
          "; npz stations without boundary:", sorted(set(sid) - {str(x) for x in [rr.record[ci].strip() for rr in r.iterShapeRecords()]}))
