#!/bin/bash
# one site: dl_nwis_one.sh <site> <outdir>; writes to .part then renames (atomic)
s=$1; OUT=$2; f="$OUT/$s.rdb"; [ -s "$f" ] && exit 0
for k in 1 2 3 4; do
  curl -sS -A "python-data-client" --max-time 300 -o "$f.part" \
    "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=$s&parameterCd=00060&statCd=00003&startDT=2010-10-01&endDT=2019-12-31&siteStatus=all" \
    && grep -q "^agency_cd\|No sites found" "$f.part" && mv "$f.part" "$f" && exit 0
  sleep $((5*k))
done
echo "FAILED $s"
