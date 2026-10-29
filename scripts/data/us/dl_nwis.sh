#!/bin/bash
# Download USGS NWIS daily mean discharge (00060, statCd 00003) 2010-10-01..2019-12-31 for the 671 CAMELS gauges (RDB, one file per site).
# Usage: dl_nwis.sh <ids.txt> <outdir>   (generic user agent; no personal data sent)
IDS=$1; OUT=$2; mkdir -p "$OUT"
while read -r s; do
  f="$OUT/$s.rdb"; [ -s "$f" ] && grep -q "^USGS" "$f" && continue
  for k in 1 2 3; do
    curl -sS -A "python-data-client" --max-time 120 -o "$f" \
      "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=$s&parameterCd=00060&statCd=00003&startDT=2010-10-01&endDT=2019-12-31&siteStatus=all" && break
    sleep 5
  done
  sleep 0.3
done < "$IDS"
echo done
