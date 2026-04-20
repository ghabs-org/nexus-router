#!/usr/bin/env python3
import sqlite3

DB = '/home/ubuntu/.local/state/nexus-router/data/router.sqlite'

conn = sqlite3.connect(DB)
conn.execute("UPDATE provider_health_state SET latency_ms_p50=NULL, latency_updated_at=NULL WHERE latency_updated_at IS NULL")
conn.commit()
for row in conn.execute("SELECT provider, latency_ms_p50, latency_updated_at, last_check_at FROM provider_health_state ORDER BY provider"):
    print(row)
conn.close()
