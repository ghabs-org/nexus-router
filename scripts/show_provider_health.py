#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('/home/ubuntu/.local/state/nexus-router/data/router.sqlite')
conn.row_factory = sqlite3.Row
for r in conn.execute('SELECT provider, auth, quota, quota_remaining_ratio, latency_ms_p50, latency_updated_at, last_check_at, health_score FROM provider_health_state ORDER BY provider'):
    print(dict(r))
conn.close()
