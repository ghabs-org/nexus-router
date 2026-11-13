"""SQLite reservations for conservative, tariff-based gateway spending limits.

All amounts are integer micro-USD. Unknown attempts keep their full reservation.
"""
from contextlib import closing
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import math
import os
import time
from . import db
from .gateway_policy import PolicyError

COST_LIMITS={'max_cost_usd_24h','max_task_cost_usd'}

def micros(value):
    if type(value) not in {int,float} or not math.isfinite(value) or not 0 <= value <= 1_000_000_000:
        raise ValueError('Cost must be finite, nonnegative, and at most one billion USD')
    return int((Decimal(str(value))*1_000_000).to_integral_value(rounding=ROUND_CEILING))

class BudgetLedger:
    def reserve(self, classification, limits, config, normalized):
        if not COST_LIMITS.intersection(limits):return False
        # Paid server tools and opaque native continuations have no locally provable
        # per-call cost ceiling. Require ordinary client-executed function tools.
        for tool in normalized.get('tools',[]):
            if tool.get('type','custom') not in {'custom','function'}:
                raise PolicyError('Strict budgets do not allow paid or opaque server tools')
        if normalized.get("safeguards"):
            raise PolicyError("Strict budgets require client-side permission checks, not opaque server checks")
        amount=Decimal(0)
        for choice in classification.choices:
            model=config['models'][choice['target']]
            prices=model.get('pricing')
            context=model.get('context_window')
            if not prices or type(context) is not int or context<1:
                raise PolicyError('Strict budgets require tariffs and a verified context ceiling for every fallback')
            output=min(normalized.get('max_tokens',config.get('max_output_tokens',16384)),
                       model['max_output_tokens'],config.get('max_output_tokens',model['max_output_tokens']))
            # USD per million tokens multiplied by tokens equals micro-USD.
            input_rate=max(Decimal(str(prices[k])) for k in ('input','cache_read','cache_write_5m','cache_write_1h'))
            amount+=context*input_rate+output*Decimal(str(prices['output']))
        reserve=int(amount.to_integral_value(rounding=ROUND_CEILING))
        budget_session=classification.explanation.get("budget_root",classification.session)
        now=time.time()
        with closing(db._connect()) as conn,conn:
            conn.execute('BEGIN IMMEDIATE')
            # Older turns predate this ledger and cannot be treated as free credit.
            legacy=conn.execute('SELECT COUNT(*) FROM gateway_turns t LEFT JOIN gateway_budget_turns b ON b.turn_id=t.id '
                                'WHERE t.principal=? AND b.turn_id IS NULL AND (t.created_at>=? OR t.session=?)',
                                (classification.principal,now-86400,budget_session)).fetchone()[0]
            if legacy:raise PolicyError('Strict budget has legacy usage; start with a reconciled budget database',402)
            rows=conn.execute('SELECT session,created_at,finished_at,held_microusd,state FROM gateway_budget_reservations WHERE principal=?',
                              (classification.principal,)).fetchall()
            # Unresolved obligations never expire into free spending capacity.
            window=sum(r['held_microusd'] for r in rows if (r['finished_at'] or r['created_at'])>=now-86400 or r['state'] in {'reserved','uncertain','breach'})
            task=sum(r['held_microusd'] for r in rows if r['session']==budget_session)
            if 'max_cost_usd_24h' in limits and window+reserve>int((Decimal(str(limits['max_cost_usd_24h']))*1_000_000).to_integral_value(rounding=ROUND_FLOOR)):
                raise PolicyError('Your spending allowance cannot cover this request reservation',402)
            if 'max_task_cost_usd' in limits and task+reserve>int((Decimal(str(limits['max_task_cost_usd']))*1_000_000).to_integral_value(rounding=ROUND_FLOOR)):
                raise PolicyError('This task cannot cover another request reservation',402)
            if any(r['state']=='breach' for r in rows):
                raise PolicyError('Provider usage exceeded its declared ceiling; administrator reconciliation required',402)
            if sum(r['state']=='reserved' for r in rows)>=limits.get('max_concurrent',4):
                raise PolicyError('Too many reserved requests; reconcile any crashed request',429)
            conn.execute('INSERT INTO gateway_budget_reservations VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                         (classification.request_id,classification.principal,budget_session,now,reserve,reserve,
                          'reserved',0,os.getpid(),None,None))
        return True

    def attempt(self, request_id):
        with closing(db._connect()) as conn,conn:
            conn.execute("UPDATE gateway_budget_reservations SET attempts=attempts+1 WHERE request_id=? AND state='reserved'",(request_id,))

    def finish(self, request_id, turn_id=None, cost=None, complete=False):
        with closing(db._connect()) as conn,conn:
            row=conn.execute('SELECT * FROM gateway_budget_reservations WHERE request_id=?',(request_id,)).fetchone()
            if not row:return
            if turn_id is not None:
                conn.execute('INSERT OR IGNORE INTO gateway_budget_turns VALUES (?,?)',(turn_id,request_id))
            if row['state']!='reserved':return
            held,state=row['reserved_microusd'],'uncertain'
            if row['attempts']==0:held,state=0,'released'
            elif row['attempts']==1 and complete and cost is not None:
                held=micros(cost)
                state='settled' if held<=row['reserved_microusd'] else 'breach'
            conn.execute('UPDATE gateway_budget_reservations SET held_microusd=?,state=?,finished_at=? WHERE request_id=?',
                         (held,state,time.time(),request_id))

    def reconcile(self, request_id, actual_cost_usd, note):
        amount=micros(actual_cost_usd)
        if not isinstance(note,str) or not 1<=len(note.strip())<=1000:raise ValueError('A reconciliation note is required')
        with closing(db._connect()) as conn,conn:
            conn.execute('BEGIN IMMEDIATE')
            row=conn.execute('SELECT * FROM gateway_budget_reservations WHERE request_id=?',(request_id,)).fetchone()
            if not row:raise ValueError('Reservation not found')
            if row['state']=='reserved':
                try:os.kill(row['owner_pid'],0)
                except ProcessLookupError:pass
                except PermissionError:raise ValueError('Cannot establish that reservation owner has stopped') from None
                else:raise ValueError('Reservation owner is alive; do not reconcile an active request')
            conn.execute('INSERT INTO gateway_budget_audit (request_id,created_at,previous_microusd,actual_microusd,note) VALUES (?,?,?,?,?)',
                         (request_id,time.time(),row['held_microusd'],amount,note))
            conn.execute("UPDATE gateway_budget_reservations SET held_microusd=?,state='reconciled',finished_at=? WHERE request_id=?",
                         (amount,time.time(),request_id))

    def status(self,principal):
        with closing(db._connect()) as conn:
            rows=[dict(r) for r in conn.execute('SELECT request_id,session,created_at,state,reserved_microusd,held_microusd,attempts '
                                               'FROM gateway_budget_reservations WHERE principal=? AND '
                                               "(COALESCE(finished_at,created_at)>=? OR state IN ('reserved','uncertain','breach')) ORDER BY created_at DESC",
                                               (principal,time.time()-86400))]
        return {'held_microusd':sum(r['held_microusd'] for r in rows),'reservations':rows}
