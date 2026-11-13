"""One-shot, explicit escalation tickets. No automatic replay of prior actions."""
from contextlib import closing
import json
import time
import uuid
from . import db
from .gateway_policy import PolicyError

class Escalations:
    def issue(self,principal,source,wire):
        with closing(db._connect()) as conn,conn:
            conn.execute('BEGIN IMMEDIATE')
            pin=conn.execute('SELECT profile,principal FROM gateway_task_pins WHERE session=?',(source,)).fetchone()
            if not pin or pin['principal']!=principal:raise PolicyError('Task not found',404)
            if pin['profile'] not in {'utility','standard'}:raise PolicyError('Task already uses the highest escalation profile',409)
            explanation=conn.execute('SELECT explanation_json FROM gateway_task_explanations WHERE session=?',(source,)).fetchone()
            if not explanation or json.loads(explanation[0]).get('wire')!=wire:
                raise PolicyError('Escalation must use the source task protocol',409)
            parent=conn.execute('SELECT root,level FROM gateway_escalation_tickets WHERE child_session=? AND principal=?',(source,principal)).fetchone()
            root,level=(parent['root'],parent['level']+1) if parent else (source,1)
            if level>2:raise PolicyError('Escalation limit reached',409)
            prior=conn.execute('SELECT * FROM gateway_escalation_tickets WHERE principal=? AND root=? AND level=?',(principal,root,level)).fetchone()
            if prior:
                if prior['state']!='pending' or time.time()-prior['created_at']>600:raise PolicyError('Escalation step already issued or expired',409)
                return dict(prior)
            ticket,new_session=uuid.uuid4().hex,uuid.uuid4().hex
            profile='standard' if pin['profile']=='utility' else 'deep'
            conn.execute('INSERT INTO gateway_escalation_tickets VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                         (ticket,principal,root,level,source,new_session,None,profile,wire,'pending',time.time()))
            return dict(conn.execute('SELECT * FROM gateway_escalation_tickets WHERE ticket=?',(ticket,)).fetchone())

    def read(self,ticket,principal,wire,body,headers):
        with closing(db._connect()) as conn:
            row=conn.execute('SELECT * FROM gateway_escalation_tickets WHERE ticket=? AND principal=?',(ticket,principal)).fetchone()
        if not row:raise PolicyError('Escalation ticket not found',404)
        row=dict(row)
        if row['state']!='pending' or time.time()-row['created_at']>600:raise PolicyError('Escalation ticket used or expired',409)
        if row['wire']!=wire or headers.get('x-session-id')!=row['new_session_id']:
            raise PolicyError('Escalation requires its new session ID and original protocol',409)
        if body.get('previous_response_id'):raise PolicyError('Escalation cannot replay a server-side continuation',409)
        for item in body.get('messages',[]):
            if item.get('role') not in {'user','system','developer'}:raise PolicyError('Submit a fresh task without prior assistant/tool actions',409)
        for item in body.get('input',[]) if isinstance(body.get('input'),list) else []:
            if not isinstance(item,dict) or item.get('role') not in {'user','system','developer'} or item.get('type','message')!='message':
                raise PolicyError('Escalation requires fresh user input',409)
        return row

    def feedback(self,principal,session,rating):
        if rating not in {'good','poor_quality','too_expensive'}:raise PolicyError('Invalid feedback rating')
        with closing(db._connect()) as conn,conn:
            pin=conn.execute('SELECT principal FROM gateway_task_pins WHERE session=?',(session,)).fetchone()
            if not pin or pin['principal']!=principal:raise PolicyError('Task not found',404)
            conn.execute('INSERT INTO gateway_feedback (principal,session,rating,created_at) VALUES (?,?,?,?)',(principal,session,rating,time.time()))

    def claim(self,ticket,principal,child_session):
        with closing(db._connect()) as conn,conn:
            cursor=conn.execute("UPDATE gateway_escalation_tickets SET state='used',child_session=? WHERE ticket=? AND principal=? AND state='pending' AND created_at>=?",
                                (child_session,ticket,principal,time.time()-600))
            if cursor.rowcount!=1:raise PolicyError('Escalation ticket already consumed or expired',409)
