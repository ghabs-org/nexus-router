import math
import pytest
from src.gateway_classifier import TaskClassifier
from test_gateway import harness,payload

PROFILES={'utility':[], 'standard':[], 'deep':[]}
@pytest.mark.parametrize('text,profile',[
 ('Übersetze diese Nachricht','utility'),('Traduis cette phrase','utility'),('Traduce este correo','utility'),
 ('Analysiere die Architektur','deep'),('Évalue les risques','deep'),('Analiza la arquitectura','deep'),
 ('Valuta i rischi','deep'),('Résume ce rapport','standard'),('Help me with this','standard'),
 ('Translate this root cause analysis','deep')])
def test_multilingual_indicators(text,profile):
    actual, explain=TaskClassifier({},PROFILES,'standard').choose(text,[])
    assert actual==profile
    assert explain['confidence'] is None
    assert text not in str(explain)

@pytest.mark.parametrize('prediction,expected,reason',[
 ({'task_type':'reasoning','complexity':'high','confidence':.92,'margin':.3},'deep',None),
 ({'task_type':'fast_utility','complexity':'low','confidence':.6,'margin':.2},'standard','low_confidence'),
 ({'task_type':'fast_utility','complexity':'low','confidence':.99,'margin':.01},'standard','low_confidence'),
 ({'task_type':'unknown','complexity':'low','confidence':.99,'margin':.5},'standard','local_model_unavailable'),
 ({'task_type':'fast_utility','complexity':'low','confidence':math.nan,'margin':.5},'standard','local_model_unavailable'),
 (None,'standard','local_model_unavailable')])
def test_local_model_gates(monkeypatch,prediction,expected,reason):
    monkeypatch.setattr('src.gateway_classifier.local_prediction',lambda text:prediction)
    profile,explain=TaskClassifier({'use_local':True},PROFILES,'standard').choose('Please process this',[])
    assert profile==expected and explain['fallback_reason']==reason

def test_local_failure_is_safe(monkeypatch):
    def fail(text):raise RuntimeError('private path must not be persisted')
    monkeypatch.setattr('src.gateway_classifier.local_prediction',fail)
    profile,explain=TaskClassifier({'use_local':True},PROFILES,'standard').choose('hello',[])
    assert profile=='standard' and 'private' not in str(explain)

def test_explanation_persistence_and_pin(harness):
    client,server,upstream,config=harness
    body=payload('Übersetze diese Nachricht')
    preview=client.post('/v1/nexus/route-preview',json=body).json()
    assert preview['profile']=='utility'
    assert preview['explanation']['source']=='multilingual_indicators'
    headers={'x-session-id':'explain-task'}
    assert client.post('/v1/messages',json=body,headers=headers).status_code==200
    assert client.post('/v1/messages',json=payload('Analyze architecture'),headers=headers).status_code==200
    status=client.get('/v1/nexus/status').json()
    explain=status['recent_turns'][0]['explanation']
    assert explain['pin_reused'] and explain['source']=='multilingual_indicators'
    assert 'Übersetze' not in str(status)

from test_gateway_clients import native

def test_native_preview_explains_without_inference(native):
    client,server,upstream=native
    response=client.post('/v1/nexus/route-preview?wire=responses',json={'model':'nexus-auto','input':'Übersetze diese Nachricht'})
    assert response.status_code==200,response.text
    assert response.json()['profile']=='utility'
    assert response.json()['explanation']['source']=='multilingual_indicators'
    assert not upstream.requests

from src.gateway_benchmark import compare_outcomes

def test_outcomes_comparison_exposes_quality_loss():
    records=[{'case_id':'a','strategy':strategy,'cost_usd':cost,'quality_score':quality,'success':success,
              'latency_ms':100,'cost_basis':'estimated'}
             for strategy,cost,quality,success in [('baseline',1,1,True),('routed',.5,.8,False)]]
    result=compare_outcomes(records)
    assert result['cost_reduction_fraction']==.5
    assert result['quality_delta']==pytest.approx(-.2)
    assert result['success_rate_delta']==-1
    assert result['routed']['cost_per_successful_task_usd'] is None
    with pytest.raises(ValueError):compare_outcomes(records[:1])
    with pytest.raises(ValueError):compare_outcomes(records+records)
    records[1]['cost_basis']='billed'
    with pytest.raises(ValueError):compare_outcomes(records)

def test_rules_mode_and_settings_validation(monkeypatch):
    def fail(text):raise AssertionError('Rules mode must not run local inference')
    monkeypatch.setattr('src.gateway_classifier.local_prediction',fail)
    profile,explain=TaskClassifier({'mode':'rules','use_local':True},PROFILES,'standard').choose('Analyze architecture',[])
    assert profile=='standard' and explain['source']=='default'
    for settings in ({'use_local':'false'},{'min_confidence':math.nan},{'min_margin':-1},{'unsupported':True}):
        with pytest.raises(ValueError):TaskClassifier(settings,PROFILES,'standard')
