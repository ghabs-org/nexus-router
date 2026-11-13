"""Bounded task classification without paid inference or prompt persistence."""
from functools import lru_cache
import math
import re
import threading

_LOCK = threading.Lock()
# These are conservative multilingual indicators, not calibrated probabilities.
_DEEP = re.compile(r'\b(architecture|architettura|architektur|arquitectura|root cause|causa radice|causa raíz|ursache|conflicting|contradictory|contraddittori\w*|contradicto\w*|widersprüch\w*|trade.off|due diligence|risk assessment|valut\w* (?:il |i |dei )?risch\w*|évalu\w* les risques|evalu\w* (?:los )?riesgos)\b', re.I)
_UTILITY = re.compile(r'^\s*(rewrite|rephrase|translate|extract|format|riscrivi|traduci|estrai|formatta|traduire|traduis|reformule|extraire|extrais|übersetze|extrahiere|umformuliere|traduce|traducir|reescribe|extrae)\b', re.I)
_SUMMARY = re.compile(r'\b(summar\w*|riassum\w*|résum\w*|resum\w*|zusammenfass\w*)\b', re.I)

@lru_cache(maxsize=1)
def _local_model():
    from .local_classifier import get_local_classifier
    return get_local_classifier()

def local_prediction(text):
    from .types import PreSignals
    with _LOCK:
        model = _local_model()
        if not model.available:
            return None
        result = model.classify(text, PreSignals(message_length=len(text)))
    if result is None:
        return None
    return {'task_type': result.classifier.task_type, 'complexity': result.classifier.complexity,
            'confidence': result.confidence, 'margin': result.margin}

class TaskClassifier:
    def __init__(self, settings, profiles, default):
        self.settings, self.profiles, self.default = settings, profiles, default
        if set(settings) - {'mode','use_local','min_confidence','min_margin','task_profiles'}:
            raise ValueError('Unknown classifier setting')
        if type(settings.get('use_local', False)) is not bool:
            raise ValueError('classifier.use_local must be boolean')
        for key, fallback in [('min_confidence', .8), ('min_margin', .15)]:
            value = settings.get(key, fallback)
            if type(value) not in {int,float} or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'Invalid classifier {key}')
        if settings.get('mode','hybrid') not in {'hybrid','rules'}:
            raise ValueError('classifier.mode must be hybrid or rules')
        self.mapping = {'fast_utility':'utility','reasoning':'deep','coding':'standard','code_review':'standard',
                        'summarization':'standard','long_context':'standard','vision':'standard','general_chat':default}
        self.mapping = {key: (value if value in profiles else default) for key,value in self.mapping.items()}
        self.mapping.update(settings.get('task_profiles',{}))
        if any(value not in profiles for value in self.mapping.values()):
            raise ValueError('Classifier task profile is absent from policy')

    def choose(self, instruction, rules):
        text = instruction[:2000]
        result = {'source':'default','task_type':'unknown','complexity':'unknown','confidence':None,
                  'margin':None,'fallback_reason':'ambiguous_instruction','calibrated':False}
        if self.settings.get('mode','hybrid') == 'hybrid' and _DEEP.search(text):
            result.update(source='multilingual_indicators',task_type='reasoning',complexity='high',fallback_reason=None)
            return self.mapping['reasoning'], result
        # Administrator rules remain explicit policy decisions, never confidence estimates.
        for index,(pattern,profile) in enumerate(rules):
            if pattern.search(text):
                result.update(source='admin_rule',rule_index=index,fallback_reason=None)
                return profile,result
        if self.settings.get('use_local',False) and self.settings.get('mode','hybrid') == 'hybrid':
            try:
                prediction = local_prediction(text)
            except Exception:
                prediction = None
            if prediction:
                confidence, margin = prediction.get('confidence'), prediction.get('margin')
                valid = all(type(v) in {int,float} and math.isfinite(v) and 0 <= v <= 1 for v in (confidence,margin))
                if valid and prediction.get('task_type') in self.mapping:
                    result.update(source='local_model', **prediction, fallback_reason='low_confidence')
                    if confidence >= self.settings.get('min_confidence',.8) and margin >= self.settings.get('min_margin',.15):
                        result['fallback_reason']=None
                        return self.mapping[prediction['task_type']], result
                    return self.default,result
            # Failure is conservative: do not turn missing model artifacts into a cheap route.
            result.update(source='local_model_unavailable',fallback_reason='local_model_unavailable')
            return self.default,result
        if self.settings.get('mode','hybrid') == 'hybrid':
            if _UTILITY.search(text) and len(text) <= 600:
                result.update(source='multilingual_indicators',task_type='fast_utility',complexity='low',fallback_reason=None)
                return self.mapping['fast_utility'], result
            if _SUMMARY.search(text):
                result.update(source='multilingual_indicators',task_type='summarization',complexity='medium',fallback_reason=None)
                return self.mapping['summarization'], result
        return self.default,result
