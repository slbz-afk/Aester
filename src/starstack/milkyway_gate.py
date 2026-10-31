"""Conservative declaration gate for visible Milky Way presentation.

A nonzero editing mask and a successful layer reconstruction do not establish
that a visible Milky Way subject is present. This gate performs no image or
astronomical classification. Explicit input-bound drawing instructions may
enable presentation, with human/agent provenance kept distinct; unvalidated
automatic classification and legacy unknown masks remain disabled.

``human-review`` is the caller's recorded assertion, not an authenticated
reviewer or a guarantee of physical truth. Hashes bind that assertion to the
source and mask; they are not signatures. Both unknown and absent decisions
must bypass pixel editing in the calling pipeline. Star separation and star
reduction are separate capabilities and do not require a visible Milky Way.
"""

from copy import deepcopy
import re


VERSION = 'milkyway-subject-evidence-1'
_FIELDS = frozenset((
    'version', 'status', 'method', 'reviewer', 'rationale',
    'source_sha256', 'milkyway_mask_sha256',
))
_STATUSES = frozenset(('present', 'absent', 'unknown'))
_METHODS = frozenset(('human-review', 'unvalidated-auto', 'explicit-region-human', 'explicit-region-agent'))


def _digest(value, name):
    if type(value) is not str or re.fullmatch(r'[0-9a-fA-F]{64}', value) is None:
        raise ValueError(f'{name} must be a 64-character hexadecimal SHA256')
    return value.lower()


def _text(value, name):
    if type(value) is not str or not value.strip():
        raise ValueError(f'{name} must be a nonempty string')
    try:
        value.encode('utf-8')
    except UnicodeEncodeError as exc:
        raise ValueError(f'{name} must contain valid Unicode text') from exc
    return value.strip()


def assess_evidence(evidence: dict | None, *, source_sha256: str,
                    mask_sha256: str) -> dict:
    """Return a JSON-safe, detached gate decision bound to two input digests.

    Only the seven documented evidence fields are accepted. Missing evidence
    means ``unknown`` and needs review; it never enables the operation.
    Unvalidated automatic claims of either presence OR absence resolve to
    unknown. Deliberate drawing instructions have their own methods, never a
    fabricated human review. The original claim remains in ``evidence``.

    This function authorizes no writes itself. Callers must enforce an exact
    pixel bypass when ``enabled`` is false and record the gate separately
    from operation success. Invalid evidence raises ``ValueError`` rather
    than silently discarding an incompatible or differently bound assertion.
    """
    source_hash = _digest(source_sha256, 'source_sha256')
    mask_hash = _digest(mask_sha256, 'mask_sha256')
    if evidence is None:
        return {
            'enabled': False,
            'status': 'unknown',
            'decision': 'needs_review',
            'reason': '未提供当前源图及蒙版的银河可见性确认，跳过银河增强；非零蒙版不能证明目标存在。',
            'evidence': None,
        }
    if (type(evidence) is not dict or any(type(key) is not str for key in evidence)
            or set(evidence) != _FIELDS):
        raise ValueError('evidence must be a dict containing exactly the documented fields')
    if type(evidence['version']) is not str or evidence['version'] != VERSION:
        raise ValueError('unsupported Milky Way evidence version')
    if type(evidence['status']) is not str or evidence['status'] not in _STATUSES:
        raise ValueError('evidence status must be present, absent, or unknown')
    if type(evidence['method']) is not str or evidence['method'] not in _METHODS:
        raise ValueError('unsupported Milky Way evidence method')
    # Construct a new, strictly typed object. In particular, no opaque extra
    # values, NaNs, mutable collections, or caller-defined object types survive.
    checked = {
        'version': VERSION,
        'status': evidence['status'],
        'method': evidence['method'],
        'reviewer': _text(evidence['reviewer'], 'reviewer'),
        'rationale': _text(evidence['rationale'], 'rationale'),
        'source_sha256': _digest(evidence['source_sha256'], 'evidence source_sha256'),
        'milkyway_mask_sha256': _digest(evidence['milkyway_mask_sha256'], 'evidence milkyway_mask_sha256'),
    }
    if checked['source_sha256'] != source_hash or checked['milkyway_mask_sha256'] != mask_hash:
        raise ValueError('Milky Way evidence is bound to a different source or mask')
    if checked['method'] == 'unvalidated-auto':
        status, decision, enabled = 'unknown', 'needs_review', False
        reason = '自动判定方法尚未验证，不能确认银河可见或不可见；跳过银河增强，等待人工复核。'
    elif checked['status'] == 'present':
        status, decision, enabled = 'present', 'apply', True
        reason = ('当前源图及蒙版已有显式银河绘制；按所画范围增强，操作者来源保留，此编辑不代表自动识别或物理真值。'
                  if checked['method'].startswith('explicit-region-') else
                  '当前源图及蒙版已有人工可见银河确认；允许局部增强，此声明不等于自动识别或物理真值。')
    elif checked['status'] == 'absent':
        status, decision, enabled = 'absent', 'skipped', False
        reason = '人工确认当前素材中没有可供增强的可见银河结构，跳过银河增强。'
    else:
        status, decision, enabled = 'unknown', 'needs_review', False
        reason = '人工复核尚不能确认可见银河结构，跳过银河增强；未知不代表不存在。'
    return {
        'enabled': enabled,
        'status': status,
        'decision': decision,
        'reason': reason,
        'evidence': deepcopy(checked),
    }
