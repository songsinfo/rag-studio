import hashlib

import pytest

from app.evaluation.observability import build_sample, describe_metrics, stage_coverage, metric_coverage


def fixture_data():
    expected = {'evidenceGroups': [
        {'alternatives': [{'documentId': 'doc', 'versionId': 'v1', 'quote': 'gold'},
                          {'documentId': 'alt', 'versionId': 'v1', 'quote': 'alternative'}]},
        {'alternatives': [{'documentId': 'doc', 'versionId': 'v1', 'quote': 'second'}]}]}
    chunks = {('build', 'child'): {'documentId': 'doc', 'versionId': 'v1', 'text': 'prefix gold suffix'},
              ('build', 'parent'): {'documentId': 'doc', 'versionId': 'v1', 'text': 'prefix gold suffix second'}}
    item = {'documentId': 'doc', 'buildId': 'build', 'chunkId': 'child', 'parentChunkId': 'parent'}
    return expected, chunks, item


def test_stage_group_alternatives_duplicates_versions_and_empty_gold():
    expected, chunks, item = fixture_data()
    snapshot = {'stage': 'dense_recall', 'complete': True, 'items': [item, item]}
    result = stage_coverage(expected, snapshot, chunks, k=5)
    assert (result['numerator'], result['denominator'], result['value']) == (1, 2, .5)
    expected['evidenceGroups'][0]['alternatives'][0]['versionId'] = 'v2'
    assert stage_coverage(expected, snapshot, chunks)['value'] == 0
    assert stage_coverage({}, snapshot, chunks)['status'] == 'NOT_APPLICABLE'
    snapshot['complete'] = False
    result = stage_coverage(expected, snapshot, chunks)
    assert result['status'] == 'UNDETERMINED' and result['value'] is None


def test_parent_context_and_child_budget_fallback_use_actual_local_digest():
    expected, chunks, item = fixture_data()
    snapshot = {'stage': 'context_budget', 'complete': True, 'items': [item]}
    item['textSha256'] = hashlib.sha256(chunks['build', 'parent']['text'].encode()).hexdigest()
    assert stage_coverage(expected, snapshot, chunks)['value'] == 1
    item['textSha256'] = hashlib.sha256(chunks['build', 'child']['text'].encode()).hexdigest()
    assert stage_coverage(expected, snapshot, chunks)['value'] == .5
    item['textSha256'] = 'missing'
    assert stage_coverage(expected, snapshot, chunks)['status'] == 'UNDETERMINED'


def test_sent_range_cannot_recover_gold_from_removed_text_or_history():
    expected, chunks, item = fixture_data()
    item.update(contextChunkId='parent', contextStart=0, contextEnd=18, mappingStatus='EXACT')
    snapshot = {'stage': 'MODEL_CONTEXT', 'complete': False, 'historyMapping': 'UNKNOWN',
                'sourceRangesComplete': True, 'items': [item], 'callId': 'call'}
    result = stage_coverage(expected, snapshot, chunks)
    assert result['scope'] == 'SENT_STRUCTURED_DOCUMENTS' and result['value'] == .5
    item['contextEnd'] = 7
    assert stage_coverage(expected, snapshot, chunks)['value'] == 0
    item['mappingStatus'] = 'UNKNOWN'
    result = stage_coverage(expected, snapshot, chunks)
    assert result['value'] is None and result['status'] == 'UNDETERMINED'


@pytest.mark.parametrize('output', [None, {'status': 'NO_EVIDENCE', 'answer': '', 'claims': []},
    {'status': 'CONFLICT', 'answer': 'synthetic', 'claims': [], 'evidence': [
        {'sourceType': 'TOOL', 'mock': True}, {'sourceType': 'DOCUMENT'}]}])
def test_sample_preserves_missing_empty_conflict_and_mock(output):
    sample = build_sample({'question': {'text': 'final'}, 'expected': {'status': 'ABSTAIN'}},
        {'mode': 'AGENT'}, 'sample', 1, output)
    assert sample['actualContext']['status'] == 'UNKNOWN'
    assert sample['history']['messages'] is None and sample['history']['status'] == 'UNKNOWN'
    assert sample['answer']['present'] == bool(output is not None and 'answer' in output)
    if output and output['status'] == 'CONFLICT':
        assert sample['sourceKinds'] == {'documents': 1, 'tools': 1, 'mockTools': 1}


def test_failed_and_old_metrics_remain_unknown_without_changing_legacy_values():
    case = {'expected': {'evidenceGroups': [{'alternatives': [{'quote': 'gold'}]}]}}
    old = {'recall': 0.0}
    failed = describe_metrics(case, 'FAILED', old, {})
    assert old == {'recall': 0.0} and all(m['value'] is None and m['status'] == 'ERROR' for m in failed)
    known = describe_metrics(case, 'COMPLETED', {'recall': .5}, {})
    coverage = metric_coverage([{'metricDiagnostics': known}, {'metricDiagnostics': failed}, {}])
    recall = next(m for m in coverage if m['name'] == 'recall')
    assert recall['computed'] == recall['errors'] == recall['unknown'] == 1
    assert recall['mean'] == .5 and recall['plannedCases'] == 3
