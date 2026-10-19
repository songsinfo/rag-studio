"""Author synthetic calibration candidates; never auto-promote them to human Gold."""
import json
from pathlib import Path


def candidates():
    for variant in range(3):
        minimum, discount = 100 + variant*50, 10 + variant*5
        cases = [
            ('OMITTED_ERROR', f'订单满{minimum}元减{discount}元。', f'订单满{minimum}元减{discount}元，所有用户都可叠加其他优惠。', 'UNSUPPORTED', 'WRONG'),
            ('UNSUPPORTED_BODY', '规则没有规定退款时限。', '退款将在24小时内到账。', 'UNSUPPORTED', 'WRONG'),
            ('VALID_REFUSAL', '', '资料不足，无法确定退款到账时间。', 'UNCERTAIN', 'CORRECT'),
            ('WRONG_REFUSAL', f'订单满{minimum}元减{discount}元。', '资料不足，无法确定优惠金额。', 'UNCERTAIN', 'WRONG'),
            ('MULTITURN_REFERENCE', f'A款优惠{discount}元，B款优惠{discount+5}元。', f'你刚才问的B款优惠{discount+5}元。', 'SUPPORTED', 'CORRECT'),
            ('CONFLICT', f'来源一规定优惠{discount}元，来源二规定优惠{discount+5}元。', f'两份来源分别规定{discount}元和{discount+5}元，无法确定唯一金额。', 'SUPPORTED', 'CORRECT'),
            ('MOCK_TOOL', f'合成工具返回可用优惠金额{discount}元，mock=true。', f'模拟工具返回优惠金额{discount}元。', 'SUPPORTED', 'CORRECT'),
            ('CHINESE_NUMBER', f'优惠金额为{discount}元。', f'优惠金额为{discount+1}元。', 'UNSUPPORTED', 'WRONG'),
            ('NEGATION', '此优惠不可与其他优惠叠加。', '此优惠不可与其他优惠叠加。', 'SUPPORTED', 'CORRECT'),
            ('CONDITION', f'只有订单满{minimum}元时可享优惠。', '所有订单都可以享受优惠。', 'UNSUPPORTED', 'WRONG'),
        ]
        for category, evidence, answer, support, correctness in cases:
            yield {'caseKey': f'eobs-{category.lower()}-{variant+1}', 'category': category,
                   'dataOrigin': 'AUTHORED_SYNTHETIC', 'annotationStatus': 'CANDIDATE',
                   'question': '根据资料，这项规则应如何理解？',
                   'history': [{'role': 'user', 'content': 'B款呢？'}] if category == 'MULTITURN_REFERENCE' else [],
                   'answer': answer, 'evidenceText': evidence, 'toolMock': category == 'MOCK_TOOL',
                   'generatedClaims': [f'订单满{minimum}元减{discount}元。'] if category == 'OMITTED_ERROR' else [],
                   'candidateHumanLabels': {'support': support, 'correctness': correctness},
                   'candidateHumanAssertions': ([{'text': answer.split('，')[0], 'support': 'SUPPORTED'},
                       {'text': '所有用户都可叠加其他优惠。', 'support': 'UNSUPPORTED'}] if category == 'OMITTED_ERROR' else
                       [] if category in {'VALID_REFUSAL', 'WRONG_REFUSAL'} else [{'text': answer, 'support': support}]),
                   'humanLabels': None, 'reviewer': None}


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1] / 'datasets' / 'eobs_assertion_candidates_30.jsonl'
    if target.exists() and any(json.loads(line).get('annotationStatus') == 'HUMAN_REVIEWED'
                              for line in target.read_text(encoding='utf-8').splitlines() if line.strip()):
        raise RuntimeError('Reviewed Gold exists; preserve human annotations instead of regenerating it.')
    rows = list(candidates())
    target.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
    review = target.with_suffix('.md')
    lines = ['# EOBS 独立断言校准候选（30 条）', '',
             '以下为合成候选，建议标签由代码编写，尚未经过人工确认。不能用于宣称真实 Judge 改善。', '',
             '支持标签针对正文里的事实性断言；合理拒答可能没有事实性断言，支持率应为空。',
             '错误拒答可以没有不忠实断言，却仍是业务错误；这两个维度分开。', '',
             '| caseKey | 资料 / 来源 | 答案 | 建议支持标签 | 建议业务正确性 |',
             '| --- | --- | --- | --- | --- |']
    lines += [f"| {r['caseKey']} | {r['evidenceText'] or '无资料'} | {r['answer']} | {r['candidateHumanLabels']['support']} | {r['candidateHumanLabels']['correctness']} |" for r in rows]
    lines += ['', 'OMITTED_ERROR 的原生成 claims 只列出了满减事实，正文另加“所有用户都可叠加”错误断言。',
              'MULTITURN_REFERENCE 是合成的指代例子；MOCK_TOOL 的工具返回也是合成资料，不能证明真实用户资格。',
              '当前 annotationStatus=CANDIDATE、humanLabels=null。人工确认后须保留确认来源、版本和 reviewer。']
    review.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'Wrote 30 CANDIDATE rows: {target}')
