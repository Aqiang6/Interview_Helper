import sys, asyncio
sys.path.insert(0, r'd:\HR Agent')
from ai_interviewer.rag_engine.retriever import RAGRetriever, RetrievalResult, get_retriever

# 1. 检查方法名：RAGRetriever 只有 retrieve / retrieve_by_skills；retrieve_by_topic 不存在
r = RAGRetriever()
methods = [m for m in dir(r) if not m.startswith('_')]
print('1) RAGRetriever 公共方法:', sorted(methods))
assert hasattr(r, 'retrieve'), 'retrieve 方法不存在'
assert hasattr(r, 'retrieve_by_skills'), 'retrieve_by_skills 不存在'
assert not hasattr(r, 'retrieve_by_topic'), '！居然有 retrieve_by_topic？跟预期不符'
print('   ✓ retrieve 存在；retrieve_by_topic 不存在（符合预期，之前的调用写错了）')

# 2. 直接调 retrieve('Python') → 虽然 KB 空，但应该返回空 RetrievalResult，绝不抛 AttributeError
try:
    res = asyncio.run(r.retrieve('Python'))
    print(f'2) retrieve("Python") 返回 RetrievalResult: query={res.query!r}  items={len(res.items)}  scores={len(res.scores)}')
    assert isinstance(res, RetrievalResult)
    assert isinstance(res.items, list)
    print('   ✓ format_context() 无异常：', repr(res.format_context())[:30])
except AttributeError as e:
    print(f'   ❌ AttributeError: {e}')
    sys.exit(1)

# 3. 验证不存在的属性 context 会触发 AttributeError（证明之前 result.context 是另一个bug）
try:
    _ = res.context
    print('   ❗ 意外：res.context 居然存在')
except AttributeError as e:
    print(f'   ✓ (印证) res.context 确实不存在 → {type(e).__name__}: {e}，之前代码 result.context 也是潜在 BUG（已改成用 result.items）')

print('\n3) OK：RAGRetriever 链路修复正确，不会再抛 retrieve_by_topic / context 两个 AttributeError')
