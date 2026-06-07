import json

# 检查 sft_t2t_mini.jsonl 中不同记录的字段差异
fields_counter = {}
total_records = 0
with open('dataset/sft_t2t_mini.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 2000:
            break
        total_records += 1
        data = json.loads(line)
        for msg in data.get('conversations', []):
            keys = tuple(sorted(msg.keys()))
            fields_counter[keys] = fields_counter.get(keys, 0) + 1

print(f'=== sft_t2t_mini.jsonl 消息字段统计 (前{total_records}条记录) ===')
for keys, count in sorted(fields_counter.items(), key=lambda x: -x[1]):
    print(f'  {count:5d} 条消息: {list(keys)}')

# 检查 SFTDataset 期望的 features
print('\n=== SFTDataset 期望的 features ===')
print("  每条消息必须有: role, content, reasoning_content, tools, tool_calls")

# 看看全量 sft 数据
import os
print('\n=== 可用的 SFT 数据文件 ===')
for f in sorted(os.listdir('dataset')):
    if 'sft' in f.lower():
        size = os.path.getsize(f'dataset/{f}') / 1024 / 1024
        print(f'  {f}: {size:.1f} MB')

# 查看第一条完整数据
print('\n=== 第一条数据示例 ===')
with open('dataset/sft_t2t_mini.jsonl') as f:
    data = json.loads(f.readline())
    for i, msg in enumerate(data['conversations']):
        print(f'  消息{i}: keys={list(msg.keys())}')
        for k, v in msg.items():
            val = str(v)[:80] if v else 'null'
            print(f'    {k}: {val}')
