"""
Qwen3.5-4B 教师模型 vs 学生模型 100题回答一致性评测脚本
======================================================
1. 加载 Full Attention 教师模型 (cuda:3) 和 RTPurbo 稀疏学生模型 (cuda:2)。
2. 对 7 个大类、共 100 道题目进行自回归生成对比。
3. 使用无外部包依赖的算法计算字符级 ROUGE-L 与 BLEU-1 重合度指标。
4. 将评测详情和最终的分类汇总报告导出。
"""

import os
import sys
import json
import math
import argparse
import time
import torch
import torch.nn.functional as F

if not hasattr(torch, "float8_e8m0fnu"):
    torch.float8_e8m0fnu = getattr(torch, "float8_e4m3fn", torch.float32)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, AutoModelForImageTextToText
from model.model_qwen_rtpurbo import convert_qwen_to_rtpurbo


# LCS 算法，计算最长公共子序列长度用于 ROUGE-L 计算
def get_lcs(x, y):
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i-1] == y[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def calculate_rouge_l(ref, hyp):
    """计算 ROUGE-L (LCS F1-Score)"""
    if not ref or not hyp:
        return 0.0
    ref_tokens = list(ref)
    hyp_tokens = list(hyp)
    lcs_len = get_lcs(ref_tokens, hyp_tokens)
    prec = lcs_len / len(hyp_tokens)
    rec = lcs_len / len(ref_tokens)
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def calculate_bleu_1(ref, hyp):
    """计算简易 BLEU-1 (基于 1-gram 词重合度)"""
    if not ref or not hyp:
        return 0.0
    ref_tokens = list(ref)
    hyp_tokens = list(hyp)
    
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
        
    hyp_counts = {}
    for t in hyp_tokens:
        hyp_counts[t] = hyp_counts.get(t, 0) + 1
        
    overlap = 0
    for t, count in hyp_counts.items():
        if t in ref_counts:
            overlap += min(count, ref_counts[t])
            
    return overlap / len(hyp_tokens)


def load_teacher(args, device):
    """加载原始全精度 Qwen3.5-4B 模型"""
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    return model.to(device).eval()


def load_student(args, device):
    """加载动态转换为 RTPurbo 后的学生模型并读入微调权重"""
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )
    with open(args.head_config, 'r', encoding='utf-8') as f:
        head_config = json.load(f)

    # 替换为 RTPurbo 稀疏结构
    model = convert_qwen_to_rtpurbo(
        model,
        head_config,
        index_dim=args.index_dim,
        local_window_size=args.local_window_size,
        retrieval_top_p=args.retrieval_top_p,
        sparse_attn=True
    )

    # 加载已微调完毕的权重
    if args.weight_path and os.path.exists(args.weight_path):
        w = torch.load(args.weight_path, map_location=device)
        missing, unexpected = model.load_state_dict(w, strict=False)
        print(f"✅ 加载 RTPurbo 学生端权重: {args.weight_path}")
        print(f"  缺失键数: {len(missing)}, 多余键数: {len(unexpected)}")
    else:
        print("⚠️ 未检测到已训练权重，使用基线权重直接推理！")

    return model.to(device).eval()


@torch.no_grad()
def greedy_generate(model, tokenizer, prompt, max_new_tokens=2048):
    """使用 KV Cache 的高效自回归文本生成 (贪婪解码)"""
    system_prompt = "请直接、简短地回答用户的问题。严禁输出任何思考过程（思考链、Thinking Process 等），只输出最终答案。"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    template = template + "直接回答："
    input_ids = tokenizer(template, return_tensors="pt").input_ids.to(model.device)

    # 1. Prefill 阶段
    with torch.amp.autocast('cuda', dtype=torch.float16):
        outputs = model(input_ids, use_cache=True)
    
    logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    past_key_values = outputs.past_key_values
    
    generated_tokens = [next_token.item()]
    
    if next_token.item() == tokenizer.eos_token_id:
        return tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # 2. Decode 阶段
    for _ in range(max_new_tokens - 1):
        with torch.amp.autocast('cuda', dtype=torch.float16):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        
        token_id = next_token.item()
        generated_tokens.append(token_id)
        if token_id == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# 100 道精心设计的测试题目
prompts_100 = [
    # 类别1: 常识百科 (General Knowledge) - 15题
    {"category": "常识百科", "prompt": "地球上海洋和陆地的比例是多少？哪个大洋的面积最大？"},
    {"category": "常识百科", "prompt": "请解释什么是光合作用，以及它对地球生态系统的重要性。"},
    {"category": "常识百科", "prompt": "简述中国四大发明是什么，并说明它们分别在什么历史时期被发明。"},
    {"category": "常识百科", "prompt": "世界上最高的山峰是什么？它位于哪两个国家的交界处？"},
    {"category": "常识百科", "prompt": "请介绍一下太阳系中八大行星的名称和基本排列顺序。"},
    {"category": "常识百科", "prompt": "什么是GDP？它是如何衡量一个国家的经济状况的？"},
    {"category": "常识百科", "prompt": "莎士比亚的四大悲剧分别是什么？请列出书名。"},
    {"category": "常识百科", "prompt": "为什么天空是蓝色的，而日落时天空会变成红色或橙色？"},
    {"category": "常识百科", "prompt": "第一届现代奥林匹克运动会是在哪一年、哪一个城市举办的？"},
    {"category": "常识百科", "prompt": "请简述牛顿三大运动定律的内容。"},
    {"category": "常识百科", "prompt": "水的三态变化是什么？它的临界点温度分别是多少？"},
    {"category": "常识百科", "prompt": "简述古埃及金字塔的主要用途以及最著名的金字塔名称。"},
    {"category": "常识百科", "prompt": "什么是相对论？它是谁在什么年代提出的？"},
    {"category": "常识百科", "prompt": "人体中最大的器官是什么？它有什么主要功能？"},
    {"category": "常识百科", "prompt": "请简述工业革命对人类社会产生的三个主要影响。"},

    # 类别2: 创意写作 (Creative Writing) - 15题
    {"category": "创意写作", "prompt": "写一首关于秋天落叶的四句现代小诗，要求意境优美。"},
    {"category": "创意写作", "prompt": "以“一次意外的旅行”为题，写一段150字左右的微型小说开头。"},
    {"category": "创意写作", "prompt": "请为一家新开张的绿色环保咖啡馆撰写一句吸引人的广告语。"},
    {"category": "创意写作", "prompt": "写一封给未来自己（十年后）的简短问候信，表达对未来的期许。"},
    {"category": "创意写作", "prompt": "虚构一段宇航员第一次踏上火星土地时的内心独白。"},
    {"category": "创意写作", "prompt": "请用生动的语言描写清晨森林里雾气消散的景象。"},
    {"category": "创意写作", "prompt": "假设你是一只流浪猫，请用第一人称写一段你眼中的城市夜晚。"},
    {"category": "创意写作", "prompt": "为一本关于时间旅行的科幻小说写一个引人入胜的简介（100字左右）。"},
    {"category": "创意写作", "prompt": "写一份邀请朋友参加你周末生日派对的幽默短信。"},
    {"category": "创意写作", "prompt": "以“风的声音”为主题，写一段富有哲理的散文段落。"},
    {"category": "创意写作", "prompt": "设计一个关于“智能机器人学会流泪”的微型科幻故事大纲。"},
    {"category": "创意写作", "prompt": "请为一首轻音乐写一段意境描述词，让人闭上眼就能感受到画面。"},
    {"category": "创意写作", "prompt": "写一封向老板申请调岗的正式商务邮件草稿，语气需委婉且专业。"},
    {"category": "创意写作", "prompt": "描写一个古老图书馆角落里的陈设，突出它的历史感和神秘感。"},
    {"category": "创意写作", "prompt": "为一款智能手表撰写一段简短的产品宣传文案，突出健康监测功能。"},

    # 类别3: 代码开发 (Code & Coding) - 15题
    {"category": "代码开发", "prompt": "用Python写一个快速排序算法函数。"},
    {"category": "代码开发", "prompt": "用JavaScript编写一个判断输入字符串是否是回文的函数。"},
    {"category": "代码开发", "prompt": "用SQL写一个查询语句，找出 employees 表中工资最高的三个员工的姓名和工资。"},
    {"category": "代码开发", "prompt": "解释什么是 RESTful API，并列举常用的四种 HTTP 方法及其含义。"},
    {"category": "代码开发", "prompt": "用Python写一个读取 JSON 文件并打印所有键值对的示例代码。"},
    {"category": "代码开发", "prompt": "在 HTML/CSS 中，如何实现一个子元素在父容器中水平和垂直居中？写出CSS代码。"},
    {"category": "代码开发", "prompt": "请用Python实现一个斐波那契数列的前N项生成生成器。"},
    {"category": "代码开发", "prompt": "解释面向对象编程（OOP）中的“封装”、“继承”和“多态”三大特征。"},
    {"category": "代码开发", "prompt": "在Python中，`__init__` 和 `__new__` 的区别是什么？"},
    {"category": "代码开发", "prompt": "用Python写一个正则表达式，匹配并提取文本中的所有电子邮件地址。"},
    {"category": "代码开发", "prompt": "用Go语言写一个简单的并发打印“Hello World”的Goroutine例子。"},
    {"category": "代码开发", "prompt": "解释什么是数据库的“事务”，以及 ACID 特性分别代表什么。"},
    {"category": "代码开发", "prompt": "用Python写一个单例模式（Singleton Pattern）的经典实现。"},
    {"category": "代码开发", "prompt": "如何用Git撤销上一次已经 commit 但还没有 push 的提交？请给出命令。"},
    {"category": "代码开发", "prompt": "用Python写一个函数，计算两个给定列表的交集。"},

    # 类别4: 数理逻辑 (Math & Logic) - 15题
    {"category": "数理逻辑", "prompt": "一个笼子里有鸡和兔共35只，脚共有94只，问鸡和兔各有多少只？请给出详细解题步骤。"},
    {"category": "数理逻辑", "prompt": "如果5个人在5天内能盖5间房子，那么100个人在100天内能盖多少间房子？"},
    {"category": "数理逻辑", "prompt": "请问 2的10次方 的值是多少？"},
    {"category": "数理逻辑", "prompt": "一个三角形的三边长分别为3、4、5，求这个三角形的面积。"},
    {"category": "数理逻辑", "prompt": "逻辑推理：甲说乙在撒谎，乙说丙在撒谎，丙说甲和乙都在撒谎。请问谁说的是真话？"},
    {"category": "数理逻辑", "prompt": "求方程 x^2 - 5x + 6 = 0 的根。"},
    {"category": "数理逻辑", "prompt": "有三个抽屉，一个放着2个红球，一个放着2个白球，一个放着1红1白。抽屉标签全贴错了。你只能从一个抽屉摸一个球，如何确定三个抽屉的正确球类？"},
    {"category": "数理逻辑", "prompt": "计算概率：掷一枚均匀的骰子两次，两次点数之和为7的概率是多少？"},
    {"category": "数理逻辑", "prompt": "证明或解释：为什么任何数的0次方（除0外）都等于1？"},
    {"category": "数理逻辑", "prompt": "如果 3x + 5 = 20，那么 x 的值是多少？"},
    {"category": "数理逻辑", "prompt": "逻辑题：有四张卡片，每张卡片正面是数字，反面是字母。现在桌上放着四张卡片，分别显示：'3', '8', 'A', 'B'。若要验证规则“如果卡片正面是偶数，反面必须是元音字母”，你需要翻看哪几张卡片？"},
    {"category": "数理逻辑", "prompt": "求 1 到 100 所有整数的和是多少？"},
    {"category": "数理逻辑", "prompt": "小明跑步的速度是每秒5米，小华骑车的速度是每小时18公里，请问谁的速度更快？"},
    {"category": "数理逻辑", "prompt": "已知 a + b = 10，ab = 21，求 a^2 + b^2 的值。"},
    {"category": "数理逻辑", "prompt": "若一个正方形的对角线长度是 10 厘米，求它的面积。"},

    # 类别5: 文本抽取与摘要 (Summary & Extraction) - 15题
    {"category": "文本抽取与摘要", "prompt": "阅读段落：“2026年6月，人工智能领域的初创公司AlphaTech宣布完成由红杉领投的5000万美元B轮融资。该公司计划将这笔资金用于下一代通用机器人大脑的研发和人才引进。”\n请提取：1. 公司名称 2. 融资轮次 3. 融资金额 4. 领投机构。"},
    {"category": "文本抽取与摘要", "prompt": "将以下段落总结为一句话：“气候变化正以前所未有的速度影响全球生态。两极冰川融化导致海平面上升，极端天气频发威胁农业生产，许多物种面临灭绝危机。科学家呼吁各国必须立即采取减排行动。”"},
    {"category": "文本抽取与摘要", "prompt": "阅读段落：“张伟是明华中学的学生，他热爱足球，周末经常和同学李强一起踢球。他的姐姐张敏在清华大学读研究生，专攻生物化学。”\n请列出张伟和李强、张敏之间的关系。"},
    {"category": "文本抽取与摘要", "prompt": "从以下段落中提取出所有提到的具体年份：“苹果公司成立于1976年。1984年推出了首款麦金塔电脑。在1997年史蒂夫·乔布斯重返苹果后，公司重回巅峰，并在2007年发布了第一代iPhone，改变了智能手机行业。”"},
    {"category": "文本抽取与摘要", "prompt": "阅读并概括下面这段产品说明书的核心功能：“本款空气净化器采用三层HEPA滤网，能有效过滤99.97%的微尘与PM2.5。同时内置活性炭层，吸附甲醛和异味。支持手机APP远程操控，实时监测室内空气指数并自动调节风速。”"},
    {"category": "文本抽取与摘要", "prompt": "提取以下会议通知中的时间、地点和主题：“为了讨论第二季度项目研发进度，定于下周三（6月10日）上午10点在公司三楼多功能会议室召开全体技术人员会议，请务必准时参加。”"},
    {"category": "文本抽取与摘要", "prompt": "阅读段落：“红茶属于全发酵茶，茶性温和；绿茶是未发酵茶，保留了较多鲜叶的天然物质，茶性偏凉；乌龙茶则是半发酵茶，介于两者之间，香气浓郁。”\n请用简短列表形式整理红茶、绿茶、乌龙茶的发酵程度和茶性。"},
    {"category": "文本抽取与摘要", "prompt": "将以下短文压缩到50字以内：“区块链是一种去中心化的分布式账本技术。它通过密码学算法保证数据的不可篡改和可追溯性，广泛应用于数字货币、供应链金融、智能合约等多个领域，是数字经济的重要基础设施。”"},
    {"category": "文本抽取与摘要", "prompt": "阅读文本：“小明购买了一张6月12日北京到上海的高铁票，车次为G21，发车时间为早上8点，票价553元。”\n请提取小明的出发地、目的地、发车时间和票价。"},
    {"category": "文本抽取与摘要", "prompt": "请提取段落中提到的两个核心科学结论：“最新研究表明，充足的睡眠不仅有助于大脑清除代谢废物，防止阿尔茨海默症的发生，而且能显著增强机体免疫细胞的活性，降低病毒感染风险。”"},
    {"category": "文本抽取与摘要", "prompt": "从以下招聘简章中提取任职要求：“本岗位招聘前端工程师一名。要求计算机相关专业本科以上学历，熟练掌握React/Vue，具有3年以上实际开发经验。有大厂工作背景者优先。”"},
    {"category": "文本抽取与摘要", "prompt": "总结以下段落的核心观点：“许多人认为独自旅行很孤单，但实际上，独自旅行迫使我们走出舒适圈，去主动结交新朋友、面对各种突发状况。在这个过程中，人们不仅能领略异域风景，更能深刻认识自己，获得内心的成长。”"},
    {"category": "文本抽取与摘要", "prompt": "阅读段落并提取出公司的客服电话及工作时间：“如有任何疑问，请拨打我们的官方客服热线 400-123-4567。我们的客服工作时间为周一至周五的上午9:00至下午18:00，法定节假日除外。”"},
    {"category": "文本抽取与摘要", "prompt": "从以下段落中提取三种提到的清洁能源名称：“随着环保意识的增强，各国纷纷布局绿色电力。除了传统的风力发电和太阳能光伏发电外，生物质能、潮汐能及地热能的开发利用也取得了突破性进展。”"},
    {"category": "文本抽取与摘要", "prompt": "将以下工作流程概括为三个步骤：“首先，申请人需要登录系统填写个人基本信息并提交相关证明文件。其次，部门主管会对提交的材料进行真实性审核并签署意见。最后，HR部门核对无误后会发放录用通知书并办理入职手续。”"},

    # 类别6: 翻译与语言 (Translation & Linguistics) - 15题
    {"category": "翻译与语言", "prompt": "将英语句子翻译为中文：'Practice makes perfect.'"},
    {"category": "翻译与语言", "prompt": "将中文句子翻译为英文：“不积跬步，无以至千里。”"},
    {"category": "翻译与语言", "prompt": "请解释英语习语 'kick the bucket' 的字面意思和实际含义。"},
    {"category": "翻译与语言", "prompt": "将英语商务短语 'win-win situation' 翻译为贴切的中文。"},
    {"category": "翻译与语言", "prompt": "把这句话翻译成英文：“请代我向你的父母问好。”"},
    {"category": "翻译与语言", "prompt": "请解释汉语成语“画蛇添足”的典故和比喻含义。"},
    {"category": "翻译与语言", "prompt": "将以下句子翻译为现代汉语：“学而时习之，不亦说乎？”"},
    {"category": "翻译与语言", "prompt": "将英语科技段落翻译为中文：'Artificial Intelligence is rapidly advancing and transforming industries by automating processes and providing deep data insights.'"},
    {"category": "翻译与语言", "prompt": "英语词汇辨析：请说明 'affect' 和 'effect' 在用法上的主要区别。"},
    {"category": "翻译与语言", "prompt": "将中文诗句翻译为英文：“床前明月光，疑是地上霜。”"},
    {"category": "翻译与语言", "prompt": "请将短语 'out of the blue' 翻译为中文，并用它造一个英文句子。"},
    {"category": "翻译与语言", "prompt": "将以下地道的中文口语翻译为英文：“这简直是小菜一碟！”"},
    {"category": "翻译与语言", "prompt": "把英语句子翻译为中文：'It is never too late to learn.'"},
    {"category": "翻译与语言", "prompt": "请解释汉语“歇后语”的概念，并给出一个经典的例子。"},
    {"category": "翻译与语言", "prompt": "将英语商务邮件常用语 'I look forward to hearing from you.' 翻译为得体的中文。"},

    # 类别7: 角色扮演与日常 (Roleplay & Conversation) - 10题
    {"category": "角色扮演与日常", "prompt": "扮演一位专业的营养师，给一位经常熬夜的白领提供三条简短的饮食建议。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位幽默的导游，向游客介绍著名的法国埃菲尔铁塔。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位健身教练，鼓励一位想放弃锻炼的学员坚持下去。"},
    {"category": "角色扮演与日常", "prompt": "模拟一次客服对话：用户购买的商品快递破损，请作为客服委婉地道歉并提供解决方案。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位智慧的老者，回答年轻人关于“如何面对生活中的焦虑”的提问。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位挑剔的食客，给一家服务态度极差的餐厅写一段差评。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位小学班主任，写一段话在家长会上表扬孩子们这学期的进步。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位心理咨询师，安慰一位因为考试失利而极度沮丧的学生。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位科幻电影里的AI管家，用温和的语气迎接疲惫回家的主人。"},
    {"category": "角色扮演与日常", "prompt": "扮演一位富有激情的创业导师，向年轻的大学生做一分钟的创业鼓动演说。"}
]


def main():
    parser = argparse.ArgumentParser(description="Qwen3.5 RTPurbo 100 Prompts Consistency Evaluation")
    parser.add_argument("--model_path", type=str, default="../model/Qwen3.5-4B")
    parser.add_argument("--weight_path", type=str, default="../out/rtpurbo_stage2_qwen_2048.pth")
    parser.add_argument("--head_config", type=str, default="../qwen_head_config_2048.json")
    parser.add_argument("--index_dim", type=int, default=16)
    parser.add_argument("--local_window_size", type=int, default=128)
    parser.add_argument("--retrieval_top_p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--teacher_device", type=str, default="cuda:3")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--output", type=str, default="../out/qwen_consistency_results.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print("=" * 60)
    print("Qwen3.5-4B 教师模型 vs 学生模型 100题一致性评测")
    print("=" * 60)
    print(f"  Student 设备: {args.device} | Teacher 设备: {args.teacher_device}")
    print(f"  测试题数: {len(prompts_100)} 题")

    print("\n[1/3] 加载双侧模型权重中...")
    teacher = load_teacher(args, args.teacher_device)
    student = load_student(args, args.device)
    print("  模型加载并结构转换成功！")

    print("\n[2/3] 开始生成回答并对比计算相似度指标...")
    eval_details = []
    
    # 按照类别统计的累加器
    category_stats = {}

    for i, item in enumerate(prompts_100, start=1):
        category = item["category"]
        prompt = item["prompt"]
        
        # 初始化分类统计
        if category not in category_stats:
            category_stats[category] = {"rouge_l_sum": 0.0, "bleu_1_sum": 0.0, "count": 0}
            
        print(f"  [{i}/100] 分类: {category} | 题面: {prompt[:30]}...")
        sys.stdout.flush()
        
        # 教师端和学生端独立推理 (附带时间打点)
        t_start = time.time()
        t_ans = greedy_generate(teacher, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        t_mid = time.time()
        s_ans = greedy_generate(student, tokenizer, prompt, max_new_tokens=args.max_new_tokens)
        t_end = time.time()
        
        t_duration = t_mid - t_start
        s_duration = t_end - t_mid
        
        print(f"     Teacher: {len(t_ans)} chars ({t_duration:.2f}s) | Student: {len(s_ans)} chars ({s_duration:.2f}s)")
        sys.stdout.flush()
        
        # 计算相似性分数
        rouge_l = calculate_rouge_l(t_ans, s_ans)
        bleu_1 = calculate_bleu_1(t_ans, s_ans)
        
        category_stats[category]["rouge_l_sum"] += rouge_l
        category_stats[category]["bleu_1_sum"] += bleu_1
        category_stats[category]["count"] += 1
        
        eval_details.append({
            "id": i,
            "category": category,
            "prompt": prompt,
            "teacher_response": t_ans.strip(),
            "student_response": s_ans.strip(),
            "rouge_l": rouge_l,
            "bleu_1": bleu_1
        })

    # 3. 汇总报告并保存
    print("\n[3/3] 评测结束，正在汇总分类数据...")
    
    report = {
        "overall": {
            "avg_rouge_l": 0.0,
            "avg_bleu_1": 0.0,
            "total_count": len(prompts_100)
        },
        "by_category": {}
    }
    
    overall_rouge_l = 0.0
    overall_bleu_1 = 0.0
    
    print("-" * 75)
    print(f"{'评测分类':<15}{'题量':<8}{'平均 ROUGE-L':<15}{'平均 BLEU-1':<15}")
    print("-" * 75)
    
    for cat, stat in category_stats.items():
        count = stat["count"]
        avg_rouge = stat["rouge_l_sum"] / count
        avg_bleu = stat["bleu_1_sum"] / count
        
        overall_rouge_l += stat["rouge_l_sum"]
        overall_bleu_1 += stat["bleu_1_sum"]
        
        print(f"{cat:<15}{count:<8}{avg_rouge:<15.4f}{avg_bleu:<15.4f}")
        
        report["by_category"][cat] = {
            "avg_rouge_l": avg_rouge,
            "avg_bleu_1": avg_bleu,
            "count": count
        }
        
    avg_overall_rouge = overall_rouge_l / len(prompts_100)
    avg_overall_bleu = overall_bleu_1 / len(prompts_100)
    
    report["overall"]["avg_rouge_l"] = avg_overall_rouge
    report["overall"]["avg_bleu_1"] = avg_overall_bleu
    
    print("-" * 75)
    print(f"{'总计平均':<15}{len(prompts_100):<8}{avg_overall_rouge:<15.4f}{avg_overall_bleu:<15.4f}")
    print("-" * 75)
    
    final_output = {
        "report": report,
        "details": eval_details
    }
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 一致性评测详情与最终报告成功导出至: {args.output}")


if __name__ == "__main__":
    main()
