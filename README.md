
目前 published 的帖子列表是：
- 2026-08-28 博客
- 2026-08-27 财务报表分析
- 2026-08-27 TI-计算器
- 2026-08-27 经济学一般均衡模型
- 2026-08-27 经济学因果推断
- 2026-08-27 量化交易
- 2026-06-23 无名
- 2026-05-04 什么是分析
- 2026-05-03 写一点小的演示
- 2026-04-23 IELTS Speaking
- 2026-04-18 学术写作2
- 2026-04-18 看老友记学英语
- 2026-04-18 词汇笔记1-词汇用法
- 2026-04-18 词汇笔记2-词汇表
- 2026-04-13 一些Ruby代码片段5-图论算法和排列组合
- 2026-03-28 经济学数据分析
- 2026-03-22 用 Python 进行程序分析
- 2026-03-22 用 Ruby 进行程序分析
- 2026-03-19 文档的表示
- 2026-03-19 组合优化问题
- 2026-03-16 大语言模型1-基本用法
- 2026-03-15 如何应对目标
- 2026-02-23 Ruby代码片段4-搜索与规划
- 2026-02-23 心情日记
- 2026-02-18 英语常见词缀词根
- 2026-02-18 英语词根和单词
- 2026-02-18 英语笔记-读音
- 2026-02-17 如何阅读一本书
- 2026-02-13 数学笔记1
- 2026-01-18 Python 片段
- 2026-01-18 Ruby 代码片段3 常见算法
- 2026-01-17 LaTeX 的使用
- 2026-01-16 Java 网络应用开发
- 2025-12-07 学术写作1
- 2025-10-01 PyTorch 与深度学习
- 2025-08-16 转载《培根・新工具》
- 2025-08-12 用 Python 进行数据分析2 统计学的方法
- 2025-08-04 数学是发明的还是发现的？
- 2025-05-11 如何记笔记
- 2025-05-08 如何用 sklearn 进行文本分析
- 2025-05-08 如何用 NumPy 实现人工神经网络
- 2025-01-10 [转载]英语短语动词
- 2025-01-09 辅助定理证明
- 2025-01-07 想要写一个故事
- 2025-01-06 读句子背单词
- 2025-01-05 The Writing Style
- 2025-01-02 读句子背单词3 老友记1背单词
- 2025-01-02 读句子背单词2 走遍美国背单词
- 2024-12-27 读句子背单词1 新概念英语背单词
- 2024-12-24 一些 Ruby 代码片段2 趣味代码
- 2024-12-23 Ruby 代码片段
- 2024-12-23 用 Python 进行数据分析1 数据的处理
- 2024-12-23 第一篇帖子，知识的由来和去向


由新到旧，根据 `_posts` 中 md 文件的 yaml 头

```python
# 生成上面的列表：读取 _posts 下顶层 .md 文件，按 yaml 头排序输出
import re
from pathlib import Path

posts_dir = Path(__file__).parent / "_posts"
results = []

# 把 date 字段标准化为 YYYY-MM-DD（去掉时间部分、补齐个位月份/日），仅用于显示
def norm_date(d):
    d = d.split(" ")[0].split("T")[0]
    return "-".join(p.zfill(2) for p in d.split("-"))

# 排序用的键：优先取 yaml 的 date，保留时间，保证按时间先后排序（"2025-5-8 16:55:00" 也能正确排序）
def date_key(d):
    head = d.split(" ")[0].split("T")[0]
    ymd = "-".join(p.zfill(2) for p in head.split("-"))
    t = ""
    for tok in d[len(head):].split():
        if re.fullmatch(r"\d{1,2}:\d{2}(:\d{2})?", tok):
            t = tok
            break
    return (ymd, t)

for f in sorted(posts_dir.glob("*.md")):
    text = f.read_text(encoding="utf-8")
    m = re.match(r"^---\r?\n(.*?)\r?\n---", text, re.DOTALL)
    if not m:
        continue
    yaml = m.group(1)

    published = "true"
    date = ""
    title = ""

    if pm := re.search(r"^published:\s*(.+)$", yaml, re.MULTILINE):
        published = pm.group(1).strip().lower()
    if dm := re.search(r"^date:\s*(.+)$", yaml, re.MULTILINE):
        date = dm.group(1).strip()
    if tm := re.search(r"^title:\s*(.+)$", yaml, re.MULTILINE):
        title = tm.group(1).strip().strip('"').strip("'")

    # 无 date 字段时用文件名日期，无 title 时用文件名（去掉日期前缀）
    if not date and (fm := re.match(r"(\d{4}-\d{2}-\d{2})", f.stem)):
        date = fm.group(1)
    if not title:
        title = re.sub(r"^\d{4}-\d{2}-\d{2}[- ]", "", f.stem)

    if published == "true":
        results.append((date, title))

for date, title in sorted(results, key=lambda x: date_key(x[0]), reverse=True):
    print(f"- {norm_date(date)} {title}")
```