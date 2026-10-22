# -*- coding: utf-8 -*-
"""
小说连载工坊 —— 本地离线大模型辅助短篇连载创作
纯 Python 标准库实现（tkinter + urllib），无第三方依赖，不会被 Smart App Control 拦截。

对接任意 OpenAI 兼容接口，例如：
  Ollama:    http://127.0.0.1:11434/v1   模型名如 qwen2.5:14b
  LM Studio: http://127.0.0.1:1234/v1

工作流程（与标签页一一对应）：
  第1步 设置项目与 API
  第2步 范文 -> 提炼文风特征
  第3步 整理人物档案
  第4步 整理人物关系与剧情大纲
  第5步 AI 生成章节摘要 -> 人工逐章检查修订
  第6步 AI 按摘要逐章生成正文 -> 人工修订 -> 导出全文
"""

import json
import os
import re
import shutil
import threading
import queue
import urllib.request
import urllib.error
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(APP_DIR, "projects")

DEFAULT_CFG = {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "",
    "model": "qwen2.5:14b",
    "temperature": 0.8,
    "max_tokens": 8192,
}

CHAR_FIELDS = ["姓名", "性别", "年龄", "职务/身份", "身高体型", "样貌",
               "穿着特点", "性格特点", "语言特点/口头禅", "典型台词",
               "动机与目标", "秘密", "成长弧线", "其他备注"]

REL_FIELDS = ["甲方", "乙方", "关系类型", "关系性质", "甲方对乙方称谓", "乙方对甲方称谓", "备注"]
OUTLINE_FIELDS = ["章节", "标题", "时间", "人物", "地点", "事件", "冲突/看点", "结尾钩子", "备注"]


# ---------------------------------------------------------------- 工具函数

def read_text(path, default=""):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return default


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def clip(text, limit=9000):
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…（内容过长已截断）"


def call_llm(cfg, messages, stream=False, on_chunk=None, abort=None):
    """同步调用 OpenAI 兼容 /chat/completions 接口。

    stream=True 时按 SSE（data: {...} 行）逐块读取，每收到一个增量片段
    回调 on_chunk(delta)；abort 为 threading.Event，置位后立即停止读取，
    已收到的部分通过返回值保留。非流式行为与之前完全一致。
    """
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    body = {
        "model": cfg["model"],
        "messages": messages,
        "temperature": float(cfg["temperature"]),
        "max_tokens": int(cfg["max_tokens"]),
        "stream": bool(stream),
    }
    headers = {"Content-Type": "application/json"}
    if cfg.get("api_key"):
        headers["Authorization"] = "Bearer " + cfg["api_key"]
    req = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=3600) as resp:
        if not stream:
            obj = json.loads(resp.read().decode("utf-8"))
            return obj["choices"][0]["message"]["content"]
        # 兼容性回退：服务端若忽略 stream 参数仍返回普通 JSON，按非流式处理
        # （内容一次性回调，界面仍能显示，只是没有渐进效果）
        if "text/event-stream" not in (resp.headers.get("Content-Type") or ""):
            obj = json.loads(resp.read().decode("utf-8"))
            content = obj["choices"][0]["message"]["content"]
            if on_chunk is not None and content:
                on_chunk(content)
            return content
        parts = []
        for raw in resp:
            if abort is not None and abort.is_set():
                break
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
                delta = obj["choices"][0].get("delta", {}).get("content", "") or ""
            except Exception:
                continue
            if delta:
                parts.append(delta)
                if on_chunk is not None:
                    on_chunk(delta)
        return "".join(parts)


def parse_summaries(text):
    """解析『第N章摘要：……』格式的输出 -> {N: 摘要文本}"""
    result = {}
    for m in re.finditer(r"第\s*(\d+)\s*章摘要\s*[:：]\s*(.+?)(?=第\s*\d+\s*章摘要|\Z)",
                         text, re.S):
        result[int(m.group(1))] = m.group(2).strip()
    return result


def ch_name(n):
    return "ch%02d" % int(n)


def read_any(fp):
    """按 utf-8 / gbk 顺序尝试读取文本文件。"""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="gbk", errors="replace") as f:
            return f.read()


def split_chunks(text, size=6000, max_chunks=30):
    """按段落边界把长文本切成若干块（超长段落硬切），供 AI 分段通读。"""
    paras = text.replace("\r\n", "\n").split("\n")
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:            # 单段超长：硬切
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:size])
            p = p[size:]
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            cur = p
        else:
            cur = cur + "\n" + p if cur else p
    if cur:
        chunks.append(cur)
    if len(chunks) > max_chunks:
        chunks = chunks[:max_chunks]
    return chunks


CHAPTER_HEAD_RE = re.compile(
    r"^[ \t]*(?:第\s*[0-9零一二三四五六七八九十百千万两]+\s*[章回节卷]|序章|楔子|引子|终章|尾声|Chapter\s+\d+)[^\n]*$",
    re.M | re.I)


def split_chapters(text):
    """按章节标题把小说原文切成章节列表；检测不到标题则整篇作为一章。"""
    text = text.replace("\r\n", "\n").strip()
    heads = list(CHAPTER_HEAD_RE.finditer(text))
    if not heads:
        return [text] if text else []
    parts = []
    for i, h in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        parts.append(text[h.start():end].strip())
    pre = text[:heads[0].start()].strip()   # 首个标题前的前言并入第一章
    if pre and len(pre) > 200 and parts:
        parts[0] = pre + "\n\n" + parts[0]
    return parts


def extract_md_section(text, keyword):
    """提取 Markdown 中标题含 keyword 的『## 章节』正文。"""
    m = re.search(r"^##[^\n]*%s[^\n]*$\n(.*?)(?=^##\s|\Z)" % keyword, text, re.M | re.S)
    return m.group(1).strip() if m else ""


def parse_outline_line(line):
    """把大纲行解析为结构化记录；兼容旧版 6 段（无标题）与新版 7 段（含标题）格式。"""
    rec = {k: "" for k in OUTLINE_FIELDS}
    parts = [p.strip() for p in str(line).split("｜")]
    head, rest = parts[0], parts[1:]
    m = re.search(r"\d+", head)
    if m and ("章" in head or head.isdigit()):
        rec["章节"] = int(m.group(0))
    else:
        rest = parts
    keys = OUTLINE_FIELDS[2:-1] if len(rest) == 6 else OUTLINE_FIELDS[1:-1]
    for k, v in zip(keys, rest):   # 旧:时间…钩子；新:标题…钩子
        rec[k] = v
    return rec


def parse_rel_string(text):
    """把自由文本关系描述按行解析为结构化记录。"""
    recs = []
    for line in str(text).splitlines():
        line = line.strip().lstrip("-*·").strip()
        if not line:
            continue
        rec = {k: "" for k in REL_FIELDS}
        m = re.match(r"(.+?)\s*(?:——|—|->|=>)\s*(.+?)\s*[：:]\s*(.+)", line)
        if m:
            rec["甲方"], rec["乙方"], rec["关系类型"] = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        else:
            rec["备注"] = line
        recs.append(rec)
    return recs


def to_rel_records(relationships):
    """统一把 AI 输出/旧文件内容转为关系记录列表。"""
    if isinstance(relationships, str):
        return parse_rel_string(relationships)
    recs = []
    for item in relationships or []:
        if isinstance(item, dict):
            recs.append({k: str(item.get(k, "") or "") for k in REL_FIELDS})
        else:
            recs.extend(parse_rel_string(str(item)))
    return [r for r in recs if any(v for k, v in r.items() if k != "备注") or r.get("备注")]


def to_outline_records(outline):
    """统一把 AI 输出/旧文件内容转为大纲记录列表，并按章节排序重编号。"""
    recs = []
    for i, item in enumerate(outline or [], 1):
        if isinstance(item, dict):
            rec = {k: str(item.get(k, "") or "") for k in OUTLINE_FIELDS}
            try:
                rec["章节"] = int(re.findall(r"\d+", rec["章节"] or str(i))[0])
            except Exception:
                rec["章节"] = i
        else:
            rec = parse_outline_line(item)
        recs.append(rec)
    recs.sort(key=lambda r: r.get("章节") or 0)
    for i, r in enumerate(recs, 1):
        r["章节"] = i
    return recs


def normalize_instance(text):
    """json 优先解析拆解实例；json 缺字段时从 Markdown 对应章节兜底提取。"""
    data = extract_instance_json(text) or parse_instance_md(text)
    if not str(data.get("relationships") or "").strip():
        data["relationships"] = extract_md_section(text, "人物关系")
    if not str(data.get("style") or "").strip():
        data["style"] = extract_md_section(text, "文风特征")
    if not str(data.get("theme") or "").strip():
        data["theme"] = extract_md_section(text, "基本盘") or extract_md_section(text, "题材")
    data["relationships"] = to_rel_records(data.get("relationships"))
    data["outline"] = to_outline_records(data.get("outline"))
    return data


def extract_instance_json(text):
    """从拆解实例中提取 ```json 结构化数据块。"""
    for block in reversed(re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S)):
        try:
            obj = json.loads(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def parse_instance_md(text):
    """JSON 块缺失时，从 Markdown 实例中兜底解析人物与大纲。"""
    data = {"style": "", "theme": "", "characters": [], "relationships": "", "outline": []}
    for m in re.finditer(r"^第\s*\d+\s*章\s*[｜|].*$", text, re.M):
        data["outline"].append(m.group(0).strip())
    parts = re.split(r"^###\s*【(.+?)】\s*$", text, flags=re.M)
    for i in range(1, len(parts) - 1, 2):
        c = {k: "" for k in CHAR_FIELDS}
        c["姓名"] = parts[i].strip()
        for line in parts[i + 1].splitlines():
            m = re.match(r"[-*\s]*([^：:]+)[：:]\s*(.+)", line.strip())
            if m and m.group(1).strip() in c and m.group(1).strip() != "姓名":
                c[m.group(1).strip()] = m.group(2).strip()
        data["characters"].append(c)
    return data


# ---------------------------------------------------------------- 主应用

class NovelStudio:
    def __init__(self, root):
        self.root = root
        root.title("小说连载工坊 · 本地离线大模型辅助创作")
        root.geometry("1100x760")

        self.project_dir = None
        self.chars = []              # 人物档案列表
        self.current_char = None     # 当前选中人物原名
        self.rels = []               # 人物关系记录
        self.current_rel = None      # 当前选中关系索引
        self.outline = []            # 章节大纲记录
        self.current_ch = None       # 当前选中章节号
        self.q = queue.Queue()
        self.busy = False
        self.abort_evt = threading.Event()   # 流式生成中止信号

        self._build_ui()
        self.root.after(120, self._poll_queue)

    # ------------------------------------------------ 界面搭建

    def _build_ui(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=6)
        self.nb = nb

        self.tab_settings = ttk.Frame(nb); nb.add(self.tab_settings, text=" 1.设置与项目 ")
        self.tab_style = ttk.Frame(nb); nb.add(self.tab_style, text=" 2.范文提炼文风 ")
        self.tab_decomp = ttk.Frame(nb); nb.add(self.tab_decomp, text=" 2.5 范文拆解仿写 ")
        self.tab_chars = ttk.Frame(nb); nb.add(self.tab_chars, text=" 3.人物档案 ")
        self.tab_rel = ttk.Frame(nb); nb.add(self.tab_rel, text=" 4.人物关系 ")
        self.tab_outline = ttk.Frame(nb); nb.add(self.tab_outline, text=" 5.剧情大纲 ")
        self.tab_summary = ttk.Frame(nb); nb.add(self.tab_summary, text=" 6.章节摘要 ")
        self.tab_draft = ttk.Frame(nb); nb.add(self.tab_draft, text=" 7.章节正文 ")

        self._build_settings()
        self._build_style()
        self._build_decomp()
        self._build_chars()
        self._build_rel()
        self._build_outline()
        self._build_summary()
        self._build_draft()

        self.status = tk.Label(self.root, text="就绪。请先在『1.设置与项目』新建或打开项目。",
                               anchor="w", bd=1, relief="sunken")
        self.status.pack(fill="x", side="bottom")

    def _entry(self, parent, var=None, width=46, show=None):
        e = ttk.Entry(parent, width=width, textvariable=var, show=show)
        return e

    # ---- Tab1 设置 ----
    def _build_settings(self):
        f = self.tab_settings
        pad = {"padx": 8, "pady": 4}

        box1 = ttk.LabelFrame(f, text=" 项目 ")
        box1.pack(fill="x", **pad)
        ttk.Label(box1, text="当前项目：").grid(row=0, column=0, sticky="w", **pad)
        self.lbl_project = tk.Label(box1, text="（未打开）", fg="#b00", anchor="w")
        self.lbl_project.grid(row=0, column=1, sticky="w", **pad)
        ttk.Button(box1, text="新建项目…", command=self.new_project).grid(row=0, column=2, **pad)
        ttk.Button(box1, text="打开项目…", command=self.open_project).grid(row=0, column=3, **pad)

        box2 = ttk.LabelFrame(f, text=" 本地大模型 API（OpenAI 兼容） ")
        box2.pack(fill="x", **pad)
        self.var_base = tk.StringVar(value=DEFAULT_CFG["base_url"])
        self.var_key = tk.StringVar(value=DEFAULT_CFG["api_key"])
        self.var_model = tk.StringVar(value=DEFAULT_CFG["model"])
        self.var_temp = tk.StringVar(value=str(DEFAULT_CFG["temperature"]))
        self.var_maxtok = tk.StringVar(value=str(DEFAULT_CFG["max_tokens"]))
        rows = [("API 地址", self.var_base, "如 Ollama: http://127.0.0.1:11434/v1 ; LM Studio: http://127.0.0.1:1234/v1"),
                ("API Key(可空)", self.var_key, ""),
                ("模型名称", self.var_model, "如 qwen2.5:14b / deepseek-r1:32b")]
        for i, (lab, var, tip) in enumerate(rows):
            ttk.Label(box2, text=lab + "：").grid(row=i, column=0, sticky="e", **pad)
            self._entry(box2, var).grid(row=i, column=1, sticky="w", **pad)
            if tip:
                tk.Label(box2, text=tip, fg="#888").grid(row=i, column=2, sticky="w")
        ttk.Label(box2, text="温度：").grid(row=3, column=0, sticky="e", **pad)
        self._entry(box2, self.var_temp, width=8).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(box2, text="max_tokens：").grid(row=4, column=0, sticky="e", **pad)
        self._entry(box2, self.var_maxtok, width=8).grid(row=4, column=1, sticky="w", **pad)
        ttk.Button(box2, text="测试连接", command=self.test_connection).grid(row=5, column=1, sticky="w", **pad)
        ttk.Button(box2, text="保存设置到项目", command=self.save_config).grid(row=5, column=2, sticky="w", **pad)

        tip = tk.Text(f, height=12, wrap="word", bg="#fafafa")
        tip.pack(fill="both", expand=True, **pad)
        tip.insert("1.0", """【使用流程】（与上方标签页编号一致）

1. 设置与项目：先启动本地大模型服务（Ollama / LM Studio 等），填好 API 地址与模型名，点「测试连接」；然后「新建项目」。
2. 范文提炼文风：把写得好的范文 txt 导入，AI 自动提炼文风特征，可人工修改保存。
2.5 范文拆解仿写（偷懒通道）：选一篇范文让 AI 分段通读，自动拆成【完整项目】——
    文风特征、人物档案、人物关系、剧情大纲、每章摘要（AI 依原文撰写）、原文按章切分存为正文。
    拆出的项目目录就是模板实例：可填仿写设定（换人名/背景）改写实例，
    之后到第 7 页对任意章点「AI 生成本章正文」即得仿写小说 —— 即「喂进范文，产出仿写」。
3. 人物档案：逐个添加出场人物（姓名、性格、语言特点、动机等），自动保存为 characters.json。
4. 人物关系：填写人物关系（亲属/上下级/朋友；明面关系与隐藏面；双方称谓）。
5. 剧情大纲：逐章填写简版大纲（时间/人物/地点/事件/看点/钩子）。
6. 章节摘要：AI 联合文风+人物+关系+大纲，扩写出每章摘要；请逐章检查、修改后保存（可让 AI 按你的指令修订）。
7. 章节正文：AI 参考文风+人物+本章摘要+上一章结尾，逐章生成正文；修订后保存，最后一键导出全文。

【文件保存在】 projects/<项目名>/ 目录下，全部为可读的 md/json 文本，可用记事本随时查看修改。

【提示】生成较慢属正常（本地推理），状态栏会显示进度；生成期间请勿重复点击。""")
        tip.config(state="disabled")

    # ---- Tab2 范文 ----
    def _build_style(self):
        f = self.tab_style
        left = ttk.LabelFrame(f, text=" 范文列表（txt） ")
        left.pack(side="left", fill="y", padx=8, pady=8)
        self.lst_samples = tk.Listbox(left, width=34, exportselection=False)
        self.lst_samples.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(left, text="添加范文…", command=self.add_samples).pack(fill="x", padx=4, pady=2)
        ttk.Button(left, text="删除所选范文", command=self.del_sample).pack(fill="x", padx=4, pady=2)
        ttk.Button(left, text="AI 提炼文风特征", command=self.extract_style).pack(fill="x", padx=4, pady=(10, 2))

        right = ttk.LabelFrame(f, text=" 文风特征说明（AI 提炼后可人工修改，正文生成时将严格模仿） ")
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.txt_style = tk.Text(right, wrap="word")
        self.txt_style.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(right, text="保存文风特征", command=self.save_style).pack(anchor="e", padx=4, pady=4)

    # ---- Tab2.5 范文拆解仿写 ----
    def _build_decomp(self):
        f = self.tab_decomp
        top = ttk.LabelFrame(f, text=" 第①步：范文通读拆解 —— 拆成完整项目：文风/人物/关系/大纲 + 逐章摘要 + 原文按章归档 ")
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="范文：").pack(side="left")
        self.cmb_dsample = ttk.Combobox(top, width=28, state="readonly",
                                        postcommand=self._refresh_decomp_samples)
        self.cmb_dsample.pack(side="left", padx=(0, 6))
        ttk.Button(top, text="导入范文…", command=self.import_sample_decomp).pack(side="left")
        ttk.Button(top, text="AI 通读拆解", command=self.deconstruct).pack(side="left", padx=10)
        ttk.Button(top, text="保存拆解实例", command=self.save_instance).pack(side="left")

        row2 = ttk.LabelFrame(f, text=" 第②步（可选）：仿写设定 —— 换人名/背景/时代等，保留情节结构；不改可跳过 ")
        row2.pack(fill="x", padx=8, pady=2)
        self.var_reuse = tk.StringVar()
        ttk.Entry(row2, textvariable=self.var_reuse, width=64).pack(side="left", padx=4, pady=4)
        ttk.Button(row2, text="AI 按仿写设定改写实例", command=self.rewrite_instance).pack(side="left", padx=4)

        row3 = ttk.Frame(f)
        row3.pack(fill="x", padx=8, pady=2)
        ttk.Button(row3, text="第③步：应用到项目素材（应用后到第 3/4 页核对，第 5/6 页即生成仿写小说）",
                   command=self.apply_instance).pack(side="left")
        tk.Label(row3, text="  实例中的 json 块供程序读取，请勿删除。", fg="#888").pack(side="left")

        ttk.Label(f, text="拆解实例（Markdown，可直接编辑）：").pack(anchor="w", padx=8, pady=(4, 0))
        self.txt_decomp = tk.Text(f, wrap="word")
        self.txt_decomp.pack(fill="both", expand=True, padx=8, pady=4)

    # ---- Tab3 人物 ----
    def _build_chars(self):
        f = self.tab_chars
        left = ttk.LabelFrame(f, text=" 人物列表 ")
        left.pack(side="left", fill="y", padx=8, pady=8)
        self.lst_chars = tk.Listbox(left, width=26, exportselection=False)
        self.lst_chars.pack(fill="both", expand=True, padx=4, pady=4)
        self.lst_chars.bind("<<ListboxSelect>>", self._on_char_select)
        ttk.Button(left, text="新增人物", command=self.new_char).pack(fill="x", padx=4, pady=2)
        ttk.Button(left, text="保存人物", command=self.save_char).pack(fill="x", padx=4, pady=2)
        ttk.Button(left, text="删除人物", command=self.del_char).pack(fill="x", padx=4, pady=2)

        right = ttk.LabelFrame(f, text=" 人物档案（多行输入，可回车换行；★ 性格/语言特点/动机 对生成质量影响最大） ")
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)
        self.char_texts = {}
        short = {"姓名", "性别", "年龄", "身高体型"}
        for i, key in enumerate(CHAR_FIELDS):
            r, c = divmod(i, 2)
            ttk.Label(right, text=key + "：").grid(row=r, column=c * 2, sticky="ne", padx=(8, 2), pady=3)
            txt = tk.Text(right, width=36, height=2 if key in short else 3, wrap="word")
            txt.grid(row=r, column=c * 2 + 1, sticky="nsew", padx=(0, 8), pady=3)
            self.char_texts[key] = txt
        for col in (1, 3):
            right.columnconfigure(col, weight=1)
        for row in range((len(CHAR_FIELDS) + 1) // 2):
            right.rowconfigure(row, weight=1)
        tip = tk.Label(right, text="提示：修改姓名后保存即视为重命名；每项改动后请点左侧「保存人物」。",
                       fg="#888")
        tip.grid(row=(len(CHAR_FIELDS) + 1) // 2, column=0, columnspan=4, sticky="w", padx=8, pady=6)

    # ---- Tab4 人物关系（结构化 JSON 表格式编辑） ----
    def _build_rel(self):
        f = self.tab_rel
        left = ttk.LabelFrame(f, text=" 人物关系（relationships.json：甲方/乙方/关系类型/性质/称谓） ")
        left.pack(fill="both", expand=True, padx=8, pady=8)
        body = ttk.Frame(left)
        body.pack(fill="both", expand=True, padx=4, pady=4)
        self.lst_rels = tk.Listbox(body, width=30, exportselection=False)
        self.lst_rels.pack(side="left", fill="y", padx=(0, 8))
        self.lst_rels.bind("<<ListboxSelect>>", self._on_rel_select)
        rform = ttk.Frame(body)
        rform.pack(side="left", fill="both", expand=True)
        self.rel_texts = {}
        for i, key in enumerate(REL_FIELDS):
            r, c = divmod(i, 2)
            ttk.Label(rform, text=key + "：").grid(row=r, column=c * 2, sticky="ne", padx=(4, 2), pady=3)
            txt = tk.Text(rform, width=28, height=3 if key == "备注" else 2, wrap="word")
            txt.grid(row=r, column=c * 2 + 1, sticky="nsew", padx=(0, 8), pady=3)
            self.rel_texts[key] = txt
        for col in (1, 3):
            rform.columnconfigure(col, weight=1)
        for row in range((len(REL_FIELDS) + 1) // 2):
            rform.rowconfigure(row, weight=1)
        rbtns = ttk.Frame(left)
        rbtns.pack(fill="x", padx=4, pady=2)
        ttk.Button(rbtns, text="新增关系", command=self.new_rel).pack(side="left")
        ttk.Button(rbtns, text="保存关系", command=self.save_rel).pack(side="left", padx=4)
        ttk.Button(rbtns, text="删除关系", command=self.del_rel).pack(side="left", padx=8)
        tk.Label(rbtns, text="建议写明：明面关系与隐藏面、双方怎么互相称呼", fg="#888").pack(side="left")

    # ---- Tab5 剧情大纲（结构化 JSON 表格式编辑） ----
    def _build_outline(self):
        f = self.tab_outline
        right = ttk.LabelFrame(f, text=" 剧情大纲（outline.json：逐章结构化记录） ")
        right.pack(fill="both", expand=True, padx=8, pady=8)
        body2 = ttk.Frame(right)
        body2.pack(fill="both", expand=True, padx=4, pady=4)
        self.lst_outline = tk.Listbox(body2, width=30, exportselection=False)
        self.lst_outline.pack(side="left", fill="y", padx=(0, 8))
        self.lst_outline.bind("<<ListboxSelect>>", self._on_outline_select)
        oform = ttk.Frame(body2)
        oform.pack(side="left", fill="both", expand=True)
        self.outline_texts = {}
        tall = {"事件", "冲突/看点", "结尾钩子", "备注", "标题"}
        for i, key in enumerate(OUTLINE_FIELDS):
            r, c = divmod(i, 2)
            ttk.Label(oform, text=key + "：").grid(row=r, column=c * 2, sticky="ne", padx=(4, 2), pady=3)
            txt = tk.Text(oform, width=28, height=3 if key in tall else 2, wrap="word")
            txt.grid(row=r, column=c * 2 + 1, sticky="nsew", padx=(0, 8), pady=3)
            self.outline_texts[key] = txt
        for col in (1, 3):
            oform.columnconfigure(col, weight=1)
        for row in range((len(OUTLINE_FIELDS) + 1) // 2):
            oform.rowconfigure(row, weight=1)
        obtns = ttk.Frame(right)
        obtns.pack(fill="x", padx=4, pady=2)
        ttk.Button(obtns, text="新增章节", command=self.new_outline).pack(side="left")
        ttk.Button(obtns, text="保存章节", command=self.save_outline).pack(side="left", padx=4)
        ttk.Button(obtns, text="删除章节", command=self.del_outline).pack(side="left", padx=8)
        tk.Label(obtns, text="保存后自动按章节号排序重编号（与 summaries/drafts 对齐）",
                 fg="#888").pack(side="left")

    # ---- Tab4 人物关系 ----
    def _load_rels(self):
        P = self._project_paths()
        md = os.path.join(P["materials"], "relationships.md")
        if not os.path.exists(P["rel"]) and os.path.exists(md):
            write_text(P["rel"], json.dumps(parse_rel_string(read_text(md)),
                                            ensure_ascii=False, indent=2))
        self.rels = []
        try:
            data = json.loads(read_text(P["rel"], "[]"))
            if isinstance(data, list):
                self.rels = [{k: str(r.get(k, "") or "") for k in REL_FIELDS}
                             for r in data if isinstance(r, dict)]
        except Exception:
            pass
        self._refresh_rel_list()

    def _refresh_rel_list(self):
        self.lst_rels.delete(0, "end")
        for r in self.rels:
            self.lst_rels.insert("end", "%s — %s %s" % (r.get("甲方", "?"), r.get("乙方", "?"),
                                                        r.get("关系类型", "")))

    def _on_rel_select(self, _e=None):
        sel = self.lst_rels.curselection()
        if not sel:
            return
        self.current_rel = sel[0]
        rec = self.rels[sel[0]]
        for k, txt in self.rel_texts.items():
            txt.delete("1.0", "end")
            txt.insert("1.0", rec.get(k, ""))

    def new_rel(self):
        self.current_rel = None
        for txt in self.rel_texts.values():
            txt.delete("1.0", "end")

    def save_rel(self):
        if not self._ensure_project():
            return
        rec = {k: txt.get("1.0", "end").strip() for k, txt in self.rel_texts.items()}
        if not rec["甲方"] or not rec["乙方"]:
            messagebox.showwarning("提示", "甲方、乙方不能为空。")
            return
        if self.current_rel is not None and 0 <= self.current_rel < len(self.rels):
            self.rels[self.current_rel] = rec
        else:
            self.rels.append(rec)
            self.current_rel = len(self.rels) - 1
        write_text(self._project_paths()["rel"], json.dumps(self.rels, ensure_ascii=False, indent=2))
        self._refresh_rel_list()
        self.lst_rels.selection_set(self.current_rel)
        self.set_status("关系『%s — %s』已保存（共 %d 条）。"
                        % (rec["甲方"], rec["乙方"], len(self.rels)))

    def del_rel(self):
        sel = self.lst_rels.curselection()
        if not sel:
            return
        if not messagebox.askyesno("确认", "删除所选关系？"):
            return
        self.rels.pop(sel[0])
        write_text(self._project_paths()["rel"], json.dumps(self.rels, ensure_ascii=False, indent=2))
        self.new_rel()
        self._refresh_rel_list()

    # ---- Tab5 大纲 ----
    def _load_outline(self):
        P = self._project_paths()
        md = os.path.join(P["materials"], "outline.md")
        if not os.path.exists(P["outline"]) and os.path.exists(md):
            write_text(P["outline"], json.dumps(to_outline_records(
                read_text(md).splitlines()), ensure_ascii=False, indent=2))
        self.outline = []
        try:
            data = json.loads(read_text(P["outline"], "[]"))
            if isinstance(data, list):
                self.outline = to_outline_records(data)
        except Exception:
            pass
        self._refresh_outline_list()

    def _refresh_outline_list(self):
        self.lst_outline.delete(0, "end")
        for r in self.outline:
            self.lst_outline.insert("end", "第%s章 %s" % (r.get("章节"), r.get("标题", "")))

    def _on_outline_select(self, _e=None):
        sel = self.lst_outline.curselection()
        if not sel:
            return
        self.current_ch = self.outline[sel[0]].get("章节")
        rec = self.outline[sel[0]]
        for k, txt in self.outline_texts.items():
            txt.delete("1.0", "end")
            txt.insert("1.0", str(rec.get(k, "")))

    def new_outline(self):
        nxt = max([r.get("章节", 0) for r in self.outline] or [0]) + 1
        self.current_ch = None
        for txt in self.outline_texts.values():
            txt.delete("1.0", "end")
        self.outline_texts["章节"].insert("1.0", str(nxt))

    def save_outline(self):
        if not self._ensure_project():
            return
        rec = {k: txt.get("1.0", "end").strip() for k, txt in self.outline_texts.items()}
        try:
            rec["章节"] = int(re.findall(r"\d+", rec["章节"])[0])
        except Exception:
            messagebox.showwarning("提示", "章节号必须为数字。")
            return
        found = None
        for i, r in enumerate(self.outline):
            if r.get("章节") == rec["章节"]:
                found = i
                break
        if found is None:
            self.outline.append(rec)
        else:
            self.outline[found] = rec
        self.outline = to_outline_records(self.outline)   # 排序 + 重编号
        self.current_ch = rec["章节"] if rec["章节"] <= len(self.outline) else None
        write_text(self._project_paths()["outline"], json.dumps(self.outline, ensure_ascii=False, indent=2))
        self._refresh_outline_list()
        self.set_status("第 %d 章大纲已保存（共 %d 章）。" % (rec["章节"], len(self.outline)))

    def del_outline(self):
        sel = self.lst_outline.curselection()
        if not sel:
            return
        if not messagebox.askyesno("确认", "删除所选章节大纲？"):
            return
        self.outline.pop(sel[0])
        self.outline = to_outline_records(self.outline)
        write_text(self._project_paths()["outline"], json.dumps(self.outline, ensure_ascii=False, indent=2))
        self.new_outline()
        self._refresh_outline_list()

    # ---- Tab5 摘要 ----
    def _build_summary(self):
        f = self.tab_summary
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="规划章数：").pack(side="left")
        self.var_n = tk.StringVar(value="10")
        ttk.Spinbox(top, from_=1, to=200, width=5, textvariable=self.var_n).pack(side="left", padx=(0, 12))
        self.var_overwrite = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="覆盖已有摘要", variable=self.var_overwrite).pack(side="left")
        ttk.Button(top, text="AI 批量生成全部章节摘要", command=self.gen_all_summaries).pack(side="left", padx=8)
        ttk.Button(top, text="AI 重新生成本章", command=self.gen_one_summary).pack(side="left")

        row2 = ttk.Frame(f)
        row2.pack(fill="x", padx=8, pady=2)
        ttk.Label(row2, text="章节：").pack(side="left")
        self.cmb_sum = ttk.Combobox(row2, width=10, state="readonly",
                                    values=[], postcommand=self._refresh_sum_list)
        self.cmb_sum.pack(side="left", padx=(0, 8))
        self.cmb_sum.bind("<<ComboboxSelected>>", lambda e: self.load_summary())
        ttk.Label(row2, text="AI 修订指令（如：把冲突加强，结尾钩子更狠）：").pack(side="left")
        self.var_revise = tk.StringVar()
        ttk.Entry(row2, textvariable=self.var_revise, width=44).pack(side="left", padx=4)
        ttk.Button(row2, text="AI 修订本章摘要", command=self.revise_summary).pack(side="left", padx=4)
        ttk.Button(row2, text="保存本章摘要", command=self.save_summary).pack(side="left", padx=12)

        self.txt_summary = tk.Text(f, wrap="word")
        self.txt_summary.pack(fill="both", expand=True, padx=8, pady=6)

    # ---- Tab6 正文 ----
    def _build_draft(self):
        f = self.tab_draft
        top = ttk.Frame(f)
        top.pack(fill="x", padx=8, pady=6)
        ttk.Label(top, text="章节：").pack(side="left")
        self.cmb_draft = ttk.Combobox(top, width=10, state="readonly",
                                      values=[], postcommand=self._refresh_draft_list)
        self.cmb_draft.pack(side="left", padx=(0, 8))
        self.cmb_draft.bind("<<ComboboxSelected>>", lambda e: self.load_draft())
        ttk.Label(top, text="每章字数目标：").pack(side="left")
        self.var_wc = tk.StringVar(value="2500")
        ttk.Spinbox(top, from_=500, to=10000, increment=100, width=7,
                    textvariable=self.var_wc).pack(side="left", padx=(0, 12))
        self.var_carry = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="携带上一章结尾以衔接", variable=self.var_carry).pack(side="left")
        self.var_scoped = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text="只带本章出场人物（档案/关系按摘要裁剪）",
                        variable=self.var_scoped).pack(side="left", padx=8)
        ttk.Button(top, text="AI 生成本章正文", command=self.gen_draft).pack(side="left", padx=8)
        self.btn_abort = ttk.Button(top, text="中止", command=self.abort_current, state="disabled")
        self.btn_abort.pack(side="left")
        ttk.Button(top, text="保存本章正文", command=self.save_draft).pack(side="left")
        ttk.Button(top, text="导出全文 md", command=self.export_all).pack(side="left", padx=8)

        mid = ttk.LabelFrame(f, text=" 本章摘要（只读，在『5.章节摘要』页修改） ")
        mid.pack(fill="x", padx=8, pady=4)
        self.txt_sum_prev = tk.Text(mid, wrap="word", height=6, bg="#fafafa")
        self.txt_sum_prev.pack(fill="x", padx=4, pady=4)

        bot = ttk.LabelFrame(f, text=" 本章正文（生成后可在此直接编辑，保存后写入 drafts/） ")
        bot.pack(fill="both", expand=True, padx=8, pady=4)
        self.txt_draft = tk.Text(bot, wrap="word")
        self.txt_draft.pack(fill="both", expand=True, padx=4, pady=4)

    # ------------------------------------------------ 项目管理

    def _project_paths(self):
        p = self.project_dir
        return {
            "config": os.path.join(p, "project.json"),
            "samples": os.path.join(p, "samples"),
            "materials": os.path.join(p, "materials"),
            "style": os.path.join(p, "materials", "style_notes.md"),
            "chars": os.path.join(p, "materials", "characters.json"),
            "rel": os.path.join(p, "materials", "relationships.json"),
            "outline": os.path.join(p, "materials", "outline.json"),
            "instance": os.path.join(p, "materials", "deconstruct_instance.md"),
            "summaries": os.path.join(p, "chapters", "summaries"),
            "drafts": os.path.join(p, "chapters", "drafts"),
            "export": os.path.join(p, "chapters", "全文.md"),
        }

    def _init_project_dirs(self, pdir):
        for sub in ["samples", os.path.join("materials"), os.path.join("chapters", "summaries"),
                    os.path.join("chapters", "drafts")]:
            os.makedirs(os.path.join(pdir, sub), exist_ok=True)
        cfgp = os.path.join(pdir, "project.json")
        if not os.path.exists(cfgp):
            write_text(cfgp, json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2))
        for name, default in [("style_notes.md", "（尚未提炼文风。先在『2.范文提炼文风』页添加范文并点「AI 提炼文风特征」。）"),
                              ("relationships.json", json.dumps(
                                  [{k: "" for k in REL_FIELDS}], ensure_ascii=False, indent=2)),
                              ("outline.json", json.dumps(
                                  [{k: (1 if k == "章节" else "") for k in OUTLINE_FIELDS}],
                                  ensure_ascii=False, indent=2))]:
            fp = os.path.join(pdir, "materials", name)
            if not os.path.exists(fp):
                write_text(fp, default)

    def new_project(self):
        name = simpledialog.askstring("新建项目", "请输入项目名（书名/系列名）：", parent=self.root)
        if not name:
            return
        name = re.sub(r'[\\/:*?"<>|]', "_", name.strip())
        os.makedirs(PROJECTS_DIR, exist_ok=True)
        pdir = os.path.join(PROJECTS_DIR, name)
        if os.path.exists(pdir):
            if not messagebox.askyesno("提示", "项目已存在，是否直接打开它？"):
                return
        else:
            self._init_project_dirs(pdir)
        self._load_project(pdir)

    def open_project(self):
        d = filedialog.askdirectory(initialdir=PROJECTS_DIR if os.path.exists(PROJECTS_DIR) else APP_DIR,
                                    title="选择项目目录（若无 project.json 会自动初始化）")
        if not d:
            return
        self._init_project_dirs(d)
        self._load_project(d)

    def _load_project(self, pdir):
        self.project_dir = pdir
        P = self._project_paths()
        cfg = dict(DEFAULT_CFG)
        try:
            cfg.update(json.loads(read_text(P["config"], "{}")))
        except Exception:
            pass
        self.var_base.set(cfg["base_url"]); self.var_key.set(cfg.get("api_key", ""))
        self.var_model.set(cfg["model"]); self.var_temp.set(str(cfg["temperature"]))
        self.var_maxtok.set(str(cfg["max_tokens"]))

        self.lbl_project.config(text=pdir, fg="#070")
        self.chars = []
        try:
            loaded = json.loads(read_text(P["chars"], "[]"))
            if isinstance(loaded, list):
                self.chars = loaded
        except Exception:
            pass
        self._refresh_char_list()
        self._load_rels()
        self._load_outline()
        self.txt_style.delete("1.0", "end"); self.txt_style.insert("1.0", read_text(P["style"]))
        self.txt_decomp.delete("1.0", "end")
        self.txt_decomp.insert("1.0", read_text(P["instance"]))
        self._refresh_samples()
        self._refresh_decomp_samples()
        self._refresh_sum_list(); self._refresh_draft_list()
        self.set_status("项目已打开：" + pdir)

    def save_config(self):
        if not self._ensure_project():
            return
        P = self._project_paths()
        write_text(P["config"], json.dumps(self.collect_config(), ensure_ascii=False, indent=2))
        self.set_status("设置已保存到 project.json")

    def collect_config(self):
        return {
            "base_url": self.var_base.get().strip(),
            "api_key": self.var_key.get().strip(),
            "model": self.var_model.get().strip(),
            "temperature": self.var_temp.get().strip() or "0.8",
            "max_tokens": self.var_maxtok.get().strip() or "8192",
        }

    def _ensure_project(self):
        if not self.project_dir:
            messagebox.showwarning("提示", "请先在『1.设置与项目』页新建或打开项目。")
            return False
        return True

    def set_status(self, text):
        self.status.config(text=text)

    # ------------------------------------------------ 异步 LLM 调用

    def llm_async(self, system, user, callback, taskname="AI 处理中",
                  on_chunk=None, on_start=None):
        """异步调用 LLM。传入 on_chunk 即开启流式：每个增量片段经队列转到
        界面线程回调 on_chunk(delta)，实现边生成边显示；abort 事件可由
        「中止」按钮置位，worker 读到后立即停止读取（已生成部分保留）。"""
        if not self._ensure_project():
            return
        if self.busy:
            messagebox.showinfo("提示", "上一次 AI 任务还在进行中，请稍候。")
            return
        cfg = self.collect_config()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        self.busy = True
        self.abort_evt.clear()
        if on_chunk:
            self._refresh_abort_btn()   # 仅流式任务可中止；非流式任务按钮保持禁用
        if on_start is not None:
            on_start()
        self.set_status(taskname + "…（本地推理中，请耐心等待）")

        def worker():
            try:
                def _relay(delta):
                    self.q.put(("chunk", delta, on_chunk))
                content = call_llm(cfg, messages, stream=bool(on_chunk),
                                   on_chunk=_relay if on_chunk else None,
                                   abort=self.abort_evt)
                if self.abort_evt.is_set():
                    self.q.put(("aborted", content, callback))
                else:
                    self.q.put(("ok", content, callback))
            except Exception as e:
                if self.abort_evt.is_set():
                    self.q.put(("aborted", "", callback))
                else:
                    self.q.put(("err", "%s\n\n请检查：本地模型服务是否启动？API 地址/模型名是否正确？" % e, None))

        threading.Thread(target=worker, daemon=True).start()

    def abort_current(self):
        """中止正在进行的流式生成（已生成部分保留在编辑器，可继续编辑/保存）。"""
        if not self.busy:
            return
        self.abort_evt.set()
        self.set_status("正在中止…（已生成的部分会保留）")
        self._refresh_abort_btn()

    def _refresh_abort_btn(self):
        btn = getattr(self, "btn_abort", None)
        if btn is not None:
            try:
                btn.config(state="normal" if self.busy else "disabled")
            except Exception:
                pass

    def test_connection(self):
        cfg = self.collect_config()
        if not cfg["base_url"] or not cfg["model"]:
            messagebox.showwarning("提示", "请先填写 API 地址与模型名称。")
            return
        self.busy = True
        self.set_status("测试连接中…")

        def worker():
            try:
                call_llm(cfg, [{"role": "user", "content": "请只回复两个字：连接成功"}])
                self.q.put(("ok", "OK", lambda _: messagebox.showinfo(
                    "成功", "连接成功！模型 %s 可用。" % cfg["model"])))
            except Exception as e:
                self.q.put(("err", "%s\n\n请检查：\n1. 本地模型服务是否已启动？\n2. API 地址是否正确？\n3. 模型名是否正确？" % e, None))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind, payload = item[0], item[1]
                cb = item[2] if len(item) > 2 else None
                if kind == "status":
                    self.set_status(payload)
                    continue
                if kind == "chunk":
                    # 流式增量：界面线程直接插入文本框，任务仍在进行（不动 busy）
                    if cb is not None:
                        try:
                            cb(payload)
                        except Exception:
                            pass
                    continue
                self.busy = False
                self._refresh_abort_btn()
                if kind == "ok":
                    cb(payload)
                    self.set_status("完成。")
                elif kind == "aborted":
                    if cb is not None:
                        cb(payload)
                    self.set_status("已中止，已保留已生成的内容（可继续编辑后保存）。")
                else:
                    messagebox.showerror("调用失败", str(payload))
                    self.set_status("调用失败。")
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    # ------------------------------------------------ 素材上下文

    def _style_text(self):
        return clip(self.txt_style.get("1.0", "end")) if self.project_dir else "（无）"

    def _chars_text(self, names=None):
        """人物档案文本；names 非空时只输出名单内人物（按本章摘要命中过滤）。"""
        chars = self.chars
        if names is not None:
            ns = {n.strip() for n in names if n and n.strip()}
            chars = [c for c in self.chars if (c.get("姓名") or "").strip() in ns]
            if not chars:
                return "（本章未检测到已录入人物，请径直按摘要写作）"
        if not chars:
            return "（尚未录入人物）"
        lines = []
        for c in chars:
            lines.append("【%s】" % c.get("姓名", "未命名"))
            for k in CHAR_FIELDS[1:]:
                v = c.get(k, "")
                if v:
                    lines.append("  %s：%s" % (k, v))
        return clip("\n".join(lines))

    def _rel_text(self, names=None):
        """人物关系文本；names 非空时只输出双方均在名单内的关系（一方未出场不带入，避免诱导 AI 写未出场角色）。"""
        rels = list(getattr(self, "rels", None) or [])
        if names is not None:
            ns = {n.strip() for n in names if n and n.strip()}
            rels = [r for r in rels
                    if (r.get("甲方") or "").strip() in ns and (r.get("乙方") or "").strip() in ns]
            if not rels:
                return "（本章未检测到相关人物关系）"
        if not rels:
            return "（尚未录入人物关系）"
        lines = []
        for r in rels:
            parts = ["%s —— %s：%s" % (r.get("甲方", ""), r.get("乙方", ""), r.get("关系类型", ""))]
            if r.get("关系性质"):
                parts.append("（%s）" % r["关系性质"])
            if r.get("甲方对乙方称谓"):
                parts.append("%s称%s为「%s」" % (r["甲方"], r["乙方"], r["甲方对乙方称谓"]))
            if r.get("乙方对甲方称谓"):
                parts.append("%s称%s为「%s」" % (r["乙方"], r["甲方"], r["乙方对甲方称谓"]))
            if r.get("备注"):
                parts.append("备注：" + r["备注"])
            lines.append("，".join(parts))
        return clip("\n".join(lines))

    def _names_in_summary(self, text):
        """检测本章摘要中出现了哪些已录入人物（按姓名/称谓匹配，供按需裁剪提示词）。"""
        if not text or not self.chars:
            return []
        # 称谓 -> 本名 的反查表（摘要里可能用「陈哥」称呼而非全名）
        alias = {}
        for r in getattr(self, "rels", None) or []:
            a, b = (r.get("甲方") or "").strip(), (r.get("乙方") or "").strip()
            if a and r.get("甲方对乙方称谓"):
                alias.setdefault(r["甲方对乙方称谓"].strip(), set()).add(b)
            if b and r.get("乙方对甲方称谓"):
                alias.setdefault(r["乙方对甲方称谓"].strip(), set()).add(a)
        hit = []
        for c in self.chars:
            name = (c.get("姓名") or "").strip()
            if not name:
                continue
            if name in text:
                hit.append(name)
                continue
            # 姓名未直接出现时，查是否有指向它的称谓出现
            for al, targets in alias.items():
                if al and al in text and name in targets:
                    hit.append(name)
                    break
        return hit

    def _outline_text(self):
        if not getattr(self, "outline", None):
            return "（尚未录入剧情大纲）"
        lines = []
        for r in self.outline:
            lines.append("第%s章｜%s｜%s｜%s｜%s｜%s｜%s｜%s"
                         % (r.get("章节"), r.get("标题"), r.get("时间"), r.get("人物"),
                            r.get("地点"), r.get("事件"), r.get("冲突/看点"), r.get("结尾钩子")))
        return clip("\n".join(lines), 12000)

    # ------------------------------------------------ Tab2 范文

    def _refresh_samples(self):
        self.lst_samples.delete(0, "end")
        if not self.project_dir:
            return
        sdir = self._project_paths()["samples"]
        if not os.path.isdir(sdir):
            return
        for fn in sorted(os.listdir(sdir)):
            self.lst_samples.insert("end", fn)

    def add_samples(self):
        if not self._ensure_project():
            return
        files = filedialog.askopenfilenames(filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
        sdir = self._project_paths()["samples"]
        for fp in files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                with open(fp, "r", encoding="gbk", errors="replace") as f:
                    text = f.read()
            write_text(os.path.join(sdir, os.path.basename(fp)), text)
        self._refresh_samples()
        self.set_status("已导入 %d 篇范文。" % len(files))

    def del_sample(self):
        if not self._ensure_project():
            return
        sel = self.lst_samples.curselection()
        if not sel:
            return
        fn = self.lst_samples.get(sel[0])
        if messagebox.askyesno("确认", "删除范文《%s》？（只从项目移除，不影响原文件已复制的内容）" % fn):
            os.remove(os.path.join(self._project_paths()["samples"], fn))
            self._refresh_samples()

    def extract_style(self):
        if not self._ensure_project():
            return
        sdir = self._project_paths()["samples"]
        files = sorted(os.listdir(sdir))
        if not files:
            messagebox.showwarning("提示", "请先添加范文 txt。")
            return
        parts = []
        for fn in files:
            parts.append("《%s》\n%s" % (fn, clip(read_text(os.path.join(sdir, fn)), 8000)))
        system = "你是一位资深小说编辑，擅长从范文中提炼可模仿的写作特征。"
        user = ("请阅读以下范文，提炼『文风特征说明』，供 AI 后续模仿写作使用。"
                "必须涵盖：1 叙事视角与人称；2 句式与语言风格；3 对话风格与占比；"
                "4 节奏与场景切换；5 人物刻画手法；6 常用词汇与标志性表达；"
                "7 情感基调；8 写作时应避免之处。条目化中文输出，800 字以内。\n\n"
                + "\n\n====\n\n".join(parts))
        self.llm_async(system, user, self._after_extract, "提炼文风特征")

    def _after_extract(self, content):
        self.txt_style.delete("1.0", "end")
        self.txt_style.insert("1.0", content.strip())
        self.save_style()

    def save_style(self):
        if not self._ensure_project():
            return
        write_text(self._project_paths()["style"], self.txt_style.get("1.0", "end"))
        self.set_status("文风特征已保存。")

    # ------------------------------------------------ Tab2.5 拆解仿写

    def _refresh_decomp_samples(self):
        vals = []
        if self.project_dir:
            sdir = self._project_paths()["samples"]
            if os.path.isdir(sdir):
                vals = sorted(fn for fn in os.listdir(sdir) if os.path.isfile(os.path.join(sdir, fn)))
        self.cmb_dsample["values"] = vals
        if vals and self.cmb_dsample.get() not in vals:
            self.cmb_dsample.set(vals[0])

    def import_sample_decomp(self):
        if not self._ensure_project():
            return
        fp = filedialog.askopenfilename(filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
        if not fp:
            return
        fn = os.path.basename(fp)
        write_text(os.path.join(self._project_paths()["samples"], fn), read_any(fp))
        self._refresh_samples()
        self._refresh_decomp_samples()
        self.cmb_dsample.set(fn)
        self.set_status("已导入范文：%s" % fn)

    def save_instance(self):
        if not self._ensure_project():
            return
        write_text(self._project_paths()["instance"], self.txt_decomp.get("1.0", "end"))
        self.set_status("拆解实例已保存到 materials/deconstruct_instance.md")

    def deconstruct(self):
        if not self._ensure_project():
            return
        if self.busy:
            messagebox.showinfo("提示", "上一次 AI 任务还在进行中，请稍候。")
            return
        fn = self.cmb_dsample.get()
        if not fn:
            messagebox.showwarning("提示", "请先选择或导入范文 txt。")
            return
        cfg = self.collect_config()
        try:
            cfg["max_tokens"] = max(8192, int(cfg["max_tokens"]))
        except ValueError:
            cfg["max_tokens"] = 8192
        full_text = read_any(os.path.join(self._project_paths()["samples"], fn))
        if not full_text.strip():
            messagebox.showwarning("提示", "范文内容为空。")
            return
        self.busy = True
        self.set_status("开始通读拆解《%s》…" % fn)

        def worker():
            try:
                chapters = split_chapters(full_text)
                head_lines = ""
                if len(chapters) >= 2:
                    titles = []
                    for i, c in enumerate(chapters[:80], 1):
                        titles.append("第%d章：%s" % (i, c.splitlines()[0].strip()[:40]))
                    head_lines = ("\n\n=== 原文检测到的章节划分（共 %d 章，"
                                  "剧情大纲必须严格按此划分，一章一条） ===\n%s"
                                  % (len(chapters), "\n".join(titles)))
                chunks = split_chunks(full_text)
                analyses = []
                for i, ch in enumerate(chunks):
                    self.q.put(("status", "AI 通读第 %d/%d 段…" % (i + 1, len(chunks)), None))
                    analyses.append(call_llm(cfg, [
                        {"role": "system", "content": "你是资深小说编辑，正在为范文拆解做准备，只做客观提取。"},
                        {"role": "user", "content": (
                            "以下是小说《%s》的第 %d/%d 段。请提取：\n"
                            "1. 本段出场人物：姓名、身份职务、外貌/性格/语言风格线索（尽量引用原文依据）；\n"
                            "2. 人物关系及其变化；\n"
                            "3. 本段情节：按场景顺序逐条列出，每条一行：时间｜人物｜地点｜事件｜冲突或看点｜留下的悬念。\n"
                            "只输出提取内容，不要评论。\n\n%s"
                            % (fn, i + 1, len(chunks), ch))}]))
                self.q.put(("status", "分段通读完成，正在汇总拆解…", None))
                merged = call_llm(cfg, [
                    {"role": "system", "content": "你是资深小说编辑，负责把范文完整拆解为可复用的创作模板。"},
                    {"role": "user", "content": (
                        "以下是范文《%s》的分段分析结果，以及原文开头（供提炼文风）。\n\n"
                        "=== 原文开头 ===\n%s\n\n=== 分段分析 ===\n%s%s\n\n"
                        "请输出【两部分】，不要输出其他内容：\n\n"
                        "【第一部分】范文拆解实例（Markdown），结构如下：\n"
                        "# 范文拆解实例：《%s》\n"
                        "## 一、作品基本盘（题材/篇幅/视角/情感基调）\n"
                        "## 二、文风特征（视角、句式、对话占比、节奏、高频词、需避免之处）\n"
                        "## 三、人物档案（每个主要人物一小节，格式必须为：### 【姓名】，"
                        "其下用『- 字段：内容』列出：性别、年龄、职务/身份、身高体型、样貌、穿着特点、"
                        "性格特点、语言特点/口头禅、典型台词、动机与目标、秘密、其他备注）\n"
                        "## 四、人物关系（明面关系与隐藏面、称谓）\n"
                        "## 五、剧情大纲（严格每章一行：第N章｜时间｜人物｜地点｜事件｜冲突/看点｜结尾钩子。"
                        "若上方给出了原文章节划分，必须逐章对齐；否则按场景/幕切分为章）\n"
                        "## 六、仿写要点（用此模板仿写出同类小说最关键的 10 条规则）\n\n"
                        "【第二部分】紧跟一个 ```json 代码块，供程序读取，字段：\n"
                        '{"style": "文风特征纯文本", "theme": "题材与情感基调", '
                        '"characters": [{"姓名":"", "性别":"", "年龄":"", "职务/身份":"", "身高体型":"", '
                        '"样貌":"", "穿着特点":"", "性格特点":"", "语言特点/口头禅":"", "典型台词":"", '
                        '"动机与目标":"", "秘密":"", "其他备注":""}], '
                        '"relationships": [{"甲方":"", "乙方":"", "关系类型":"", '
                        '"关系性质":"公开/隐藏/伪装", "甲方对乙方称谓":"", "乙方对甲方称谓":"", "备注":""}], '
                        '"outline": [{"章节":1, "标题":"", "时间":"", "人物":"", "地点":"", '
                        '"事件":"", "冲突/看点":"", "结尾钩子":""}]}\n'
                        "JSON 中 characters/relationships/outline 必须与第一部分对应内容一致；"
                        "relationships 为数组，每条一组人物关系；outline 为数组，每条对应一章，"
                        "字段不得缺失（无内容填空字符串）。"
                        % (fn, clip(full_text, 6000), clip("\n\n".join(analyses), 24000),
                           head_lines, fn))}])
                # ---- 写入完整项目目录 ----
                P = self._project_paths()
                self.q.put(("status", "正在写入素材文件（文风/人物/关系/大纲/实例）…", None))
                data = normalize_instance(merged)
                stats = self._write_materials(data)
                write_text(P["instance"], merged.strip())
                # 原文按章节写入 drafts/
                for i, ctext in enumerate(chapters, 1):
                    write_text(os.path.join(P["drafts"], ch_name(i) + ".md"), ctext)
                # AI 逐章生成摘要 -> summaries/
                n_sum = 0
                for i, ctext in enumerate(chapters, 1):
                    spath = os.path.join(P["summaries"], ch_name(i) + ".md")
                    if i > 60:
                        write_text(spath, "（原文摘要未自动生成：超出单次拆解 60 章上限，"
                                          "可到第 6 页单独生成本章摘要。）")
                        continue
                    self.q.put(("status", "生成第 %d/%d 章摘要…" % (i, len(chapters)), None))
                    out = call_llm(cfg, [
                        {"role": "system", "content": "你是资深小说编辑，负责为小说章节撰写规范摘要。"},
                        {"role": "user", "content": (
                            "以下是《%s》第 %d 章（共 %d 章）原文。\n\n%s\n\n"
                            "请写 200~400 字章节摘要，必须包含：时间地点、出场人物、核心事件、"
                            "剧情看点/冲突、结尾钩子。输出格式严格为：第%d章摘要：<内容>（不要其他文字）。"
                            % (fn, i, len(chapters), clip(ctext, 6000), i))}])
                    parsed = parse_summaries(out)
                    write_text(spath, parsed.get(i) or out.strip())
                    n_sum += 1
                self.q.put(("ok", (merged, stats, len(chapters), n_sum), self._after_deconstruct))
            except Exception as e:
                self.q.put(("err", "%s\n\n请检查本地模型服务与 API 设置。" % e, None))

        threading.Thread(target=worker, daemon=True).start()

    def _after_deconstruct(self, payload):
        merged, stats, n_ch, n_sum = payload
        self.txt_decomp.delete("1.0", "end")
        self.txt_decomp.insert("1.0", merged.strip())
        self._reload_material_ui()
        has_json = extract_instance_json(merged) is not None
        msg = ("拆解完成，已生成完整项目目录：\n"
               "· 文风特征、人物档案 %d 人、人物关系、剧情大纲 %d 章（materials/）\n"
               "· 拆解实例 deconstruct_instance.md\n"
               "· 章节正文 %d 章为原文（chapters/drafts/）\n"
               "· AI 章节摘要 %d 份（chapters/summaries/）" % (stats["n_chars"], stats["n_outline"], n_ch, n_sum))
        if stats["n_outline"] and stats["n_outline"] != n_ch:
            msg += ("\n\n注意：AI 大纲为 %d 章，与原文实际章节数（%d）不一致，"
                    "摘要以原文章节为准；可到第 5 页修订大纲。" % (stats["n_outline"], n_ch))
        if not has_json:
            msg += "\n\n注意：未检测到 json 块，本次按 Markdown 兜底解析，人物/大纲可能不全。"
        msg += ("\n\n仿写方法：实例已即模板。到第 7 页选任意章点「AI 生成本章正文」，"
                "AI 会按文风+人物+本章摘要重写出仿写正文（先在第 2.5 页填仿写设定改写实例，可换人名背景）。")
        messagebox.showinfo("完成", msg)

    def rewrite_instance(self):
        if not self._ensure_project():
            return
        inst = self.txt_decomp.get("1.0", "end").strip()
        instr = self.var_reuse.get().strip()
        if not inst:
            messagebox.showwarning("提示", "请先完成『AI 通读拆解』。")
            return
        if not instr:
            messagebox.showwarning("提示", "请先填写仿写设定，例如：主角改为女法医沈青，背景改为滨海市。")
            return
        system = "你是小说改编策划，负责在保留情节结构的前提下替换设定。"
        user = ("以下是一份范文拆解实例（Markdown + json 块）。\n\n%s\n\n"
                "仿写设定：%s\n\n"
                "要求：保持整体结构、章节划分、Markdown 格式与 json 字段完全不变，"
                "只替换人物姓名/身份背景/地点时代等设定，情节结构与冲突点保留；"
                "json 块内容须与正文一致。只输出改写后的完整实例。"
                % (clip(inst, 16000), instr))
        self.llm_async(system, user, self._after_rewrite, "按仿写设定改写实例")

    def _after_rewrite(self, content):
        self.txt_decomp.delete("1.0", "end")
        self.txt_decomp.insert("1.0", content.strip())
        self.save_instance()
        self.set_status("实例已按仿写设定改写并保存。")

    def _write_materials(self, data):
        """把拆解数据落盘为项目素材文件（线程安全，只做文件写入）。"""
        P = self._project_paths()
        chars = [{k: (c.get(k, "") or "") for k in CHAR_FIELDS}
                 for c in data.get("characters", []) if isinstance(c, dict) and c.get("姓名")]
        outline = to_outline_records(data.get("outline"))
        rels = to_rel_records(data.get("relationships"))
        style = (data.get("style") or "").strip()
        theme = (data.get("theme") or "").strip()
        style_md = "## 文风特征\n\n%s\n" % style if style else ""
        if theme:
            style_md += "\n## 题材与情感基调\n\n%s\n" % theme
        if style_md:
            write_text(P["style"], style_md)
        if chars:
            write_text(P["chars"], json.dumps(chars, ensure_ascii=False, indent=2))
        if rels:
            write_text(P["rel"], json.dumps(rels, ensure_ascii=False, indent=2))
        if outline:
            write_text(P["outline"], json.dumps(outline, ensure_ascii=False, indent=2))
        return {"style_len": len(style), "n_chars": len(chars), "n_outline": len(outline),
                "n_rels": len(rels)}

    def _reload_material_ui(self):
        """从磁盘重新加载素材到各标签页界面。"""
        P = self._project_paths()
        self.chars = []
        try:
            loaded = json.loads(read_text(P["chars"], "[]"))
            if isinstance(loaded, list):
                self.chars = loaded
        except Exception:
            pass
        self._refresh_char_list()
        self.new_char()
        self._load_rels()
        self.new_rel()
        self._load_outline()
        self.new_outline()
        self.txt_style.delete("1.0", "end"); self.txt_style.insert("1.0", read_text(P["style"]))
        self._refresh_sum_list()
        self._refresh_draft_list()

    def apply_instance(self):
        if not self._ensure_project():
            return
        text = self.txt_decomp.get("1.0", "end")
        if not text.strip():
            messagebox.showwarning("提示", "当前没有拆解实例。")
            return
        data = normalize_instance(text)
        if not data.get("characters") and not data.get("outline"):
            messagebox.showwarning("提示", "未能从实例中解析出人物或大纲，请检查实例内容（或重新拆解）。")
            return
        stats = self._write_materials(data)
        self._reload_material_ui()
        self.set_status("拆解实例已应用到项目素材：文风 %d 字，人物 %d 人，大纲 %d 章。"
                        % (stats["style_len"], stats["n_chars"], stats["n_outline"]))
        messagebox.showinfo("完成",
                            "已应用到项目素材：\n· 文风特征（第2页）\n· 人物档案 %d 人（第3页，请核对）\n"
                            "· 人物关系 %d 条（第4页）\n· 剧情大纲 %d 章（第5页）\n\n"
                            "现在可到第 6 页生成章节摘要（仿写时建议先修改大纲中的时间地点等设定），"
                            "再到第 7 页逐章生成仿写正文。"
                            % (stats["n_chars"], stats["n_rels"], stats["n_outline"]))

    # ------------------------------------------------ Tab3 人物

    def _refresh_char_list(self):
        self.lst_chars.delete(0, "end")
        for c in self.chars:
            self.lst_chars.insert("end", c.get("姓名", "未命名"))

    def _on_char_select(self, _e=None):
        sel = self.lst_chars.curselection()
        if not sel:
            return
        name = self.lst_chars.get(sel[0])
        for c in self.chars:
            if c.get("姓名") == name:
                self.current_char = name
                for k, txt in self.char_texts.items():
                    txt.delete("1.0", "end")
                    txt.insert("1.0", c.get(k, ""))
                return

    def new_char(self):
        self.current_char = None
        for txt in self.char_texts.values():
            txt.delete("1.0", "end")

    def save_char(self):
        if not self._ensure_project():
            return
        data = {k: txt.get("1.0", "end").strip() for k, txt in self.char_texts.items()}
        data["姓名"] = " ".join(data["姓名"].split())  # 姓名压成单行
        if not data["姓名"]:
            messagebox.showwarning("提示", "姓名不能为空。")
            return
        if self.current_char and self.current_char != data["姓名"]:
            self.chars = [c for c in self.chars if c.get("姓名") != self.current_char]
        for i, c in enumerate(self.chars):
            if c.get("姓名") == data["姓名"]:
                self.chars[i] = data
                break
        else:
            self.chars.append(data)
        write_text(self._project_paths()["chars"], json.dumps(self.chars, ensure_ascii=False, indent=2))
        self.current_char = data["姓名"]
        self._refresh_char_list()
        self.set_status("人物『%s』已保存（共 %d 人）。" % (data["姓名"], len(self.chars)))

    def del_char(self):
        sel = self.lst_chars.curselection()
        if not sel:
            return
        name = self.lst_chars.get(sel[0])
        if messagebox.askyesno("确认", "删除人物『%s』？" % name):
            self.chars = [c for c in self.chars if c.get("姓名") != name]
            write_text(self._project_paths()["chars"], json.dumps(self.chars, ensure_ascii=False, indent=2))
            self.new_char()
            self._refresh_char_list()

    # ------------------------------------------------ Tab5 摘要

    def _summary_path(self, n):
        return os.path.join(self._project_paths()["summaries"], ch_name(n) + ".md")

    def _existing_summaries(self):
        if not self.project_dir:
            return []
        return sorted(int(fn[2:4]) for fn in os.listdir(self._project_paths()["summaries"])
                      if re.match(r"ch\d+\.md$", fn))

    def _refresh_sum_list(self):
        self.cmb_sum["values"] = ["第%d章" % n for n in self._existing_summaries()]

    def _sum_context(self):
        return ("【文风特征】\n%s\n\n【人物档案】\n%s\n\n【人物关系】\n%s\n\n【剧情大纲】\n%s"
                % (self._style_text(), self._chars_text(), self._rel_text(), self._outline_text()))

    def gen_all_summaries(self):
        if not self._ensure_project():
            return
        if self.rels:
            write_text(self._project_paths()["rel"], json.dumps(self.rels, ensure_ascii=False, indent=2))
        if self.outline:
            write_text(self._project_paths()["outline"], json.dumps(self.outline, ensure_ascii=False, indent=2))
        n = int(self.var_n.get())
        sdir = self._project_paths()["summaries"]
        if not self.var_overwrite.get():
            existing = self._existing_summaries()
            if existing:
                if not messagebox.askyesno("确认",
                        "已有 %d 章摘要，勾选了跳过则只补缺失章节。\n继续？" % len(existing)):
                    return
        system = "你是经验丰富的连载小说策划编辑。"
        user = ("%s\n\n任务：规划并撰写第 1 至第 %d 章的章节摘要。"
                "每章摘要 200~400 字，必须包含：时间地点、出场人物、核心事件、"
                "剧情看点/冲突、本章结尾悬念（钩子）。大纲已给到哪章就写到哪章；"
                "若大纲不足 %d 章，请按主线合理延展。\n\n"
                "输出格式必须严格为每章一段（不要其他说明文字）：\n"
                "第1章摘要：<内容>\n第2章摘要：<内容>\n……"
                % (self._sum_context(), n, n))
        self.llm_async(system, user, self._after_gen_all, "批量生成章节摘要")

    def _after_gen_all(self, content):
        parsed = parse_summaries(content)
        if not parsed:
            messagebox.showwarning("提示", "未能从 AI 输出中解析出章节摘要，原文已放入摘要编辑框，"
                                           "请检查模型输出或重试。")
            self.txt_summary.delete("1.0", "end")
            self.txt_summary.insert("1.0", content)
            return
        skip = []
        for n, text in sorted(parsed.items()):
            p = self._summary_path(n)
            if os.path.exists(p) and not self.var_overwrite.get():
                skip.append(n)
                continue
            write_text(p, text)
        self._refresh_sum_list()
        msg = "已写入 %d 章摘要。" % (len(parsed) - len(skip))
        if skip:
            msg += "（第 %s 章已存在被跳过，可勾选「覆盖已有摘要」重试）" % "、".join(map(str, skip))
        self.set_status(msg)
        messagebox.showinfo("完成", msg + "\n\n请务必逐章检查修订后保存。")

    def gen_one_summary(self):
        if not self._ensure_project():
            return
        sel = self.cmb_sum.get()
        if not sel:
            messagebox.showwarning("提示", "请先选择章节。")
            return
        n = int(re.findall(r"\d+", sel)[0])
        system = "你是经验丰富的连载小说策划编辑。"
        user = ("%s\n\n任务：只撰写第 %d 章摘要（300~400 字），包含：时间地点、出场人物、"
                "核心事件、剧情看点/冲突、结尾钩子。需与前文自然衔接。"
                "输出格式：第%d章摘要：<内容>（不要其他说明）。"
                % (self._sum_context(), n, n))
        self.llm_async(system, user, lambda c: self._after_one_summary(n, c), "生成本章摘要")

    def _after_one_summary(self, n, content):
        parsed = parse_summaries(content)
        text = parsed.get(n, content.strip())
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("1.0", text)
        self.set_status("第 %d 章摘要已生成，请检查修订后点「保存本章摘要」。" % n)

    def load_summary(self):
        sel = self.cmb_sum.get()
        if not sel:
            return
        n = int(re.findall(r"\d+", sel)[0])
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("1.0", read_text(self._summary_path(n)))

    def save_summary(self):
        sel = self.cmb_sum.get()
        if not sel:
            messagebox.showwarning("提示", "请先选择章节。")
            return
        n = int(re.findall(r"\d+", sel)[0])
        write_text(self._summary_path(n), self.txt_summary.get("1.0", "end"))
        self.set_status("第 %d 章摘要已保存。" % n)

    def revise_summary(self):
        sel = self.cmb_sum.get()
        if not sel:
            messagebox.showwarning("提示", "请先选择章节。")
            return
        n = int(re.findall(r"\d+", sel)[0])
        instruction = self.var_revise.get().strip()
        if not instruction:
            messagebox.showwarning("提示", "请先填写修订指令。")
            return
        old = self.txt_summary.get("1.0", "end").strip() or read_text(self._summary_path(n))
        system = "你是连载小说编辑，负责按作者要求修订章节摘要。"
        user = ("【原摘要】\n%s\n\n【修订要求】%s\n\n请输出修订后的完整摘要，"
                "保持 200~400 字，保留时间地点/人物/事件/看点/钩子结构，只输出摘要正文。"
                % (old, instruction))
        self.llm_async(system, user, lambda c: self._after_revise(n, c), "修订摘要")

    def _after_revise(self, n, content):
        self.txt_summary.delete("1.0", "end")
        self.txt_summary.insert("1.0", content.strip())
        self.set_status("第 %d 章摘要已按指令修订，确认无误后点「保存本章摘要」。" % n)

    # ------------------------------------------------ Tab6 正文

    def _refresh_draft_list(self):
        if not self.project_dir:
            self.cmb_draft["values"] = []
            return
        nums = self._existing_summaries()
        self.cmb_draft["values"] = ["第%d章" % n for n in nums]
        if nums and not self.cmb_draft.get():
            self.cmb_draft.set("第%d章" % nums[0])

    def _cur_chapter(self):
        sel = self.cmb_draft.get()
        if not sel:
            return None
        return int(re.findall(r"\d+", sel)[0])

    def load_draft(self):
        n = self._cur_chapter()
        if n is None:
            return
        self.txt_sum_prev.config(state="normal")
        self.txt_sum_prev.delete("1.0", "end")
        self.txt_sum_prev.insert("1.0", read_text(self._summary_path(n), "（本章摘要缺失，请回『5.章节摘要』生成）"))
        self.txt_sum_prev.config(state="disabled")
        self.txt_draft.delete("1.0", "end")
        self.txt_draft.insert("1.0", read_text(os.path.join(self._project_paths()["drafts"], ch_name(n) + ".md")))

    def gen_draft(self):
        if not self._ensure_project():
            return
        n = self._cur_chapter()
        if n is None:
            messagebox.showwarning("提示", "请先选择章节（需已生成该章摘要）。")
            return
        summary = read_text(self._summary_path(n))
        if not summary.strip():
            messagebox.showwarning("提示", "该章摘要为空，请先在『5.章节摘要』生成并保存。")
            return
        prev_end = ""
        if n > 1 and self.var_carry.get():
            prev = read_text(os.path.join(self._project_paths()["drafts"], ch_name(n - 1) + ".md"))
            if prev.strip():
                prev_end = "\n\n【前一章结尾】（衔接用，不要重复原句）\n" + clip(prev, 900)
        # 只附带本章摘要中出现的人物（档案+关系），避免 40 个角色全量灌入稀释注意力
        hit = self._names_in_summary(summary) if self.var_scoped.get() else []
        if hit:
            chars_md, rel_md = self._chars_text(hit), self._rel_text(hit)
            scope = "人物 %d/%d：%s" % (len(hit), len(self.chars), "、".join(hit))
        else:
            chars_md, rel_md = self._chars_text(), self._rel_text()
            scope = "未匹配到已录入人物（附全量档案）"
        system = "你是该连载小说的作者，严格按给定文风与人设写作。"
        user = ("【文风特征】（必须严格模仿）\n%s\n\n【人物档案】（言行必须符合人设）\n%s\n\n"
                "【人物关系】\n%s\n\n【第 %d 章摘要】\n%s%s\n\n"
                "写作要求：\n1. 正文约 %s 字；\n"
                "2. 严格按本章摘要展开，不得遗漏关键情节；\n"
                "3. 人物语言符合各自的语言特点；\n"
                "4. 结尾按摘要设置钩子，吸引读者追更；\n"
                "5. 只输出正文，章节标题格式为：第%d章 <标题>，不要任何额外说明。"
                % (self._style_text(), chars_md, rel_md,
                   n, summary, prev_end, self.var_wc.get(), n))
        def _on_start():
            # 流式开始：清空正文框，准备逐块显示
            self.txt_draft.delete("1.0", "end")
            self.set_status("正在流式生成第 %d 章正文（%s）…可随时点「中止」" % (n, scope))

        def _on_chunk(delta):
            # 每个增量片段立即追加到正文框并滚动到底部
            self.txt_draft.insert("end", delta)
            self.txt_draft.see("end")

        self.llm_async(system, user, lambda c: self._after_draft(n, c),
                       "生成第 %d 章正文（%s）" % (n, scope),
                       on_chunk=_on_chunk, on_start=_on_start)

    def _after_draft(self, n, content):
        text = content.strip()
        if text.startswith("第%d章" % n) is False and "第%d章" % n not in text[:30]:
            text = "第%d章\n\n%s" % (n, text)
        # 流式已逐块显示：内容一致就不重插，避免末尾重复
        cur = self.txt_draft.get("1.0", "end").strip()
        if cur != text:
            self.txt_draft.delete("1.0", "end")
            self.txt_draft.insert("1.0", text)
        self.txt_draft.see("end")
        self.set_status("第 %d 章正文已生成，请审阅修改后点「保存本章正文」。" % n)

    def save_draft(self):
        n = self._cur_chapter()
        if n is None:
            return
        write_text(os.path.join(self._project_paths()["drafts"], ch_name(n) + ".md"),
                   self.txt_draft.get("1.0", "end"))
        self.set_status("第 %d 章正文已保存。" % n)

    def export_all(self):
        if not self._ensure_project():
            return
        ddir = self._project_paths()["drafts"]
        files = sorted(fn for fn in os.listdir(ddir) if re.match(r"ch\d+\.md$", fn))
        if not files:
            messagebox.showwarning("提示", "尚无任何已保存的正文。")
            return
        parts = ["# %s\n" % os.path.basename(self.project_dir)]
        for fn in files:
            parts.append("\n\n----\n\n" + read_text(os.path.join(ddir, fn)))
        out = self._project_paths()["export"]
        write_text(out, "".join(parts))
        self.set_status("已导出全文：%s" % out)
        messagebox.showinfo("完成", "已导出全文（%d 章）：\n%s" % (len(files), out))


def main():
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except Exception:
        pass
    root.geometry("1100x760")
    try:
        root.state("zoomed")   # Windows 下默认最大化全屏
    except Exception:
        pass
    NovelStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
