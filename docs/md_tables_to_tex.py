"""把 PAPER_DRAFT.md 里的 markdown 表格转成 booktabs LaTeX（直接粘进 Springer 模板）。

用法：
    python docs/md_tables_to_tex.py    ->  docs/TABLES.tex

规则
----
· 只处理前面有 `**Table X.**` 题注的表格，题注文本一起带过去；
· 列对齐：表头/内容主要是数字的列右对齐，其余左对齐（自动判断）；
· 表格宽度不强制，超出双栏时请在模板里用 \\small 或 \\resizebox。
"""

import re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "PAPER_DRAFT.md"
DST = Path(__file__).resolve().parent / "TABLES.tex"

CAP_RE = re.compile(r"^\*\*(Table [A-Z0-9]+)\.\*\*\s*(.*)$")
NUM_RE = re.compile(r"^[−+\-–—]?[\d.,]*\d$")


def split_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def is_numeric(col: list[str]) -> bool:
    """按每个单元格的第一个词判断：支持 "0.68 ± 0.002"、"(+0.0041"、"**0.5367**"。"""
    vals = [v for v in col if v.strip() and v.strip() != "—"]
    if not vals:
        return False
    hits = 0
    for v in vals:
        v = v.replace("**", "").replace("±", " ").replace(r"$\pm$", " ")
        tok = v.split()[0] if v.split() else ""
        if NUM_RE.match(tok.strip("()")):
            hits += 1
    return hits / len(vals) >= 0.6


def bold_to_tex(s: str) -> str:
    out, open_ = [], False
    i = 0
    while i < len(s):
        if s.startswith("**", i):
            out.append(r"\textbf{" if not open_ else "}")
            open_ = not open_
            i += 2
            continue
        out.append(s[i])
        i += 1
    if open_:
        out.append("}")
    return "".join(out)


def convert(caption_no: str, caption: str, rows: list[list[str]]) -> str:
    head, body = rows[0], rows[1:]
    ncol = len(head)
    align = []
    for j in range(ncol):
        col = [r[j] for r in body if j < len(r)]
        align.append("r" if is_numeric(col) else "l")

    def fmt(cell: str) -> str:
        cell = cell.replace("−", "$-$")
        cell = cell.replace("±", r"$\pm$")
        cell = bold_to_tex(cell)
        cell = cell.replace("%", r"\%").replace("&", r"\&")
        # 保护数学段内的下划线
        cell = re.sub(r"(?<!\$)\$(?!\$)", "\x00", cell)
        cell = cell.replace("_", r"\_")
        cell = cell.replace("\x00", "$")
        return cell

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{fmt(caption)}}}",
        rf"\label{{tab:{caption_no.lower().replace(' ', '')}}}",
        rf"\begin{{tabular}}{{{''.join(align)}}}",
        r"\toprule",
        " & ".join(fmt(c) for c in head) + r" \\",
        r"\midrule",
    ]
    for r in body:
        cells = [fmt(c) if j < len(r) else "" for j, c in enumerate(r)]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    text = SRC.read_text(encoding="utf-8").splitlines()
    out, i, n = [], 0, 0
    while i < len(text):
        m = CAP_RE.match(text[i].strip())
        if m:
            j = i + 1
            while j < len(text) and not text[j].strip().startswith("|"):
                j += 1
            if j < len(text):
                rows, k = [], j
                while k < len(text) and text[k].strip().startswith("|"):
                    line = text[k].strip()
                    if not re.match(r"^\|[\s:\-|]+\|$", line):
                        rows.append(split_row(line))
                    k += 1
                if len(rows) >= 2:
                    caption = " ".join([m.group(2)] +
                                       [t.strip() for t in text[i + 1:j] if t.strip()])
                    out.append(convert(m.group(1), caption, rows))
                    n += 1
                i = k
                continue
        i += 1

    header = ("% 由 docs/md_tables_to_tex.py 自动生成 —— 请勿手改，改 PAPER_DRAFT.md 后重跑。\n"
              "% 需要 \\usepackage{booktabs}\n\n")
    DST.write_text(header + "\n".join(out), encoding="utf-8")
    print(f"已生成 {DST.name}：{n} 张表")


if __name__ == "__main__":
    main()
