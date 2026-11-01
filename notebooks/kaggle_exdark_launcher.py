"""ExDark 检测评测的启动器 —— 粘进 Kaggle 单元格即可。

为什么不把整个评测脚本粘到格子里：
    评测脚本有 600 多行，每改一次都要重新复制粘贴，很容易贴错或贴成旧版。
    改成「脚本放在代码包里 + 格子里只调用」之后，以后更新脚本只需在 Kaggle
    数据集页点一次 New Version，格子内容永远不用动。

用法：
    直接运行本格即可。
    想先快速验证（只评测前 1500 张），取消下面那行注释。
"""

import os
import runpy
import sys
from pathlib import Path

# os.environ["EXDARK_LIMIT"] = "1500"   # 快速验证模式，只跑前 1500 张
# os.environ["EXDARK_CLASS_MODE"] = "id0"  # 手动指定 class 编码：id0 / id1 / name

SCRIPT_NAME = "kaggle_exdark_detect_cell.py"
IN = Path("/kaggle/input")


def find_script(base: Path) -> Path | None:
    """在整个 /kaggle/input 下按文件名找脚本，不依赖数据集命名与挂载层级。"""
    if not base.exists():
        return None
    cands = sorted(base.rglob(SCRIPT_NAME), key=lambda p: len(str(p)))
    return cands[0] if cands else None


def main() -> None:
    script = find_script(IN)
    if script is None:
        print(f"[!] 在 {IN} 下没找到 {SCRIPT_NAME}")
        print()
        print("原因通常是代码包还没更新到最新版本。请：")
        print("  1. Kaggle -> Datasets -> indnet-code -> New Version -> 上传最新 indnet-code.tar.gz")
        print("  2. 回到本 Notebook，在右侧 Input 里重新挂载 indnet-code（或重启会话）")
        print()
        print("当前已挂载的内容：")
        try:
            for p in sorted(IN.rglob("*"))[:40]:
                print("   -", p.relative_to(IN))
        except OSError:
            pass
        return

    print("运行脚本:", script)
    print("=" * 78)
    sys.argv = [str(script)]
    runpy.run_path(str(script), run_name="__main__")


main()
