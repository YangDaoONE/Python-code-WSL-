#!/usr/bin/env bash
# newproject.sh — WSL/Ubuntu 下的一键新建 Python 项目脚本
# 功能：
#   1) 创建项目目录
#   2) 创建并激活虚拟环境（.venv）
#   3) 升级 pip/setuptools/wheel
#   4) 安装经典四件套（numpy/scipy/matplotlib/pandas）
#      或六件套（再加 requests/tqdm，使用 -6 开关）
#   5) 写入 VS Code 解释器配置（.vscode/settings.json）
#   6) 写入 fonts_helper.py，运行时强制指定中文字体
#   7) 写入 main.py 示例（已调用中文字体工具）
#
# 用法：
#   bash newproject.sh -n 项目名 [-r 根目录] [-6] [--font 字体文件路径]
# 示例：
#   bash newproject.sh -n nmr-exp3
#   bash newproject.sh -n demo -r "$HOME/code/Python" -6 --font "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"

set -euo pipefail

# ===== 参数解析 =====
NAME=""
ROOT="${HOME}/code"
CLASSIC_SIX=false
FONT_PATH=""

usage() {
  cat <<EOF
用法: $(basename "$0") -n 项目名 [-r 根目录] [-6] [--font 字体文件路径]

参数：
  -n, --name        项目名（必填）
  -r, --root        根目录（默认：\$HOME/code）
  -6, --classic-six 安装六件套：在四件套基础上加 requests、tqdm
      --font        指定一个中文字体文件路径（.ttf/.ttc/.otf）
  -h, --help        显示帮助

示例：
  bash $(basename "$0") -n nmr-exp3
  bash $(basename "$0") -n demo -r "\$HOME/code/Python" -6 --font "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--name)
      NAME="${2:-}"; shift 2;;
    -r|--root)
      ROOT="${2:-}"; shift 2;;
    -6|--classic-six)
      CLASSIC_SIX=true; shift;;
    --font)
      FONT_PATH="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "未知参数: $1"; usage; exit 1;;
  esac
done

if [[ -z "${NAME}" ]]; then
  echo "错误：必须指定项目名：-n 项目名"
  usage
  exit 1
fi

# ===== 环境检查 =====
if ! command -v python3 >/dev/null 2>&1; then
  echo "错误：未找到 python3，请先安装：sudo apt update && sudo apt install -y python3"
  exit 1
fi

if ! python3 -c "import venv" >/dev/null 2>&1; then
  echo "错误：当前系统缺少 venv 模块。请先执行："
  echo "  sudo apt update && sudo apt install -y python3-venv"
  exit 1
fi

# ===== 创建项目目录 =====
PROJ="$ROOT/$NAME"
mkdir -p "$PROJ/.vscode" "$PROJ/fonts"
cd "$PROJ"

# ===== 创建虚拟环境 =====
echo "[1/6] 创建虚拟环境 .venv ..."
if ! python3 -m venv .venv; then
  echo "创建虚拟环境失败。如果提示 ensurepip 不可用，请执行：sudo apt install -y python3-venv"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# ===== 升级安装工具 =====
echo "[2/6] 升级 pip/setuptools/wheel ..."
python -m pip install -U pip setuptools wheel

# ===== 写入 requirements.txt =====
echo "[3/6] 写入 requirements.txt ..."
cat > requirements.txt <<'REQ'
numpy
scipy
matplotlib
pandas
REQ

# ===== 安装依赖 =====
echo "[4/6] 安装依赖包 ..."
if [[ "${CLASSIC_SIX}" == "true" ]]; then
  python -m pip install -U numpy scipy matplotlib pandas requests tqdm
else
  python -m pip install -U numpy scipy matplotlib pandas
fi

# ===== VS Code 解释器设置 =====
echo "[5/6] 写入 .vscode/settings.json ..."
cat > .vscode/settings.json <<'JSON'
{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true
}
JSON

# ===== 写入 fonts_helper.py（强制中文字体） =====
echo "[6/6] 写入 fonts_helper.py ..."
cat > fonts_helper.py <<'PY'
from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from pathlib import Path
from typing import Optional

# 常见可用的 SC 字体（优先系统 Noto CJK；也支持把 .ttf/.ttc 放到项目 fonts/ 目录）
_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Light.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "fonts/NotoSansCJK-Regular.ttc",
    "fonts/NotoSansCJKsc-Regular.otf",
    "fonts/SimHei.ttf",
    "fonts/Microsoft YaHei.ttf",
]

def use_cn_font(ttc_path: Optional[str] = None) -> None:
    """
    在当前进程内把 Matplotlib 的默认字体切到指定中文字体（文件路径级别）。
    在任何绘图代码之前调用一次即可。
    """
    path: Optional[Path] = None
    if ttc_path:
        p = Path(ttc_path)
        if not p.exists():
            raise FileNotFoundError(f"指定字体文件不存在: {ttc_path}")
        path = p
    else:
        for c in _CANDIDATES:
            p = Path(c)
            if p.exists():
                path = p
                break

    if path is None:
        raise RuntimeError(
            "未找到可用中文字体文件。请先安装：\n"
            "  sudo apt install -y fonts-noto-cjk fonts-noto-cjk-extra\n"
            "或将 .ttf/.ttc 拷到项目 fonts/ 目录，并传入路径给 use_cn_font(ttc_path)."
        )

    # 注册并设为全局默认
    font_manager.fontManager.addfont(str(path))
    name = FontProperties(fname=str(path)).get_name()
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
PY

# ===== 写入 main.py 示例（可留用/可删除） =====
cat > main.py <<'PY'
from fonts_helper import use_cn_font
use_cn_font()  # 如需点名某个字体文件：use_cn_font("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc")

from matplotlib import pyplot as plt

plt.figure()
plt.title("中文标题：正弦波")
plt.plot([0, 1, 2, 3], [0, 1, 0, -1], label="示例曲线")
plt.xlabel("时间 / s")
plt.ylabel("幅度")
plt.legend()
plt.tight_layout()
plt.savefig("demo.png", dpi=150)
print("OK，已生成 demo.png")
PY

# ===== 如果用户传了 --font，就把路径写入 .env 提示（可选） =====
if [[ -n "${FONT_PATH}" ]]; then
  echo "FONT_FILE=${FONT_PATH}" > .env
fi

# ===== 完成信息 =====
echo
echo "✔ Project ready at: $PROJ"
echo "  Python: $(which python)"
echo "  Pip   : $(which pip)"
echo
echo "下一步："
echo "  1) 在 VS Code 中打开：code \"$PROJ\""
echo "  2) 入口脚本里在任何绘图代码之前调用："
echo "       from fonts_helper import use_cn_font"
echo "       use_cn_font()"
echo "  3) 若仍想指定某个字体文件：use_cn_font(\"/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc\")"
#bash /home/yangdaoone/code/Python/newproject/newproject.sh -n myproj