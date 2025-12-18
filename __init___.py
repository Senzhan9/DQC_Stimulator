import sys
from pathlib import Path

# 添加父目录到sys.path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))