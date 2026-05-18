import importlib.util
import sys
from pathlib import Path


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# 숫자로 시작하는 파일명을 import 가능하게 등록
load_module("02_csv_excel_rag", "02_csv_excel_rag.py")
