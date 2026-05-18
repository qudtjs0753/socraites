"""
load_csv() 동작 검증 테스트
실행: pytest test_load_csv.py -v
"""

import csv
import os

import pytest

from 02_csv_excel_rag import load_csv


# ── 픽스처 ────────────────────────────────────────────────

def write_csv(path, rows: list[dict], encoding="utf-8-sig"):
    with open(path, "w", encoding=encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ── 테스트 ────────────────────────────────────────────────

def test_기본_행이_Document로_변환된다(tmp_path):
    path = tmp_path / "sample.csv"
    write_csv(path, [
        {"이름": "홍길동", "부서": "개발팀", "직급": "과장"},
        {"이름": "김철수", "부서": "인프라팀", "직급": "대리"},
    ])

    docs = load_csv(str(path))

    assert len(docs) == 2
    assert "홍길동" in docs[0].page_content
    assert "개발팀" in docs[0].page_content


def test_메타데이터에_파일명과_행번호가_들어간다(tmp_path):
    path = tmp_path / "meta_test.csv"
    write_csv(path, [
        {"항목": "A", "값": "1"},
        {"항목": "B", "값": "2"},
    ])

    docs = load_csv(str(path))

    assert docs[0].metadata["source"] == "meta_test.csv"
    assert docs[0].metadata["row"] == 0
    assert docs[1].metadata["row"] == 1


def test_빈_값_컬럼은_content에서_제외된다(tmp_path):
    path = tmp_path / "empty_cols.csv"
    write_csv(path, [{"이름": "홍길동", "메모": "", "부서": "개발팀"}])

    docs = load_csv(str(path))

    assert "메모" not in docs[0].page_content
    assert "이름: 홍길동" in docs[0].page_content
    assert "부서: 개발팀" in docs[0].page_content


def test_모든_컬럼이_비어있는_행은_제외된다(tmp_path):
    path = tmp_path / "all_empty.csv"
    write_csv(path, [
        {"이름": "홍길동", "부서": "개발팀"},
        {"이름": "", "부서": ""},
    ])

    docs = load_csv(str(path))

    assert len(docs) == 1


def test_content_형식이_key_value_줄바꿈이다(tmp_path):
    path = tmp_path / "format.csv"
    write_csv(path, [{"이름": "홍길동", "부서": "개발팀"}])

    docs = load_csv(str(path))

    assert docs[0].page_content == "이름: 홍길동\n부서: 개발팀"


def test_utf8_bom_없는_파일도_읽힌다(tmp_path):
    path = tmp_path / "utf8.csv"
    write_csv(path, [{"이름": "홍길동"}], encoding="utf-8")

    docs = load_csv(str(path), encoding="utf-8")

    assert "홍길동" in docs[0].page_content


def test_euc_kr_파일도_encoding_지정하면_읽힌다(tmp_path):
    path = tmp_path / "euckr.csv"
    write_csv(path, [{"이름": "홍길동", "부서": "개발팀"}], encoding="euc-kr")

    docs = load_csv(str(path), encoding="euc-kr")

    assert "홍길동" in docs[0].page_content


def test_빈_파일은_빈_리스트를_반환한다(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("이름,부서\n", encoding="utf-8-sig")

    docs = load_csv(str(path))

    assert docs == []


def test_파일이_없으면_예외가_발생한다(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(str(tmp_path / "없는파일.csv"))
