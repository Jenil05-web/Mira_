"""
flat_file_adapter.py
=====================
MIRA India — Flat File Adapter

WHO THIS IS FOR:
  Indian clinics and hospitals that store patient data in:
    - Microsoft Excel (.xlsx, .xls)
    - CSV files (exported from Tally, legacy HIS, or custom tools)
    - OpenDocument Spreadsheets (.ods)
    - Google Sheets (public share link)

HOW IT WORKS:
  1. Load every sheet/file into a pandas DataFrame
  2. Load each DataFrame into an in-memory SQLite database
  3. Inherit DBAdapter — same SELECT-only queries, concept map, labs API
  4. The SQL agent does not know or care that the source was a spreadsheet.

SAFETY:
  - In-memory SQLite: data never written to disk by MIRA
  - Same SQL safety gate as DBAdapter: SELECT-only
  - Original file never modified
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from adapters.db import DBAdapter

logger = logging.getLogger(__name__)


def _sanitize_table_name(name: str) -> str:
    """Convert sheet/file names to valid SQLite table names."""
    name = re.sub(r"[^\w]", "_", name.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    if name and name[0].isdigit():
        name = "t_" + name
    return name or "data"


class FlatFileAdapter(DBAdapter):
    """
    Loads flat files (Excel / CSV / Google Sheets) into an in-memory
    SQLite database and exposes the same query interface as DBAdapter.
    """

    def __init__(self):
        self.connection_string = "sqlite:///:memory:"
        self.engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False, "timeout": 15},
        )
        self._schema_cache = None
        self._table_map = None
        self._source_label = "flat_file"
        self._loaded_tables: list[str] = []

    @classmethod
    def from_excel(cls, path: str | Path) -> "FlatFileAdapter":
        adapter = cls()
        adapter._source_label = str(path)
        xl = pd.ExcelFile(path, engine="openpyxl")
        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            adapter._load_dataframe(df, _sanitize_table_name(sheet))
        adapter.discover_schema()
        return adapter

    @classmethod
    def from_csv(cls, path: str | Path, encoding: str = "utf-8") -> "FlatFileAdapter":
        adapter = cls()
        adapter._source_label = str(path)
        try:
            df = pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        adapter._load_dataframe(df, _sanitize_table_name(Path(path).stem))
        adapter.discover_schema()
        return adapter

    @classmethod
    def from_bytes(cls, data: bytes, filename: str) -> "FlatFileAdapter":
        adapter = cls()
        adapter._source_label = filename
        suffix = Path(filename).suffix.lower()

        if suffix in (".xlsx", ".xls", ".xlsm"):
            xl = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
            for sheet in xl.sheet_names:
                adapter._load_dataframe(xl.parse(sheet), _sanitize_table_name(sheet))
        elif suffix == ".ods":
            xl = pd.ExcelFile(io.BytesIO(data), engine="odf")
            for sheet in xl.sheet_names:
                adapter._load_dataframe(xl.parse(sheet), _sanitize_table_name(sheet))
        elif suffix == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(data), encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(data), encoding="latin-1")
            adapter._load_dataframe(df, _sanitize_table_name(Path(filename).stem))
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                "Supported: .xlsx, .xls, .xlsm, .ods, .csv"
            )

        adapter.discover_schema()
        return adapter

    @classmethod
    def from_google_sheets(cls, url: str) -> "FlatFileAdapter":
        adapter = cls()
        adapter._source_label = url

        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9_-]+)", url)
        if not match:
            raise ValueError("Could not extract Google Sheets ID from URL.")
        sheet_id = match.group(1)
        gid_match = re.search(r"gid=(\d+)", url)
        gid = gid_match.group(1) if gid_match else "0"
        csv_url = (
            f"https://docs.google.com/spreadsheets/d/{sheet_id}"
            f"/export?format=csv&gid={gid}"
        )
        import requests
        resp = requests.get(csv_url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        adapter._load_dataframe(df, _sanitize_table_name(f"sheet_{gid}"))
        adapter.discover_schema()
        return adapter

    def _load_dataframe(self, df: pd.DataFrame, table_name: str):
        if df.empty:
            logger.warning(f"Skipping empty sheet/file: {table_name}")
            return
        df = df.copy()
        df.columns = [
            re.sub(r"[^\w]", "_", str(c).strip().lower()).strip("_") or f"col_{i}"
            for i, c in enumerate(df.columns)
        ]
        df = df.dropna(how="all")
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        self._loaded_tables.append(table_name)
        logger.info(f"Loaded {len(df)} rows into table '{table_name}'")

    def add_dataframe(self, df: pd.DataFrame, table_name: str):
        self._load_dataframe(df, _sanitize_table_name(table_name))
        self.discover_schema(force_refresh=True)

    def test_connection(self) -> dict:
        result = super().test_connection()
        row_counts = {}
        for tbl in self._loaded_tables:
            try:
                with self.engine.connect() as conn:
                    r = conn.execute(text(f'SELECT COUNT(*) FROM "{tbl}"')).fetchone()
                    row_counts[tbl] = int(r[0]) if r else 0
            except Exception:
                row_counts[tbl] = 0
        result["row_counts"] = row_counts
        result["source_label"] = self._source_label
        result["db_type"] = f"SQLite (from {Path(self._source_label).suffix or 'file'})"
        return result

    def get_schema_description(self) -> str:
        body = super().get_schema_description()
        suffix = Path(self._source_label).suffix or "file"
        return body.replace(
            "DATABASE SCHEMA (",
            f"DATABASE SCHEMA (In-Memory SQLite from {suffix} — ",
            1,
        )

    @property
    def loaded_tables(self) -> list[str]:
        return list(self._loaded_tables)
