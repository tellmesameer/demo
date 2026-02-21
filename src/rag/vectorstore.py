import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import Column, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from src.config import paths, EMBEDDING_MODEL


class IPCBNSRelationalStore:
    """
    Structured IPC → BNS mapping in SQLite.

    This is the authoritative layer for section mappings.
    """

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine: Engine = create_engine(f"sqlite:///{db_path}")
        self.meta = MetaData()
        self.mapping = Table(
            "ipcbns_mapping",
            self.meta,
            Column("ipc_section", String, primary_key=True),
            Column("bns_section", String),
            Column("notes", String),
        )
        self.meta.create_all(self.engine)

    def upsert_mapping(self, ipc: str, bns: str, notes: str = "") -> None:
        with self.engine.begin() as conn:
            conn.execute(
                self.mapping.insert()
                .values(ipc_section=ipc, bns_section=bns, notes=notes)
                .prefix_with("OR REPLACE")
            )

    def get_by_ipc(self, ipc: str) -> Optional[Dict[str, str]]:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.mapping).where(self.mapping.c.ipc_section == ipc)
            ).fetchone()
            if not row:
                return None
            return dict(row._mapping)

    def get_by_bns(self, bns: str) -> Optional[Dict[str, str]]:
        with self.engine.begin() as conn:
            row = conn.execute(
                select(self.mapping).where(self.mapping.c.bns_section == bns)
            ).fetchone()
            if not row:
                return None
            return dict(row._mapping)


class IPCBNSVectorStore:
    """
    Semantic store over processed PDF chunks, backed by Chroma.
    """

    def __init__(self, persist_dir: Optional[str] = None):
        if persist_dir is None:
            persist_dir = paths.CHROMA_DIR
        self.persist_dir = persist_dir
        Path(self.persist_dir).parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
        )
        self.store: Optional[Chroma] = None

    def build_from_json(self, json_path: str) -> None:
        p = Path(json_path)
        if not p.exists():
            raise FileNotFoundError(
                f"Chunks JSON not found at {p}. "
                "Please run 'python src/rag/pdf_processor.py' to generate it."
            )
        data = json.loads(p.read_text(encoding="utf-8"))

        texts = [c["text"] for c in data["chunks"]]
        metas = [c["metadata"] for c in data["chunks"]]

        self.store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metas,
            persist_directory=self.persist_dir,
        )

    def load_or_build(self) -> None:
        if Path(self.persist_dir).exists():
            self.store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        else:
            self.build_from_json(paths.PROCESSED_CHUNKS)

    def query(
        self, query: str, k: int = 5
    ) -> List[Tuple[str, Dict[str, Any], float]]:
        if self.store is None:
            self.load_or_build()
        docs = self.store.similarity_search_with_score(query, k=k)
        return [(d.page_content, d.metadata, float(score)) for d, score in docs]
