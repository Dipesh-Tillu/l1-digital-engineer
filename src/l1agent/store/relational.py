"""SQLAlchemy models + session factory. Schema mirrors SPEC.md §4.5.

`DATABASE_URL` picks the backend (sqlite:/// by default, postgresql://...
for production); the DDL is written to be valid in both.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from l1agent.schemas import (
    IssueRecord,
    IssueStatus,
    ProductProfile,
    ResolutionAuthor,
    ResolutionOutcome,
    ResolutionRecord,
)


class Base(DeclarativeBase):
    pass


class ProductProfileRow(Base):
    __tablename__ = "product_profile"

    product_id = Column(String, primary_key=True)
    display_name = Column(String, nullable=False)
    profile_json = Column(Text, nullable=False)  # full ProductProfile, source of truth
    created_at = Column(DateTime, nullable=False)

    def to_profile(self) -> ProductProfile:
        return ProductProfile.model_validate_json(self.profile_json)


class IssueRecordRow(Base):
    __tablename__ = "issue_record"

    issue_id = Column(String, primary_key=True)
    product_id = Column(String, ForeignKey("product_profile.product_id"), nullable=False, index=True)
    summary = Column(Text, nullable=False)
    body = Column(Text, nullable=False)
    category = Column(String, nullable=True)
    status = Column(String, nullable=False)
    resolution_id = Column(String, nullable=True)
    source_channel = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    # --- demo workflow addendum (l1-support-agent-demo-workflow-spec.md §4) ---
    escalated_from_issue_id = Column(String, nullable=True)
    escalation_queue = Column(String, nullable=True)
    demo_mode = Column(Boolean, nullable=False, default=False, server_default="0")
    attempt_count = Column(Integer, nullable=False, default=0, server_default="0")

    def to_record(self) -> IssueRecord:
        return IssueRecord(
            issue_id=self.issue_id,
            product_id=self.product_id,
            summary=self.summary,
            body=self.body,
            category=self.category,
            status=IssueStatus(self.status),
            resolution_id=self.resolution_id,
            source_channel=self.source_channel,
            created_at=self.created_at,
            updated_at=self.updated_at,
            escalated_from_issue_id=self.escalated_from_issue_id,
            escalation_queue=self.escalation_queue,
            demo_mode=bool(self.demo_mode),
            attempt_count=self.attempt_count or 0,
        )


class ResolutionRecordRow(Base):
    __tablename__ = "resolution_record"

    resolution_id = Column(String, primary_key=True)
    issue_id = Column(String, ForeignKey("issue_record.issue_id"), nullable=False)
    authored_by = Column(String, nullable=False)
    body = Column(Text, nullable=False)
    cited_sources = Column(Text, nullable=False)  # JSON array
    confidence_score = Column(String, nullable=False)
    outcome = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)
    # --- demo workflow addendum (l1-support-agent-demo-workflow-spec.md §4) ---
    attempt_number = Column(Integer, nullable=False, default=1, server_default="1")
    rejection_reason = Column(Text, nullable=True)
    gate_passed = Column(Boolean, nullable=True)
    gate_reasons = Column(Text, nullable=True)  # JSON array

    def to_record(self) -> ResolutionRecord:
        return ResolutionRecord(
            resolution_id=self.resolution_id,
            issue_id=self.issue_id,
            attempt_number=self.attempt_number or 1,
            authored_by=ResolutionAuthor(self.authored_by),
            body=self.body,
            cited_sources=json.loads(self.cited_sources),
            confidence_score=float(self.confidence_score),
            outcome=ResolutionOutcome(self.outcome),
            rejection_reason=self.rejection_reason,
            created_at=self.created_at,
            gate_passed=self.gate_passed,
            gate_reasons=json.loads(self.gate_reasons) if self.gate_reasons else [],
            # kb_hits/issue_hits are intentionally not persisted - see
            # ResolutionRecord's docstring-comment in schemas.py.
        )


class KBDocumentMetaRow(Base):
    __tablename__ = "kb_document_meta"

    doc_id = Column(String, primary_key=True)
    product_id = Column(String, ForeignKey("product_profile.product_id"), nullable=False)
    source_uri = Column(Text, nullable=False)
    section_path = Column(Text, nullable=True)
    last_synced_at = Column(DateTime, nullable=False)


_engine = None
_SessionLocal: sessionmaker | None = None


_NEW_COLUMNS: dict[str, list[tuple[str, str]]] = {
    # table -> [(column, DDL type), ...] added by the demo workflow addendum
    # (l1-support-agent-demo-workflow-spec.md §4). `create_all` only creates
    # missing *tables*, not missing *columns* on a table that already exists
    # on disk (e.g. the real l1agent.db seeded with 268 PowerScale issues) -
    # so existing rows get these lightweight ALTER TABLEs instead of a real
    # migration tool, matching this repo's "no Alembic yet" scope.
    "issue_record": [
        ("escalated_from_issue_id", "VARCHAR"),
        ("escalation_queue", "VARCHAR"),
        ("demo_mode", "BOOLEAN DEFAULT 0"),
        ("attempt_count", "INTEGER DEFAULT 0"),
    ],
    "resolution_record": [
        ("attempt_number", "INTEGER DEFAULT 1"),
        ("rejection_reason", "TEXT"),
        ("gate_passed", "BOOLEAN"),
        ("gate_reasons", "TEXT"),
    ],
}


def _ensure_new_columns(engine) -> None:
    if engine.dialect.name != "sqlite":
        # Other backends (Postgres in production) get a real migration tool
        # before this schema ships there - out of scope for this prototype.
        return
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table, columns in _NEW_COLUMNS.items():
            if table not in existing_tables:
                continue
            existing_cols = {c["name"] for c in inspector.get_columns(table)}
            for name, ddl_type in columns:
                if name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl_type}"))


def get_engine(database_url: str | None = None):
    global _engine
    url = database_url or os.environ.get("DATABASE_URL", "sqlite:///./l1agent.db")
    if _engine is None or str(_engine.url) != url:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
        Base.metadata.create_all(_engine)
        _ensure_new_columns(_engine)
    return _engine


def get_session_factory(database_url: str | None = None) -> sessionmaker:
    global _SessionLocal
    engine = get_engine(database_url)
    if _SessionLocal is None or _SessionLocal.kw.get("bind") is not engine:
        _SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope(database_url: str | None = None) -> Iterator[Session]:
    factory = get_session_factory(database_url)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_cache() -> None:
    """Test helper: force get_engine/get_session_factory to rebuild on next call."""
    global _engine, _SessionLocal
    _engine = None
    _SessionLocal = None


# --------------------------------------------------------------------------
# Repository-style helpers used by ingestion/agent/api code
# --------------------------------------------------------------------------

def upsert_product_profile(session: Session, profile: ProductProfile) -> None:
    row = session.get(ProductProfileRow, profile.product_id)
    payload = profile.model_dump_json()
    if row is None:
        row = ProductProfileRow(
            product_id=profile.product_id,
            display_name=profile.display_name,
            profile_json=payload,
            created_at=profile.created_at,
        )
        session.add(row)
    else:
        row.display_name = profile.display_name
        row.profile_json = payload


def get_product_profile(session: Session, product_id: str) -> ProductProfile | None:
    row = session.get(ProductProfileRow, product_id)
    return row.to_profile() if row else None


def list_product_profiles(session: Session) -> list[ProductProfile]:
    """Every onboarded product - powers the demo chat UI's product
    selector (spec §6), which should only ever offer products that are
    actually indexed and queryable, not a hardcoded guess."""
    rows = session.query(ProductProfileRow).order_by(ProductProfileRow.display_name.asc()).all()
    return [row.to_profile() for row in rows]


def upsert_issue(session: Session, issue: IssueRecord) -> None:
    row = session.get(IssueRecordRow, issue.issue_id)
    if row is None:
        row = IssueRecordRow(issue_id=issue.issue_id)
        session.add(row)
    row.product_id = issue.product_id
    row.summary = issue.summary
    row.body = issue.body
    row.category = issue.category
    row.status = issue.status.value
    row.resolution_id = issue.resolution_id
    row.source_channel = issue.source_channel
    row.created_at = issue.created_at
    row.updated_at = issue.updated_at
    row.escalated_from_issue_id = issue.escalated_from_issue_id
    row.escalation_queue = issue.escalation_queue
    row.demo_mode = issue.demo_mode
    row.attempt_count = issue.attempt_count


def get_issue(session: Session, issue_id: str) -> IssueRecord | None:
    row = session.get(IssueRecordRow, issue_id)
    return row.to_record() if row else None


def insert_resolution(session: Session, resolution: ResolutionRecord) -> None:
    row = ResolutionRecordRow(
        resolution_id=resolution.resolution_id,
        issue_id=resolution.issue_id,
        attempt_number=resolution.attempt_number,
        authored_by=resolution.authored_by.value,
        body=resolution.body,
        cited_sources=json.dumps(resolution.cited_sources),
        confidence_score=str(resolution.confidence_score),
        outcome=resolution.outcome.value,
        rejection_reason=resolution.rejection_reason,
        created_at=resolution.created_at,
        gate_passed=resolution.gate_passed,
        gate_reasons=json.dumps(resolution.gate_reasons),
    )
    session.add(row)


def get_latest_resolution(session: Session, issue_id: str) -> ResolutionRecord | None:
    row = (
        session.query(ResolutionRecordRow)
        .filter(ResolutionRecordRow.issue_id == issue_id)
        .order_by(ResolutionRecordRow.created_at.desc())
        .first()
    )
    return row.to_record() if row else None


def get_resolution(session: Session, resolution_id: str) -> ResolutionRecord | None:
    row = session.get(ResolutionRecordRow, resolution_id)
    return row.to_record() if row else None


def list_resolutions_for_issue(session: Session, issue_id: str) -> list[ResolutionRecord]:
    """Every attempt for one ticket, oldest first - powers the ticket
    board's detail view (spec §6: "every attempt's draft, citations, and
    gate outcome, not just the latest")."""
    rows = (
        session.query(ResolutionRecordRow)
        .filter(ResolutionRecordRow.issue_id == issue_id)
        .order_by(ResolutionRecordRow.attempt_number.asc(), ResolutionRecordRow.created_at.asc())
        .all()
    )
    return [row.to_record() for row in rows]


def update_resolution_outcome(
    session: Session, resolution_id: str, outcome: ResolutionOutcome, rejection_reason: str | None = None,
) -> ResolutionRecord | None:
    row = session.get(ResolutionRecordRow, resolution_id)
    if row is None:
        return None
    row.outcome = outcome.value
    if rejection_reason is not None:
        row.rejection_reason = rejection_reason
    return row.to_record()


def list_issues(
    session: Session, product_id: str | None = None, status: str | None = None,
    source_channel: str | None = None,
) -> list[IssueRecord]:
    """Powers GET /tickets (the demo ticket board, spec §5/§6), most
    recently updated first. `source_channel` lets the board filter out
    historically-ingested issues (hundreds of them, backfilled at
    onboarding time with source_channel="manual_import") so a live demo
    ticket (source_channel="chat_demo") isn't lost in the noise."""
    query = session.query(IssueRecordRow)
    if product_id:
        query = query.filter(IssueRecordRow.product_id == product_id)
    if status:
        query = query.filter(IssueRecordRow.status == status)
    if source_channel:
        # Comma-separated so the board can ask for "chat_demo,escalation" in
        # one call - a live demo ticket plus any L2 hand-off it produced,
        # without the hundreds of onboarding-ingested rows drowning it out.
        channels = [c.strip() for c in source_channel.split(",") if c.strip()]
        query = query.filter(IssueRecordRow.source_channel.in_(channels))
    rows = query.order_by(IssueRecordRow.updated_at.desc()).all()
    return [row.to_record() for row in rows]


def upsert_kb_document_meta(
    session: Session, doc_id: str, product_id: str, source_uri: str,
    section_path: str | None, last_synced_at: datetime,
) -> None:
    row = session.get(KBDocumentMetaRow, doc_id)
    if row is None:
        row = KBDocumentMetaRow(doc_id=doc_id)
        session.add(row)
    row.product_id = product_id
    row.source_uri = source_uri
    row.section_path = section_path
    row.last_synced_at = last_synced_at
