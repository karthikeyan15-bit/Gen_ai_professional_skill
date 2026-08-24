"""
Task 8: Multi-Agent Collaborative Swarm with Shared Transactional Blackboard Architecture
-----------------------------------------------------------------------------------------
Objective: Architect complex, multi-agent cooperative workflows to solve multi-step
analytical problems, utilizing decentralized consensus and shared memory stores.

Required Tech Stack: Python, SQLite (Transactional DB), Redis/In-Memory Memory Store, Asyncio
Agents:
  - Code Generator Agent
  - System Auditor Agent
  - QA Analyst Agent
Shared Memory: Redis-backed (or in-memory mock) Blackboard with mutual exclusion key locking
"""

import asyncio
import json
import sqlite3
import time
from typing import Any

# =====================================================================
# 1. Transactional Database & Redis Blackboard Store
# =====================================================================

class SQLiteTransactionalDB:
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_schema()

    def _init_schema(self):
        with self.conn:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS task_audits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE,
                    code_snippet TEXT,
                    audit_status TEXT,
                    qa_verdict TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def commit_final_audit(self, task_id: str, code: str, audit_status: str, qa_verdict: str):
        with self.conn:
            self.conn.execute("""
                INSERT OR REPLACE INTO task_audits (task_id, code_snippet, audit_status, qa_verdict)
                VALUES (?, ?, ?, ?)
            """, (task_id, code, audit_status, qa_verdict))

    def fetch_audit(self, task_id: str) -> tuple | None:
        cursor = self.conn.cursor()
        cursor.execute("SELECT task_id, code_snippet, audit_status, qa_verdict FROM task_audits WHERE task_id = ?", (task_id,))
        return cursor.fetchone()


class SharedBlackboardRedisMemory:
    """
    Simulates Redis key-value store with distributed lock support (Redlock pattern concept).
    """
    def __init__(self):
        self.store: dict[str, Any] = {}
        self.locks: dict[str, str] = {} # key -> lock_owner_agent

    async def acquire_lock(self, key: str, agent_id: str, timeout_sec: float = 2.0) -> bool:
        start = time.time()
        while time.time() - start < timeout_sec:
            if key not in self.locks or self.locks[key] == agent_id:
                self.locks[key] = agent_id
                return True
            await asyncio.sleep(0.02)
        return False

    async def release_lock(self, key: str, agent_id: str):
        if self.locks.get(key) == agent_id:
            del self.locks[key]

    async def set_key(self, key: str, value: Any, agent_id: str) -> bool:
        if self.locks.get(key) != agent_id:
            raise PermissionError(f"Agent '{agent_id}' does not hold lock for key '{key}'")
        self.store[key] = value
        return True

    async def get_key(self, key: str) -> Any | None:
        return self.store.get(key)


# =====================================================================
# 2. Specialist Agents
# =====================================================================

class BaseAgent:
    def __init__(self, agent_id: str, blackboard: SharedBlackboardRedisMemory):
        self.agent_id = agent_id
        self.blackboard = blackboard


class CodeGeneratorAgent(BaseAgent):
    async def generate_solution(self, task_id: str, prompt: str):
        key = f"blackboard:{task_id}"
        print(f"[{self.agent_id}] Attempting lock on {key}...")
        if await self.blackboard.acquire_lock(key, self.agent_id):
            try:
                print(f"[{self.agent_id}] Locked {key}. Generating code for prompt: '{prompt}'...")
                await asyncio.sleep(0.05) # Work simulation
                code = f"def solution():\n    # Implementation for {prompt}\n    return True"
                state = {
                    "task_id": task_id,
                    "code": code,
                    "status": "CODE_GENERATED",
                    "audited": False,
                    "qa_approved": False
                }
                await self.blackboard.set_key(key, state, self.agent_id)
                print(f"[{self.agent_id}] Wrote generated code to Blackboard state.")
            finally:
                await self.blackboard.release_lock(key, self.agent_id)
                print(f"[{self.agent_id}] Released lock on {key}.")


class SystemAuditorAgent(BaseAgent):
    async def audit_code(self, task_id: str):
        key = f"blackboard:{task_id}"
        print(f"[{self.agent_id}] Attempting lock on {key}...")
        if await self.blackboard.acquire_lock(key, self.agent_id):
            try:
                state = await self.blackboard.get_key(key)
                if state and state.get("status") == "CODE_GENERATED":
                    print(f"[{self.agent_id}] Auditing code security & performance...")
                    await asyncio.sleep(0.05)
                    state["audited"] = True
                    state["audit_status"] = "PASSED_NO_VULNERABILITIES"
                    state["status"] = "AUDITED"
                    await self.blackboard.set_key(key, state, self.agent_id)
                    print(f"[{self.agent_id}] Updated Blackboard state to AUDITED.")
            finally:
                await self.blackboard.release_lock(key, self.agent_id)
                print(f"[{self.agent_id}] Released lock on {key}.")


class QAAnalystAgent(BaseAgent):
    async def validate_and_commit(self, task_id: str, db: SQLiteTransactionalDB):
        key = f"blackboard:{task_id}"
        print(f"[{self.agent_id}] Attempting lock on {key}...")
        if await self.blackboard.acquire_lock(key, self.agent_id):
            try:
                state = await self.blackboard.get_key(key)
                if state and state.get("status") == "AUDITED":
                    print(f"[{self.agent_id}] Running QA unit test assertions...")
                    await asyncio.sleep(0.05)
                    state["qa_approved"] = True
                    state["status"] = "APPROVED_COMMITTED"
                    
                    # Transactional database commit
                    db.commit_final_audit(
                        task_id=state["task_id"],
                        code=state["code"],
                        audit_status=state["audit_status"],
                        qa_verdict="PASSED"
                    )
                    await self.blackboard.set_key(key, state, self.agent_id)
                    print(f"[{self.agent_id}] Successfully committed final audit state to transactional SQLite DB.")
            finally:
                await self.blackboard.release_lock(key, self.agent_id)
                print(f"[{self.agent_id}] Released lock on {key}.")


# =====================================================================
# 3. Main Execution & Swarm Orchestration
# =====================================================================

async def main():
    print("=" * 70)
    print("Task 8: Multi-Agent Swarm with Shared Blackboard Verification")
    print("=" * 70)

    blackboard = SharedBlackboardRedisMemory()
    db = SQLiteTransactionalDB()

    # Agent Swarm Initialization
    code_agent = CodeGeneratorAgent("Agent_CodeGen", blackboard)
    audit_agent = SystemAuditorAgent("Agent_Auditor", blackboard)
    qa_agent = QAAnalystAgent("Agent_QA", blackboard)

    task_id = "TASK_801"
    prompt = "Vectorized Cosine Similarity Search"

    print(f"\nInitiating Multi-Agent Workflow for Task '{task_id}'...")

    # Phase 1: Code Generation
    await code_agent.generate_solution(task_id, prompt)

    # Phase 2: Security & Architecture Audit
    await audit_agent.audit_code(task_id)

    # Phase 3: QA Verification & Transactional Database Commit
    await qa_agent.validate_and_commit(task_id, db)

    # Verify final commit from DB
    record = db.fetch_audit(task_id)
    print("\n--- Transactional Database Verification ---")
    print(f"Fetched Record for '{task_id}':")
    print(f"  Task ID:      {record[0]}")
    print(f"  Code Snippet:\n{record[1]}")
    print(f"  Audit Status: {record[2]}")
    print(f"  QA Verdict:   {record[3]}")

    assert record is not None and record[3] == "PASSED", "Transactional DB commit failed!"
    print("\nTask 8 completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
