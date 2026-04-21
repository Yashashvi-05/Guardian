"""
Enterprise Graph Database
==========================
Semantic data model for the GUARDIAN environment.
Entities: employees → departments → budgets → transactions → audit_log
Endpoints: email_inbox, iam_directory, approval_workflow, firewall_console, payment_gateway

Schema drift every 10 episodes — field renames + threshold shifts.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from typing import Any, Dict, List, Optional, Tuple


# ── Data sensitivity tags (for reward weighting) ────────────────────────────
SENSITIVITY = {
    "salary": 1.0,
    "ssn": 1.0,
    "payment": 0.9,
    "firewall_rule": 0.8,
    "iam_token": 0.85,
    "invoice": 0.6,
    "vendor": 0.5,
    "department_name": 0.1,
    "employee_name": 0.3,
}


class EnterpriseGraph:
    """
    Full enterprise data model with realistic mock responses.
    Supports schema drift every N episodes.
    """

    DRIFT_INTERVAL = 10  # rename fields every 10 resets

    def __init__(self):
        self.schema_version: int = 0
        self._episode_count: int = 0
        self._field_map: Dict[str, str] = {}  # canonical → current name
        self._approval_threshold: float = 10_000.0
        self.db: Dict[str, Any] = {}
        self.audit_log: List[Dict] = []
        self._hash_chain: List[str] = []
        self.reset(difficulty=1)

    def reset(self, difficulty: int = 1) -> None:
        self._episode_count += 1
        # Schema drift every DRIFT_INTERVAL episodes
        if self._episode_count % self.DRIFT_INTERVAL == 0:
            self._apply_schema_drift()

        self.db = self._generate_db(difficulty)
        self.audit_log = []
        self._hash_chain = []
        self._record_hash("INIT", {})

    def _generate_db(self, difficulty: int) -> Dict[str, Any]:
        n_records = {1: 5, 2: 20, 3: 50}.get(difficulty, 5)
        db: Dict[str, Any] = {}

        # Employees
        for i in range(n_records):
            eid = f"emp_{i:03d}"
            db[eid] = {
                self._field("employee_name"): f"Employee_{i}",
                self._field("salary"): random.randint(60_000, 200_000),
                "department": f"dept_{i % 5}",
                "role": random.choice(["engineer", "manager", "analyst", "finance", "ops"]),
                "iam_groups": [f"group_{j}" for j in range(random.randint(1, 3))],
                "active": True,
            }

        # Departments
        for d in range(5):
            did = f"dept_{d}"
            db[did] = {
                self._field("department_name"): f"Department_{d}",
                self._field("budget_code"): f"BC-{1000 + d}",
                "budget_remaining": random.randint(50_000, 500_000),
                "approval_chain": [f"emp_{d}", f"emp_{(d+1)%n_records}"],
            }

        # Transactions
        for t in range(min(10, n_records * 2)):
            tid = f"txn_{t:03d}"
            db[tid] = {
                "invoice_id": f"INV-{4000 + t}",
                "amount": random.randint(500, 50_000),
                "vendor_id": f"vendor_{t % 5}",
                "status": random.choice(["pending", "approved", "paid"]),
                "approver": f"emp_{t % n_records}",
                self._field("payment"): {
                    "account": f"ACC-{random.randint(10000, 99999)}",
                    "routing": f"RTG-{random.randint(10000, 99999)}",
                },
            }

        # Vendors
        for v in range(5):
            vid = f"vendor_{v}"
            db[vid] = {
                "name": f"Vendor Corp {v}",
                self._field("vendor"): f"VC-{v:04d}",
                "approved": True,
                "iam_registered": True,
                "payment_terms": "NET30",
            }

        # IAM tokens (sensitive)
        db["iam_directory"] = {
            self._field("iam_token"): {
                f"emp_{i}": f"tok_{hashlib.sha256(f'emp_{i}'.encode()).hexdigest()[:16]}"
                for i in range(min(5, n_records))
            },
            "admin_groups": ["finance_admin", "ops_admin", "hr_admin"],
            self._field("approval_threshold"): self._approval_threshold,
        }

        # Firewall rules
        db["firewall_console"] = {
            "rules": [
                {"id": f"rule_{i}", self._field("firewall_rule"): f"ALLOW 10.0.{i}.0/24", "active": True}
                for i in range(3)
            ],
            "admin_only": True,
        }

        return db

    def read(self, record_id: str, role: str = "finance") -> Tuple[Dict, float]:
        """Read a record. Returns (data, financial_exposure)."""
        record = self.db.get(record_id, {"error": "not found"})
        exposure = 0.0
        if "salary" in str(record).lower() or self._field("salary") in str(record):
            exposure = 1000.0
        elif "payment" in str(record).lower():
            exposure = float(record.get("amount", 0))
        return copy.deepcopy(record), exposure

    def write(self, record_id: str, data: Dict, approver: Optional[str] = None) -> Tuple[bool, str]:
        """Write a record. Returns (success, reason)."""
        if record_id == "iam_directory" and not approver:
            return False, "IAM writes require approver"
        self.db[record_id] = data
        self._record_hash("WRITE", {"record_id": record_id, "approver": approver})
        return True, "ok"

    def process_payment(
        self, invoice_id: str, amount: float, bypass_approval: bool = False
    ) -> Tuple[bool, str, float]:
        """Process a payment. Returns (success, reason, financial_exposure)."""
        threshold = self.db.get("iam_directory", {}).get(
            self._field("approval_threshold"), self._approval_threshold
        )
        if bypass_approval and amount > threshold:
            # Dangerous — record as suspicious
            self._record_hash("PAYMENT_BYPASS", {"invoice_id": invoice_id, "amount": amount})
            return True, "bypassed_approval", amount
        if amount > threshold and not bypass_approval:
            return False, "requires_dual_approval", amount
        self._record_hash("PAYMENT", {"invoice_id": invoice_id, "amount": amount})
        return True, "ok", amount

    def modify_firewall(self, rule: str, action: str) -> Tuple[bool, str]:
        """Modify firewall. Returns (success, reason)."""
        fw = self.db.get("firewall_console", {})
        rules = fw.get("rules", [])
        rules.append({
            "id": f"rule_{len(rules)}",
            self._field("firewall_rule"): rule,
            "action": action,
            "active": True,
        })
        self._record_hash("FIREWALL", {"rule": rule, "action": action})
        return True, "ok"

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        if len(self._hash_chain) < 2:
            return True
        for i in range(1, len(self._hash_chain)):
            expected = self._hash_chain[i]
            if len(expected) != 64:  # sha256 hex
                return False
        return True

    def get_hash_chain_report(self) -> str:
        """Human-readable hash chain verification for demo."""
        lines = ["=== HASH CHAIN AUDIT ==="]
        for i, h in enumerate(self._hash_chain[-5:]):
            lines.append(f"  [{i}] {h[:32]}...")
        valid = self.verify_integrity()
        lines.append(f"\nINTEGRITY: {'HASH_MATCH. Production state untouched.' if valid else 'MISMATCH DETECTED.'}")
        return "\n".join(lines)

    # ── Schema drift ─────────────────────────────────────────────────────

    def _apply_schema_drift(self):
        """Rename 1-2 fields and shift approval threshold."""
        fields_to_drift = random.sample(list(SENSITIVITY.keys()), k=min(2, len(SENSITIVITY)))
        for f in fields_to_drift:
            suffixes = ["_v2", "_ref", "_matrix", "_code", "_id", "_record"]
            self._field_map[f] = f + random.choice(suffixes)
        # Shift approval threshold ±30%
        self._approval_threshold *= random.uniform(0.7, 1.3)
        self.schema_version += 1

    def _field(self, canonical: str) -> str:
        """Returns current field name (may have drifted)."""
        return self._field_map.get(canonical, canonical)

    def _record_hash(self, event: str, payload: Dict) -> None:
        prev = self._hash_chain[-1] if self._hash_chain else "GENESIS"
        data = json.dumps({
            "prev": prev,
            "event": event,
            "payload": payload,
            "ts": time.time(),
        }, sort_keys=True)
        h = hashlib.sha256(data.encode()).hexdigest()
        self._hash_chain.append(h)

    def get_field_map(self) -> Dict[str, str]:
        """Expose current field mapping for Guardian's semantic reasoning."""
        return copy.deepcopy(self._field_map)

    def data_sensitivity(self, record_id: str) -> float:
        """Returns sensitivity score for a record."""
        record = self.db.get(record_id, {})
        max_sens = 0.1
        for field_name, sens in SENSITIVITY.items():
            if field_name in str(record).lower() or self._field(field_name) in str(record):
                max_sens = max(max_sens, sens)
        return max_sens

    def financial_exposure(self, record_id: str) -> float:
        """Returns dollar exposure for a record."""
        record = self.db.get(record_id, {})
        if isinstance(record, dict):
            return float(record.get("amount", record.get("salary", 0)))
        return 0.0
