"""
SmartCPR Guardian — MC3G Inspired Counterfactual Generator
"What would need to change for a different outcome?"
Inspired by: Dasgupta et al. — MC3G (arXiv 2025)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
import copy


@dataclass
class Counterfactual:
    """A single counterfactual explanation."""
    feature: str
    current_value: float
    target_value: float
    direction: str          # "increase" or "decrease"
    clinical_action: str    # what a doctor/device should do
    feasibility: str        # "immediate", "short-term", "long-term"
    causal_chain: List[str] # causal steps from action to outcome

    def __str__(self):
        arrow = "↑" if self.direction == "increase" else "↓"
        return (
            f"  {arrow} {self.feature}: "
            f"{self.current_value} → {self.target_value}\n"
            f"    Action: {self.clinical_action}\n"
            f"    Timeline: {self.feasibility}"
        )


# ── Clinical causal knowledge base (ASP-style rules) ──────────────────
# Maps: feature → {direction, action, causal_chain, feasibility}
CLINICAL_INTERVENTIONS = {
    "heart_rate": {
        "increase": {
            "action": "Administer atropine 0.5mg IV or initiate transcutaneous pacing",
            "feasibility": "immediate (2-5 min)",
            "causal_chain": ["Atropine blocks vagal tone",
                             "SA node rate increases",
                             "Cardiac output improves"]
        },
        "decrease": {
            "action": "Administer adenosine 6mg IV or synchronized cardioversion",
            "feasibility": "immediate (1-3 min)",
            "causal_chain": ["Adenosine blocks AV node",
                             "Ventricular rate decreases",
                             "Hemodynamic stability improves"]
        }
    },
    "spo2": {
        "increase": {
            "action": "High-flow oxygen via mask (15L/min) or bag-valve-mask ventilation",
            "feasibility": "immediate (1-2 min)",
            "causal_chain": ["O2 delivery increases",
                             "Alveolar O2 partial pressure rises",
                             "SpO2 normalizes"]
        }
    },
    "systolic_bp": {
        "increase": {
            "action": "IV fluid bolus 250mL NS + consider vasopressors (epinephrine 1mg IV)",
            "feasibility": "immediate (5-10 min)",
            "causal_chain": ["Preload increases",
                             "Stroke volume improves",
                             "Systolic BP rises"]
        }
    },
    "ecg_rr_variability": {
        "decrease": {
            "action": "Defibrillation 200J biphasic (for VFib) — SmartCPR delivers automatically",
            "feasibility": "immediate (SmartCPR auto-triggered)",
            "causal_chain": ["Electrical shock depolarizes myocardium",
                             "Chaotic VFib terminated",
                             "Normal sinus rhythm may resume"]
        }
    }
}

# Normal ranges for counterfactual targets
NORMAL_RANGES = {
    "heart_rate":         (60, 100),
    "spo2":               (95, 100),
    "systolic_bp":        (100, 130),
    "ecg_rr_variability": (0.05, 0.35),
    "ecg_amplitude":      (0.5, 2.5),
    "pulse_pressure":     (40, 60),
}


class MC3GCounterfactualGenerator:
    """
    MC3G-inspired causally-constrained counterfactual generator.

    For each abnormal feature that triggered the FOLD-RM arrest rule,
    generates:
    - What value is needed for a safe outcome
    - What clinical action achieves it
    - The causal chain explaining WHY it works
    - Feasibility timeline

    Reference: Dasgupta, Arias, Salazar, Gupta — MC3G arXiv:2508.17221
    """

    def __init__(self):
        self.causal_constraints = self._build_causal_graph()

    def _build_causal_graph(self) -> Dict:
        """
        Build clinical causal graph.
        Encodes which features causally affect which others.
        (Simplified version of MC3G's causal SCM)
        """
        return {
            "systolic_bp": ["pulse_pressure", "spo2"],
            "heart_rate":  ["pulse_pressure", "spo2"],
            "spo2":        ["heart_rate"],     # hypoxia causes tachycardia
        }

    def generate(self, vitals: Dict[str, float],
                 fired_rules: List[str],
                 label: str) -> List[Counterfactual]:
        """
        Generate counterfactual explanations for abnormal vitals.

        Returns list of counterfactuals ordered by:
        1. Feasibility (immediate first)
        2. Clinical impact (highest first)
        """
        if label == "normal":
            return []

        counterfactuals = []

        for feature, (low, high) in NORMAL_RANGES.items():
            val = vitals.get(feature)
            if val is None:
                continue

            if val < low:
                # Need to increase
                cf = self._make_counterfactual(
                    feature, val, low, "increase", vitals)
                if cf:
                    counterfactuals.append(cf)

            elif val > high:
                # Need to decrease
                cf = self._make_counterfactual(
                    feature, val, high, "decrease", vitals)
                if cf:
                    counterfactuals.append(cf)

        # Sort: immediate first, then short-term
        priority = {"immediate": 0, "SmartCPR": 0, "short-term": 1, "long-term": 2}
        counterfactuals.sort(
            key=lambda c: priority.get(c.feasibility.split(" ")[0], 3))

        return counterfactuals

    def _make_counterfactual(self, feature: str, current: float,
                              target: float, direction: str,
                              all_vitals: Dict) -> "Counterfactual | None":
        """Build a single causally-grounded counterfactual."""
        interv = CLINICAL_INTERVENTIONS.get(feature, {}).get(direction)
        if not interv:
            return None

        # Apply causal constraints: what else changes?
        causal_chain = interv["causal_chain"]
        downstream = self.causal_constraints.get(feature, [])
        if downstream:
            causal_chain = causal_chain + [
                f"Cascade: {', '.join(downstream)} also improve"]

        return Counterfactual(
            feature=feature,
            current_value=round(current, 2),
            target_value=round(target, 2),
            direction=direction,
            clinical_action=interv["action"],
            feasibility=interv["feasibility"],
            causal_chain=causal_chain
        )

    def format_report(self, vitals: Dict, counterfactuals: List[Counterfactual],
                      label: str, confidence: float) -> str:
        """Generate full doctor-facing counterfactual report."""
        lines = ["=" * 60]
        lines.append("SMARTCPR GUARDIAN — COUNTERFACTUAL CLINICAL REPORT")
        lines.append("=" * 60)
        lines.append(f"Outcome: {label.upper().replace('_', ' ')}")
        lines.append(f"Confidence: {confidence:.0%}")
        lines.append("")

        if not counterfactuals:
            lines.append("✅ No interventions needed — patient stable.")
            return "\n".join(lines)

        lines.append("🔄 COUNTERFACTUAL INTERVENTIONS")
        lines.append("(What needs to change for patient stabilization)")
        lines.append("")

        for i, cf in enumerate(counterfactuals, 1):
            arrow = "↑" if cf.direction == "increase" else "↓"
            lines.append(f"[{i}] {arrow} {cf.feature.upper().replace('_',' ')}")
            lines.append(f"    Current: {cf.current_value} → Target: {cf.target_value}")
            lines.append(f"    Action:  {cf.clinical_action}")
            lines.append(f"    Timeline: {cf.feasibility}")
            lines.append(f"    Causal chain:")
            for step in cf.causal_chain:
                lines.append(f"      → {step}")
            lines.append("")

        lines.append("=" * 60)
        lines.append("⚠️  This report is AI-generated. Clinical judgment required.")
        lines.append("Generated by SmartCPR MC3G Engine (Gupta Lab, UTD)")
        return "\n".join(lines)

    def patient_friendly_summary(self, counterfactuals: List[Counterfactual]) -> str:
        """Generate plain-language summary for family/patient app."""
        if not counterfactuals:
            return "Your family member is stable. Emergency services have been alerted."

        immediate = [c for c in counterfactuals
                     if "immediate" in c.feasibility or "SmartCPR" in c.feasibility]
        lines = [
            "⚠️ EMERGENCY IN PROGRESS",
            "",
            "SmartCPR Guardian has detected a cardiac emergency.",
            "The device is providing CPR automatically.",
            "🚑 An ambulance is on its way.",
            "",
            "What is happening:",
        ]
        if any(c.feature == "heart_rate" for c in immediate):
            lines.append("  • Heart rate is dangerously low")
        if any(c.feature == "spo2" for c in immediate):
            lines.append("  • Blood oxygen is critically low")
        if any(c.feature == "systolic_bp" for c in immediate):
            lines.append("  • Blood pressure has dropped")
        lines.append("")
        lines.append("SmartCPR is working to stabilize the situation.")
        lines.append("Please stay calm. Help is coming.")
        return "\n".join(lines)
