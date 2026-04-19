"""
SmartCPR Guardian — Emergency Alert & Dispatch System
Handles: GPS-based hospital routing, ambulance dispatch, family alerts.
Simulates HL7/FHIR hospital API integration.
"""

import json
import time
import math
import random
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class Location:
    latitude: float
    longitude: float
    address: str = "Unknown"

    def distance_km(self, other: "Location") -> float:
        """Haversine distance between two GPS coordinates."""
        R = 6371  # Earth radius km
        lat1, lon1 = math.radians(self.latitude), math.radians(self.longitude)
        lat2, lon2 = math.radians(other.latitude), math.radians(other.longitude)
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return R * 2 * math.asin(math.sqrt(a))


@dataclass
class Hospital:
    name: str
    location: Location
    has_cardiac_unit: bool
    phone: str
    fhir_endpoint: str


@dataclass
class EmergencyAlert:
    patient_id: str
    timestamp: str
    event_type: str
    confidence: float
    location: Location
    vitals: Dict
    fold_rm_rules: List[str]
    explanation: str
    cpr_started: bool
    eta_ambulance_min: float


@dataclass
class DispatchResult:
    success: bool
    hospital: Optional[Hospital]
    ambulance_eta_min: float
    alerts_sent: List[str]
    pre_arrival_data_sent: bool
    message: str


# ── Simulated Hospital Database (UTD area, Dallas TX) ──────────────────────
HOSPITALS_DB = [
    Hospital(
        name="UT Southwestern Medical Center",
        location=Location(32.8124, -96.8392, "5323 Harry Hines Blvd, Dallas TX"),
        has_cardiac_unit=True,
        phone="214-645-8300",
        fhir_endpoint="https://fhir.utsouthwestern.edu/api/r4"
    ),
    Hospital(
        name="Baylor University Medical Center",
        location=Location(32.8068, -96.7836, "3500 Gaston Ave, Dallas TX"),
        has_cardiac_unit=True,
        phone="214-820-0111",
        fhir_endpoint="https://fhir.baylordallas.org/api/r4"
    ),
    Hospital(
        name="Medical City Dallas",
        location=Location(32.9068, -96.7718, "7777 Forest Ln, Dallas TX"),
        has_cardiac_unit=True,
        phone="972-566-7000",
        fhir_endpoint="https://fhir.medicalcitydallas.com/api/r4"
    ),
    Hospital(
        name="Parkland Memorial Hospital",
        location=Location(32.8136, -96.8409, "5200 Harry Hines Blvd, Dallas TX"),
        has_cardiac_unit=True,
        phone="214-590-8000",
        fhir_endpoint="https://fhir.parklandhospital.com/api/r4"
    ),
]


class EmergencyDispatcher:
    """
    Core emergency dispatch engine.
    - Finds nearest cardiac-capable hospital
    - Dispatches ambulance with GPS routing
    - Sends pre-arrival patient data via HL7/FHIR
    - Notifies family via SMS simulation
    """

    def __init__(self, patient_db=None):
        self.patient_db = patient_db or {}
        self.dispatch_log: List[Dict] = []

    def dispatch(self, alert: EmergencyAlert,
                 family_contacts: List[str] = None) -> DispatchResult:
        """Full emergency dispatch pipeline."""
        print(f"\n{'='*60}")
        print(f"🚨 EMERGENCY DISPATCH INITIATED — {alert.timestamp}")
        print(f"{'='*60}")

        alerts_sent = []

        # 1. Find nearest cardiac hospital
        hospital = self._find_nearest_hospital(alert.location)
        if not hospital:
            return DispatchResult(False, None, 0, [], False,
                                  "No cardiac hospital found nearby")

        dist_km = alert.location.distance_km(hospital.location)
        eta = self._estimate_eta(dist_km)

        print(f"🏥 Nearest Cardiac Hospital: {hospital.name}")
        print(f"   Distance: {dist_km:.1f} km | ETA: {eta:.1f} min")

        # 2. Send pre-arrival data to hospital ER
        pre_arrival_sent = self._send_pre_arrival_data(alert, hospital)
        if pre_arrival_sent:
            alerts_sent.append(f"Hospital ER: {hospital.name}")
            print(f"✅ Pre-arrival data sent to {hospital.name} ER")
            print(f"   ER is preparing cardiac team...")

        # 3. Dispatch ambulance
        amb_result = self._dispatch_ambulance(alert, hospital, eta)
        if amb_result:
            alerts_sent.append(f"Ambulance dispatched (ETA: {eta:.0f} min)")
            print(f"🚑 Ambulance dispatched | ETA: {eta:.0f} min")
            print(f"   GPS: {alert.location.latitude:.4f}, {alert.location.longitude:.4f}")

        # 4. Notify family
        contacts = family_contacts or ["Family Contact"]
        for contact in contacts:
            msg = self._notify_family(contact, alert, hospital, eta)
            alerts_sent.append(f"Family notified: {contact}")
            print(f"📱 Family alert sent to: {contact}")

        # 5. Log dispatch
        result = DispatchResult(
            success=True,
            hospital=hospital,
            ambulance_eta_min=eta,
            alerts_sent=alerts_sent,
            pre_arrival_data_sent=pre_arrival_sent,
            message=f"Full dispatch complete. CPR active. Ambulance ETA {eta:.0f}min."
        )
        self.dispatch_log.append(asdict(result))
        return result

    def _find_nearest_hospital(self, patient_loc: Location) -> Optional[Hospital]:
        """Find nearest hospital with cardiac unit."""
        cardiac_hospitals = [h for h in HOSPITALS_DB if h.has_cardiac_unit]
        if not cardiac_hospitals:
            return None
        return min(cardiac_hospitals,
                   key=lambda h: patient_loc.distance_km(h.location))

    def _estimate_eta(self, dist_km: float) -> float:
        """Estimate ambulance ETA in minutes."""
        avg_speed_kmh = 60  # urban ambulance speed
        return round((dist_km / avg_speed_kmh) * 60 + 2, 1)  # +2 min prep

    def _send_pre_arrival_data(self, alert: EmergencyAlert,
                                hospital: Hospital) -> bool:
        """
        Send pre-arrival patient data to hospital ER via HL7 FHIR.
        In production: POST to hospital.fhir_endpoint
        """
        fhir_bundle = self._build_fhir_bundle(alert)
        # Simulated API call
        print(f"\n   📋 HL7/FHIR Pre-Arrival Bundle:")
        print(f"   Patient: {alert.patient_id}")
        print(f"   Event: {alert.event_type} (Conf: {alert.confidence:.0%})")
        print(f"   HR: {alert.vitals.get('heart_rate','?')} bpm | "
              f"SpO2: {alert.vitals.get('spo2','?')}% | "
              f"BP: {alert.vitals.get('systolic_bp','?')} mmHg")
        print(f"   CPR Active: {'Yes ✅' if alert.cpr_started else 'No'}")
        return True

    def _dispatch_ambulance(self, alert: EmergencyAlert,
                             hospital: Hospital, eta: float) -> bool:
        """Dispatch nearest ambulance unit."""
        unit_id = f"AMB-{random.randint(100,999)}"
        print(f"\n   🚑 Dispatching unit {unit_id}")
        print(f"   Route: Current location → {hospital.name}")
        print(f"   Priority: CARDIAC ARREST — LIGHTS & SIRENS")
        return True

    def _notify_family(self, contact: str, alert: EmergencyAlert,
                        hospital: Hospital, eta: float) -> str:
        msg = (
            f"⚠️ EMERGENCY ALERT\n"
            f"Your family member needs help. "
            f"SmartCPR Guardian detected a cardiac emergency.\n"
            f"📍 Location: {alert.location.address}\n"
            f"🤖 CPR is being administered automatically.\n"
            f"🚑 Ambulance ETA: {eta:.0f} minutes.\n"
            f"🏥 Being directed to: {hospital.name}\n"
            f"📞 Hospital: {hospital.phone}\n"
            f"⏰ Alert time: {alert.timestamp}"
        )
        return msg

    def _build_fhir_bundle(self, alert: EmergencyAlert) -> Dict:
        """Build a minimal HL7 FHIR R4 bundle for pre-arrival data."""
        return {
            "resourceType": "Bundle",
            "type": "transaction",
            "timestamp": alert.timestamp,
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": alert.patient_id,
                        "active": True
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "status": "final",
                        "code": {"coding": [{"system": "http://loinc.org",
                                             "code": "8867-4",
                                             "display": "Heart rate"}]},
                        "valueQuantity": {
                            "value": alert.vitals.get("heart_rate"),
                            "unit": "bpm"
                        }
                    }
                },
                {
                    "resource": {
                        "resourceType": "Condition",
                        "code": {
                            "coding": [{"system": "http://snomed.info/sct",
                                        "code": "410429000",
                                        "display": "Cardiac arrest (disorder)"}]
                        },
                        "onsetDateTime": alert.timestamp
                    }
                }
            ]
        }
