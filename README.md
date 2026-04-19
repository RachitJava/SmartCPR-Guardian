# 🫀 SmartCPR Guardian
## AI-Driven Wearable Cardiac Arrest Detection, Emergency Dispatch & Automated CPR System

![Simulation CI](https://github.com/RachitJava/SmartCPR-Guardian/actions/workflows/simulation.yml/badge.svg)


**Research Proposal for RA Position**
**University of Texas at Dallas — Department of Computer Science**
**Advisor (Target): Prof. Gopal Gupta | Co-Advisor: Prof. Lakshman Tamil**

---

## 📋 Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Proposed Solution](#3-proposed-solution)
4. [Full System Architecture](#4-full-system-architecture)
5. [Hardware Device Design](#5-hardware-device-design)
6. [AI & Software Stack](#6-ai--software-stack)
7. [Integration with Prof. Gupta's Lab](#7-integration-with-prof-guptas-lab)
8. [Integration with Prof. Tamil's Lab](#8-integration-with-prof-tamils-lab)
9. [Emergency Response Flow](#9-emergency-response-flow)
10. [Development Phases](#10-development-phases)
11. [Dataset Plan](#11-dataset-plan)
12. [Publication Roadmap](#12-publication-roadmap)
[🚀 Getting Started Guide](#-getting-started)


---

## 1. Executive Summary

Sudden cardiac arrest (SCA) kills **350,000+ Americans per year**. Survival rate drops **10% every minute** without CPR. Average ambulance response time is **8–12 minutes** — far too late for most victims.

**SmartCPR Guardian** is a wearable device that:
- 🔍 **Detects** cardiac arrest in real-time using an explainable AI model (FOLD-RM)
- 🚨 **Alerts** the nearest hospital and ambulance automatically with patient vitals and GPS
- 🤖 **Initiates CPR** mechanically within seconds of detection

This project directly extends Prof. Gupta's NSF-funded **Automated HF Diagnosis** project (Award #1916206) and Prof. Tamil's **wearable IoT cardiac monitoring** platform — bridging both labs into a single life-critical AI+hardware system.

---

## 2. Problem Statement

```mermaid
graph TD
    A["💔 Cardiac Arrest Occurs"] --> B["⏱️ Average Response Time: 8-12 min"]
    B --> C["📉 Survival drops 10% per minute"]
    C --> D["😢 Only 10% survive out-of-hospital SCA"]

    E["❌ Current Limitations"] --> F["Wearables: detect but cannot act"]
    E --> G["CPR machines: hospital-only, bulky"]
    E --> H["AI models: black-box, no explanation"]
    E --> I["No system combines all three"]

    style A fill:#c0392b,color:#fff
    style D fill:#c0392b,color:#fff
    style E fill:#e67e22,color:#fff
```

| Statistic | Value |
|---|---|
| Annual SCA deaths (USA) | **350,000+** |
| Survival without CPR in 10 min | **< 5%** |
| Survival WITH immediate CPR | **up to 45%** |
| Average ambulance response | **8–12 minutes** |
| Gap between arrest & CPR | **The critical window we close** |

---

## 3. Proposed Solution

```mermaid
graph LR
    subgraph Device ["🩺 SmartCPR Guardian (Wearable)"]
        S1["ECG Sensor"] --> AI
        S2["SpO2 Sensor"] --> AI
        S3["Accelerometer"] --> AI
        S4["BP Sensor"] --> AI
        AI["FOLD-RM\nAI Engine"] --> D1["🚨 Alert Module"]
        AI --> D2["🤖 CPR Mechanism"]
    end

    D1 --> H["🏥 Nearest Hospital\n(Pre-arrival data)"]
    D1 --> AM["🚑 Ambulance Dispatch\n(GPS location)"]
    D1 --> FAM["📱 Family Alert\n(SMS/App)"]
    D2 --> CPR["⚡ Automated Chest\nCompressions Begin"]

    style Device fill:#1a237e,color:#fff
    style AI fill:#4caf50,color:#fff
    style CPR fill:#f44336,color:#fff
```

---

## 4. Full System Architecture

```mermaid
graph TB
    subgraph HW ["🔧 Layer 1: Hardware Sensors (Tamil's Lab)"]
        ECG["ECG\n(3-lead)"]
        SPO["SpO2\nPulse Ox"]
        ACC["3-axis\nAccelerometer"]
        BP["Blood\nPressure"]
        GPS["GPS\nModule"]
        CELL["4G/5G\nModule"]
    end

    subgraph PROC ["⚙️ Layer 2: Edge AI Processor"]
        PP["Signal\nPreprocessing"]
        FOLD["FOLD-RM\nClassifier"]
        CONF["Confidence\nScoring"]
        THRESH["Threshold\nDecision Engine"]
    end

    subgraph CLOUD ["☁️ Layer 3: Cloud AI (Gupta's Lab)"]
        SCASP["s(CASP)\nReasoning Engine"]
        MC3G["MC3G\nCounterfactual\nGenerator"]
        LLM["LLM\n(GPT/LLaMA)\nGrounded"]
        DB["Patient\nHistory DB"]
    end

    subgraph RESPONSE ["🚨 Layer 4: Emergency Response"]
        AMB["Ambulance\nDispatch API"]
        HOSP["Hospital\nER System\n(HL7/FHIR)"]
        FAM["Family\nAlert App"]
        DOC["Doctor\nDashboard"]
    end

    subgraph ACTION ["🤖 Layer 5: Physical CPR"]
        VEST["Pneumatic\nCompression Vest"]
        PUMP["Air Pump\n(30 BPM)"]
        DEFIB["Defibrillation\nModule"]
    end

    ECG & SPO & ACC & BP --> PP
    GPS & CELL --> RESPONSE
    PP --> FOLD --> CONF --> THRESH
    THRESH --> |"Arrest Confirmed"| SCASP
    THRESH --> |"Arrest Confirmed"| ACTION
    THRESH --> |"Alert Triggered"| RESPONSE
    SCASP --> MC3G --> LLM
    LLM --> HOSP & DOC
    DB --> SCASP

    style HW fill:#1565c0,color:#fff
    style PROC fill:#2e7d32,color:#fff
    style CLOUD fill:#4a148c,color:#fff
    style RESPONSE fill:#e65100,color:#fff
    style ACTION fill:#b71c1c,color:#fff
```

---

## 5. Hardware Device Design

### 5.1 Physical Design

```mermaid
graph LR
    subgraph Vest ["👕 SmartCPR Vest (Wearable)"]
        subgraph Front ["Front Panel"]
            ECG_P["ECG Patches\n(3 leads)"]
            CPR_P["Pneumatic\nBladder\n(sternum area)"]
            DEFIB_P["Defibrillation\nPads"]
        end
        subgraph Back ["Back Panel / Housing"]
            MCU["MCU\n(ARM Cortex-M7)"]
            BAT["Battery\n(Li-Po 5000mAh)"]
            PUMP_H["Air Pump"]
            MODEM["4G Modem"]
            GPS_H["GPS Chip"]
        end
        subgraph Side ["Side Sensors"]
            SPO_H["SpO2\nFinger Clip"]
            ACC_H["Accelerometer"]
        end
    end

    style Vest fill:#1a237e,color:#fff
    style Front fill:#283593,color:#fff
    style Back fill:#283593,color:#fff
    style Side fill:#283593,color:#fff
```

### 5.2 Hardware Components List

| Component | Spec | Purpose |
|---|---|---|
| **ECG Module** | ADS1292R (3-lead) | Heart rhythm detection |
| **SpO2 Sensor** | MAX30102 | Blood oxygen level |
| **Accelerometer** | MPU-6050 (6-DOF) | Motion/fall detection |
| **BP Sensor** | MEMS piezo | Blood pressure |
| **MCU** | STM32H7 (ARM Cortex-M7) | Edge AI processing |
| **4G Module** | SIM7600 | Emergency cellular alert |
| **GPS** | u-blox NEO-M8N | Location for dispatch |
| **Battery** | Li-Po 5000mAh | ~12hr operation |
| **Air Pump** | Micro DC pump 12V | CPR compression |
| **Pneumatic Bladder** | Silicone inflatable | Chest compressions |
| **Defibrillator** | 200J capacitor discharge | Shock for VFib |

### 5.3 CPR Mechanism Detail

```mermaid
graph TD
    DETECT["Cardiac Arrest Confirmed\n(FOLD-RM Score > 99%)"] --> TIMER["⏱️ 5-second\nConfirmation Wait"]
    TIMER --> CHECK["❓ No Response\nfrom Patient?"]
    CHECK --> |"No response"| INFLATE["💨 Pump Inflates\nBladder Rapidly"]
    CHECK --> |"Patient responds"| CANCEL["✅ Cancel —\nFalse Alarm"]
    INFLATE --> COMPRESS["⬇️ Compress Chest\n5-6cm depth"]
    COMPRESS --> RELEASE["⬆️ Release\n(Bladder deflates)"]
    RELEASE --> RATE["🔄 Repeat at\n30 compressions/min\n(AHA Standard)"]
    RATE --> COMPRESS
    RATE --> DEFIB_CHECK["⚡ Every 2 min:\nCheck for Shockable\nRhythm"]
    DEFIB_CHECK --> |"VFib detected"| SHOCK["⚡ Deliver Shock\n200J"]
    DEFIB_CHECK --> |"No VFib"| RATE

    style DETECT fill:#b71c1c,color:#fff
    style COMPRESS fill:#c62828,color:#fff
    style SHOCK fill:#f57f17,color:#fff
    style CANCEL fill:#2e7d32,color:#fff
```

---

## 6. AI & Software Stack

### 6.1 On-Device AI (Edge) — FOLD-RM

```mermaid
graph LR
    subgraph INPUT ["📥 Input Features"]
        F1["Heart Rate"]
        F2["ECG Rhythm Pattern"]
        F3["SpO2 Level"]
        F4["Motion/Activity"]
        F5["Respiration Rate"]
        F6["Blood Pressure"]
    end

    subgraph FOLDRM ["🧠 FOLD-RM (On Device)"]
        RULE1["rule1: cardiac_arrest(X) :-\n  heart_rate(X,N), N < 20."]
        RULE2["rule2: cardiac_arrest(X) :-\n  ecg(X,'VFib'),\n  spo2(X,N), N < 85."]
        RULE3["rule3: :- motion(X,'normal'),\n  not cardiac_arrest(X)."]
        OUT["Classification:\nARREST / NO ARREST\n+ Confidence 0-100%"]
    end

    INPUT --> FOLDRM
    FOLDRM --> OUT

    style FOLDRM fill:#2e7d32,color:#fff
```

### 6.2 Cloud AI — s(CASP) + MC3G + LLM

```mermaid
graph TD
    EDGE["📲 Device sends:\nVitals + FOLD-RM rules\n+ Patient ID"] --> CLOUD

    subgraph CLOUD ["☁️ Cloud (Gupta's Lab Stack)"]
        HIST["Patient History\nEHR Database"]
        SCASP2["s(CASP) Engine\nReasoning over rules"]
        MC3G2["MC3G\nCounterfactual Generator"]
        LLM2["LLM (GPT-4)\nGrounded in ASP rules"]
    end

    HIST --> SCASP2
    EDGE --> SCASP2
    SCASP2 --> MC3G2
    MC3G2 --> LLM2

    LLM2 --> DOC_OUT["👨‍⚕️ Doctor Report:\n'Cardiac arrest at 16:14.\nHistory: CHF + Diabetes.\nCounterfactual: if sodium\nhad been managed, risk\nwas 40% lower.\nMeds on board: Metoprolol.'"]

    LLM2 --> PAT_OUT["📱 Family Alert:\n'Emergency: John had a\ncardiac event. SmartCPR\nis providing CPR.\nAmbulance ETA: 6 min.'"]

    style CLOUD fill:#4a148c,color:#fff
    style DOC_OUT fill:#1565c0,color:#fff
    style PAT_OUT fill:#e65100,color:#fff
```

---

## 7. Integration with Prof. Gupta's Lab

```mermaid
graph LR
    subgraph GUPTA ["🎓 Gupta Lab (Existing)"]
        FOLDRM_G["FOLD-RM\n(Binary Classifier)"]
        MC3G_G["MC3G\n(Counterfactual)"]
        SCASP_G["s(CASP)\n(Reasoning Engine)"]
        LLM_G["LLM + ASP\n(AutoCompanion)"]
        CHF_G["CHF Physician\nAdvisory System\n(2016)"]
    end

    subgraph OUR ["🚀 Our Contribution"]
        EDGE_OUR["Edge FOLD-RM\n(on-device, real-time)"]
        TEMPORAL["Temporal Reasoning\n(Event Calculus)"]
        DRUG["Drug Interaction\nSafety Rules"]
        UPDATED["Updated 2022 AHA\nGuidelines in ASP"]
        REALTIME["Real-time Stream\nProcessing"]
    end

    FOLDRM_G --> |"port to embedded C"| EDGE_OUR
    SCASP_G --> |"add temporal rules"| TEMPORAL
    CHF_G --> |"update guidelines"| UPDATED
    MC3G_G --> |"connect to device output"| REALTIME
    LLM_G --> |"dual output: doctor + patient"| OUR

    style GUPTA fill:#4a148c,color:#fff
    style OUR fill:#1b5e20,color:#fff
```

### What We Reuse vs. What We Build New

| Gupta's Existing Tool | We Reuse | We Extend |
|---|---|---|
| FOLD-RM (Python) | Core algorithm | Port to C for edge MCU |
| s(CASP) | Reasoning engine | Add temporal Event Calculus rules |
| MC3G | Counterfactual generation | Apply to cardiac arrest context |
| CHF Guidelines (ASP) | Existing rules | Update to 2022 AHA + add arrest rules |
| LLM + ASP pipeline | Grounding mechanism | Dual output (doctor + patient language) |

---

## 8. Integration with Prof. Tamil's Lab

```mermaid
graph TD
    subgraph TAMIL ["🔬 Tamil Lab (Existing)"]
        ECG_T["Wearable ECG\nSensor Platform"]
        IOT_T["IoT Data\nTransmission Framework"]
        SIG_T["Signal Processing\nAlgorithms (Beat Detection)"]
        TELE_T["Telemedicine\nPlatform (CHF)"]
    end

    subgraph OUR2 ["🚀 Our Hardware Layer"]
        VEST_O["Pneumatic CPR Vest\n(New Design)"]
        DEFIB_O["Defibrillation Module\n(New)"]
        INTEG_O["Hardware-AI\nIntegration Layer"]
        PROTO_O["MCU Integration\n(STM32H7)"]
    end

    ECG_T --> |"ECG signal pipeline"| PROTO_O
    IOT_T --> |"data transmission protocol"| INTEG_O
    SIG_T --> |"beat detection reused"| PROTO_O
    TELE_T --> |"hospital interface API"| INTEG_O

    VEST_O --> INTEG_O
    DEFIB_O --> INTEG_O
    PROTO_O --> INTEG_O

    style TAMIL fill:#1565c0,color:#fff
    style OUR2 fill:#b71c1c,color:#fff
```

### Tamil Lab API Integration Points

```mermaid
sequenceDiagram
    participant Vest as SmartCPR Vest
    participant Tamil as Tamil IoT Platform
    participant Gupta as Gupta Cloud AI
    participant ER as Hospital ER System

    Vest->>Tamil: Raw ECG + SpO2 stream (continuous)
    Tamil->>Tamil: Signal processing / beat detection
    Tamil->>Gupta: Processed vitals + features
    Gupta->>Gupta: FOLD-RM classification
    Gupta->>Vest: "Arrest Detected — Confidence 99%"
    Vest->>Vest: ⚡ Trigger CPR mechanism
    Vest->>Tamil: Alert with GPS + vitals snapshot
    Tamil->>ER: Pre-arrival patient data (HL7/FHIR)
    Tamil->>ER: Ambulance dispatch request
    ER->>ER: Prepare cardiac team
    Note over Vest,ER: CPR already running before ambulance arrives
```

---

## 9. Emergency Response Flow

```mermaid
flowchart TD
    A["👤 Patient wearing SmartCPR"] --> B["📡 Continuous Monitoring\n(ECG, SpO2, BP, Motion)"]
    B --> C{"🧠 FOLD-RM\nDetects Anomaly?"}
    C --> |"No"| B
    C --> |"Yes — Score > 85%"| D["⚠️ Pre-Alert State\n5-sec Confirmation Window"]
    D --> E{"👆 Patient\nResponds?"}
    E --> |"Yes — presses button"| F["✅ False Alarm\nLogged & Monitored"]
    F --> B
    E --> |"No response in 5 sec"| G["🔴 CARDIAC ARREST CONFIRMED"]

    G --> H["🤖 CPR STARTS\nAutomatically"]
    G --> I["📍 GPS Location Captured"]
    G --> J["☁️ Cloud AI Activated"]

    I --> K["🚑 Ambulance Dispatch\nNearest unit alerted"]
    I --> L["🏥 Hospital ER Notified\nPatient data sent ahead"]
    I --> M["📱 Family Notified\nSMS + App push"]

    J --> N["s(CASP) + MC3G\nGenerates context report"]
    N --> L

    H --> O["⏱️ CPR Continues\n30 compressions/min"]
    O --> P{"⚡ Check VFib\nevery 2 min"}
    P --> |"VFib present"| Q["⚡ Deliver Defibrillation\n200 Joules"]
    P --> |"No VFib"| O
    Q --> O

    K --> R["🚑 Ambulance Arrives\n(Avg 6-8 min)"]
    R --> S["👨‍⚕️ Paramedics take over\nCPR data log handed off"]

    style G fill:#b71c1c,color:#fff
    style H fill:#c62828,color:#fff
    style Q fill:#f57f17,color:#fff
    style S fill:#2e7d32,color:#fff
```

---

## 10. Development Phases

```mermaid
gantt
    title SmartCPR Guardian — 12-Month Development Plan
    dateFormat  MM
    axisFormat  Month %m

    section Phase 1: AI Detection
    Literature Review & Dataset Prep     :01, 1M
    Train FOLD-RM on ECG Arrest Data     :02, 2M
    Optimize for Edge Deployment         :03, 1M
    Validate (MIT-BIH + MIMIC-III)       :04, 1M

    section Phase 2: Alert System
    GPS + Cellular Module Integration    :04, 1M
    Hospital API (HL7/FHIR) Setup        :05, 1M
    Family Alert App (MVP)               :05, 1M
    End-to-End Alert Testing             :06, 1M

    section Phase 3: Cloud AI
    s(CASP) CHF Rules Update (2022 AHA)  :05, 1M
    MC3G Integration for Cardiac Arrest  :06, 2M
    LLM Dual-Output (Doctor + Patient)   :07, 1M
    Cloud-Device Pipeline Testing        :08, 1M

    section Phase 4: CPR Hardware
    Pneumatic Vest Prototype Design      :07, 2M
    Defibrillation Module                :08, 1M
    MCU Integration (STM32H7)            :09, 1M
    Safety Testing (False Positive Rate) :10, 1M

    section Phase 5: Integration & Paper
    Full System Integration Testing      :10, 1M
    IRB / Ethics Approval Process        :10, 2M
    Paper Writing (AAAI / IEEE EMBC)     :11, 2M
```

### Phase Details

| Phase | Months | Deliverables | Tools |
|---|---|---|---|
| **1: AI Detection** | 1–4 | Trained FOLD-RM on ECG, >99% sensitivity | Python, FOLD-RM lib, MIT-BIH dataset |
| **2: Alert System** | 4–6 | Working GPS dispatch + hospital API | SIM7600, HL7/FHIR, REST APIs |
| **3: Cloud AI** | 5–8 | Updated CHF rules + MC3G + LLM pipeline | s(CASP), Python, GPT-4 API |
| **4: CPR Hardware** | 7–10 | Working pneumatic vest prototype | STM32H7, Arduino, pneumatic components |
| **5: Integration** | 10–12 | Full system demo + paper submission | All of above |

---

## 11. Dataset Plan

| Dataset | Content | Use |
|---|---|---|
| **MIT-BIH Arrhythmia DB** | 48 ECG recordings, annotated | Train FOLD-RM for arrest detection |
| **MIMIC-III / MIMIC-IV** | ICU patient EHR, vitals, labs | CHF patient history + comorbidities |
| **PhysioNet 2015 Challenge** | Cardiac arrest ECG sequences | Test detection algorithm |
| **UCI Heart Disease** | 303 patient records | FOLD-RM classification benchmark |
| **AHA Cardiac Arrest Registry** | 150,000+ cases | Validation & epidemiology |

> All datasets are **publicly available and free** — no IRB needed for training phase.

---

## 12. Publication Roadmap

```mermaid
graph LR
    P1["📄 Paper 1\n(Month 6)\nFOLD-RM for Real-Time\nCardiac Arrest Detection\n→ ICLP 2026 / AAAI 2026"]
    P2["📄 Paper 2\n(Month 9)\nCounterfactual Explanations\nfor Cardiac Events\n→ TPLP (Gupta's own journal!)"]
    P3["📄 Paper 3\n(Month 12)\nSmartCPR Full System:\nAI + IoT + Auto-CPR\n→ IEEE EMBC / Nature Dig. Med."]
    PAT["📋 Patent\n(Month 10)\nWearable CPR Auto-Trigger\nUS Patent Application"]

    P1 --> P2 --> P3
    P3 --> PAT

    style P1 fill:#1565c0,color:#fff
    style P2 fill:#4a148c,color:#fff
    style P3 fill:#1b5e20,color:#fff
    style PAT fill:#e65100,color:#fff
```
---

## 🚀 Getting Started

Follow these steps to run the **SmartCPR Guardian** prototype simulation on your local machine.

### Prerequisites
- Python 3.8+
- `pip`

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/RachitJava/SmartCPR-Guardian.git
   cd SmartCPR-Guardian
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Prototype
1. **Start Detection Simulation**: `python ai_engine/fold_rm.py`
2. **Start Alert Dispatcher**: `python alert_system/dispatcher.py`

---

## 📎 Supporting Materials Checklist
- [ ] This proposal document (PDF export)
- [ ] CV (Python/ML + Embedded exp)
- [ ] Transcripts
- [ ] GitHub showing relevant projects

---

> **Key References to Read:**
> - FOLD-RM Paper: Wang, Shakerin, Gupta — TPLP 2022
> - MC3G Paper: Dasgupta et al. — arXiv 2025
> - AutoCompanion: Zeng et al. — TPLP Jan 2025
> - CHF Advisory System: Chen, Gupta — ICLP 2016
> - NSF Award #1916206 (HF Diagnosis project)
