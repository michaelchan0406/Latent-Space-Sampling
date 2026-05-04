#!/usr/bin/env python3
"""
baselines.py

Reviewer-friendly baseline generation script for medical benchmark question generation.

Implemented baselines:
  1. m1               : single-prompt direct LLM question generation
  2. m2               : bulk-prompt direct LLM question generation
  3. self_instruct    : Distilabel SelfInstruct baseline
  4. wizardlm         : Distilabel EvolInstruct / WizardLM-style baseline
  5. med_rag          : retrieval-augmented generation from PubMedQA + MedQuAD contexts

Design goals:
  - No local paths, names, institutions, or API keys.
  - All model/provider settings are controlled by CLI arguments and environment variables.
  - Same code works for OpenAI-compatible models such as GPT, Qwen, and Gemini.
  - Outputs are JSONL with a consistent schema.
  - Supports resume, de-duplication, seed control, and basic error handling.

Expected environment variables:
  OPENAI_API_KEY   : API key for OpenAI-compatible GPT models.
  QWEN_API_KEY     : API key for Qwen / DashScope OpenAI-compatible endpoint.
  GEMINI_API_KEY   : API key for Gemini OpenAI-compatible endpoint.

Optional environment variables:
  OPENAI_BASE_URL  : Custom OpenAI-compatible endpoint for GPT-like models.
  QWEN_BASE_URL    : Defaults to https://dashscope.aliyuncs.com/compatible-mode/v1
  GEMINI_BASE_URL  : Defaults to https://generativelanguage.googleapis.com/v1beta/openai/

Example usage:

  # Run M1 and M2 with three models.
  python baselines.py \
    --baselines m1,m2 \
    --models qwen3.5-flash,gpt-5.4,gemini-3-flash-preview \
    --n 2000 \
    --output_dir outputs/baselines

  # Run SelfInstruct for one model.
  python baselines.py \
    --baselines self_instruct \
    --models gemini-3-flash-preview \
    --num_instructions 13 \
    --output_dir outputs/baselines

  # Run WizardLM/EvolInstruct.
  python baselines.py \
    --baselines wizardlm \
    --models gpt-5.4 \
    --wizardlm_expansion_factor 13 \
    --output_dir outputs/baselines

  # Run Med-RAG baseline.
  python baselines.py \
    --baselines med_rag \
    --models gpt-5.4 \
    --n 2500 \
    --max_workers 10 \
    --output_dir outputs/baselines

Dependencies:
  pip install openai pandas numpy tqdm datasets distilabel
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "The `openai` package is required. Install it with: pip install openai"
    ) from exc

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

SUPPORTED_BASELINES = {
    "m1",
    "m2",
    "self_instruct",
    "wizardlm",
    "med_rag",
}

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

MEDICAL_TOPICS = [
    "Infectious and parasitic diseases",
    "Neoplasms and Cancers",
    "Hematology",
    "Endocrine and metabolic diseases",
    "Psychiatry",
    "Neurology",
    "Cardiology",
    "Pulmonology",
    "Gastroenterology",
]

TARGET_AUDIENCES = [
    "medical students",
    "specialist doctors",
    "nurses",
    "clinical researchers",
    "general practitioners",
]

MEDICAL_CATEGORIES: Dict[str, List[str]] = {
    "Cardiology": ["heart", "cardiovascular", "hypertension", "arrhythmia", "ecg"],
    "Oncology": ["cancer", "tumor", "chemotherapy", "carcinoma", "malignancy"],
    "Neurology": ["brain", "neurological", "seizure", "stroke", "alzheimer"],
    "Infectious_Diseases": ["infection", "virus", "bacteria", "antibiotic", "vaccine"],
    "Endocrinology": ["diabetes", "thyroid", "hormone", "metabolic", "adrenal"],
    "Pediatrics": ["child", "infant", "pediatric", "neonatal", "developmental"],
    "Gastroenterology": ["digestive", "liver", "gastric", "intestine", "hepatitis"],
    "Psychiatry": ["mental", "depression", "anxiety", "schizophrenia", "psychological"],
    "Dermatology": ["skin", "rash", "dermatitis", "melanoma", "psoriasis"],
    "Rare_Diseases": ["congenital", "genetic", "syndrome", "hereditary", "mutation"],
}


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    """Provider-specific OpenAI-compatible client configuration."""

    provider: str
    api_key: str
    base_url: Optional[str]


@dataclass
class GenerationRecord:
    """Unified output schema for all baselines."""

    baseline: str
    model: str
    sample_id: int
    text: str
    raw_text: Optional[str]
    metadata: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(
            {
                "baseline": self.baseline,
                "model": self.model,
                "sample_id": self.sample_id,
                "text": self.text,
                "raw_text": self.raw_text,
                "metadata": self.metadata,
            },
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------
# Logging and utility helpers
# ---------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def safe_model_name(model_name: str) -> str:
    """Convert a model name into a filesystem-safe string."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name.strip())


def parse_csv_list(value: str) -> List[str]:
    """Parse comma-separated CLI values."""
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def progress_bar(iterable: Iterable[Any], **kwargs: Any) -> Iterable[Any]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def get_output_path(args: argparse.Namespace, baseline: str, model: str) -> Path:
    if args.output_file:
        baselines = parse_csv_list(args.baselines)
        models = parse_csv_list(args.models)
        if len(baselines) != 1 or len(models) != 1:
            raise ValueError(
                "--output_file can only be used when exactly one baseline and one model are requested."
            )
        return Path(args.output_file)

    filename = f"{baseline}__{safe_model_name(model)}.jsonl"
    return Path(args.output_dir) / filename


def append_jsonl(path: Path, record: GenerationRecord) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(record.to_json() + "\n")


def read_existing_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning("Skipping malformed JSONL line %d in %s", line_no, path)
    return records


def existing_text_set(path: Path) -> set[str]:
    records = read_existing_jsonl(path)
    return {
        str(record.get("text", "")).strip()
        for record in records
        if str(record.get("text", "")).strip()
    }


def existing_record_count(path: Path) -> int:
    return len(read_existing_jsonl(path))


# ---------------------------------------------------------------------
# Provider and LLM call handling
# ---------------------------------------------------------------------

def infer_provider(model_name: str) -> str:
    lower = model_name.lower()
    if "qwen" in lower:
        return "qwen"
    if "gemini" in lower:
        return "gemini"
    return "openai"


def get_provider_config(model_name: str, args: argparse.Namespace) -> ProviderConfig:
    provider = args.provider
    if provider == "auto":
        provider = infer_provider(model_name)

    if provider == "qwen":
        api_key = os.getenv(args.qwen_api_key_env)
        base_url = args.qwen_base_url or os.getenv("QWEN_BASE_URL", DEFAULT_QWEN_BASE_URL)
    elif provider == "gemini":
        api_key = os.getenv(args.gemini_api_key_env)
        base_url = args.gemini_base_url or os.getenv("GEMINI_BASE_URL", DEFAULT_GEMINI_BASE_URL)
    elif provider == "openai":
        api_key = os.getenv(args.openai_api_key_env)
        base_url = args.openai_base_url or os.getenv("OPENAI_BASE_URL")
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    if not api_key:
        env_var = {
            "qwen": args.qwen_api_key_env,
            "gemini": args.gemini_api_key_env,
            "openai": args.openai_api_key_env,
        }[provider]
        raise EnvironmentError(
            f"Missing API key for provider '{provider}'. "
            f"Please set environment variable {env_var}."
        )

    return ProviderConfig(provider=provider, api_key=api_key, base_url=base_url)


def make_openai_client(config: ProviderConfig, timeout: float) -> OpenAI:
    if config.base_url:
        return OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=timeout)
    return OpenAI(api_key=config.api_key, timeout=timeout)


def _chat_completion_with_token_fallback(
    client: OpenAI,
    provider: str,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> str:
    """
    OpenAI-compatible endpoints differ on whether they expect `max_tokens`
    or `max_completion_tokens`. This helper tries the likely parameter first
    and falls back to the other parameter if needed.
    """
    common_kwargs: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    token_param_order = (
        ["max_completion_tokens", "max_tokens"]
        if provider == "openai"
        else ["max_tokens", "max_completion_tokens"]
    )

    last_error: Optional[Exception] = None

    for token_param in token_param_order:
        kwargs = dict(common_kwargs)
        kwargs[token_param] = max_tokens
        try:
            response = client.chat.completions.create(**kwargs)
            if response.choices:
                content = response.choices[0].message.content
                return (content or "").strip()
            return ""
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if token_param == token_param_order[-1]:
                break
            if "max_tokens" not in message and "max_completion_tokens" not in message:
                break

    if last_error is not None:
        raise last_error
    return ""


def call_llm(
    client: OpenAI,
    provider: str,
    model_name: str,
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 512,
    retries: int = 3,
    retry_sleep: float = 2.0,
) -> str:
    """Call an OpenAI-compatible chat-completion model with retries."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(retries + 1):
        try:
            return _chat_completion_with_token_fallback(
                client=client,
                provider=provider,
                model_name=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            if attempt >= retries:
                logging.error(
                    "LLM call failed after %d attempts | model=%s | error=%s",
                    retries + 1,
                    model_name,
                    repr(exc),
                )
                return ""

            wait_time = retry_sleep * (2 ** attempt)
            logging.warning(
                "LLM call failed; retrying in %.1fs | attempt=%d/%d | model=%s | error=%s",
                wait_time,
                attempt + 1,
                retries,
                model_name,
                repr(exc),
            )
            time.sleep(wait_time)

    return ""


# ---------------------------------------------------------------------
# Question cleaning and validation
# ---------------------------------------------------------------------

def clean_one_question(text: str) -> str:
    """
    Clean a model output into a single question-like sentence.

    This intentionally preserves the notebook behavior:
      - keep only the first line;
      - remove common prefixes / numbering;
      - truncate after the first question mark.
    """
    text = (text or "").strip()

    if "\n" in text:
        text = text.split("\n")[0].strip()

    text = re.sub(r"^\s*(Question\s*:\s*)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*[\-\*\•]\s*", "", text)
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text)
    text = text.strip().strip('"').strip("'").strip()

    if "?" in text:
        text = text.split("?")[0].strip() + "?"

    return text


def is_valid_question(text: str, min_chars: int = 20) -> bool:
    if not text:
        return False
    text = text.strip()
    if len(text) < min_chars:
        return False
    if not text.endswith("?"):
        return False
    if "?" not in text:
        return False
    return True


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Parse a JSON object from model output, with a conservative fallback."""
    text = (text or "").strip()
    if not text:
        return None

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None

    try:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        return None

    return None


def deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for item in items:
        item = item.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


# ---------------------------------------------------------------------
# Built-in medical seed pool
# ---------------------------------------------------------------------

def default_medical_seed_instructions() -> List[str]:
    """
    Built-in seed pool derived from the notebook seed list.

    Users can replace this with --seed_file for exact reproducibility if the
    paper release includes a separate seed file.
    """
    seeds = [
        # Cardiology
        "What are the primary diagnostic criteria for acute ST-segment elevation myocardial infarction (STEMI)?",
        "Explain the pathophysiology of heart failure with preserved ejection fraction (HFpEF).",
        "Outline the 'CHADS2-VASc' score and its role in anticoagulation for Atrial Fibrillation.",
        "Describe the clinical signs and diagnostic workup for acute infective endocarditis.",
        "What are the indications for permanent pacemaker insertion in patients with AV block?",
        "Differentiate between the murmurs of aortic stenosis and hypertrophic obstructive cardiomyopathy.",
        "Describe the management of a hypertensive emergency with evidence of encephalopathy.",
        "What are the EKG findings suggestive of Brugada syndrome?",
        "Explain the role of cardiac biomarkers (Troponin I vs. T) in the diagnosis of NSTEMI.",
        "Describe the pathophysiology and treatment of restrictive cardiomyopathy vs. constrictive pericarditis.",

        # Neurology
        "Differentiate between ischemic and hemorrhagic stroke using non-contrast CT imaging.",
        "What are the early clinical markers for Amyotrophic Lateral Sclerosis (ALS)?",
        "Describe the diagnostic criteria for Multiple Sclerosis according to the McDonald Criteria.",
        "What is the management of status epilepticus in the first 30 minutes?",
        "Distinguish between Parkinson's Disease and Progressive Supranuclear Palsy (PSP).",
        "Explain the pathophysiology of Myasthenia Gravis and the role of the Tensilon test.",
        "Describe the clinical presentation of a subarachnoid hemorrhage and the 'Thunderclap' headache.",
        "What are the characteristic CSF findings in Guillain-Barré Syndrome?",
        "Outline the management of idiopathic intracranial hypertension (Pseudotumor Cerebri).",
        "Describe the Glasgow Coma Scale (GCS) and its limitations in intubated patients.",

        # Oncology
        "Describe the mechanism of action and side effects of Immune Checkpoint Inhibitors (PD-1/PD-L1).",
        "What is Tumor Lysis Syndrome (TLS) and how is it managed prophylactically?",
        "Explain the TNM staging system for Non-Small Cell Lung Cancer (NSCLC).",
        "What are the primary risk factors and screening guidelines for Colorectal Cancer?",
        "Describe the 'B Symptoms' associated with Hodgkin Lymphoma and their prognostic value.",
        "What are the indications for BRCA1/BRCA2 genetic testing in breast cancer patients?",
        "Explain the clinical management of febrile neutropenia in an oncology patient.",
        "Describe the paraneoplastic syndromes associated with Small Cell Lung Cancer.",
        "What is the role of PSA kinetics in monitoring prostate cancer recurrence?",
        "Describe the grading and staging of Multiple Myeloma using the Revised International Staging System.",

        # Endocrinology
        "Explain the diagnostic Low-Dose Dexamethasone Suppression Test for Cushing’s Syndrome.",
        "What are the long-term microvascular complications of uncontrolled Type 2 Diabetes?",
        "Describe the clinical management protocol for Myxedema Coma.",
        "What is the 'Whipple Triad' and its significance in diagnosing insulinoma?",
        "Differentiate between Graves' Disease and Subacute Thyroiditis based on radioactive iodine uptake.",
        "Explain the pathophysiology of Diabetes Insipidus (Central vs. Nephrogenic).",
        "What are the electrolyte abnormalities typically seen in Primary Adrenal Insufficiency (Addison's)?",
        "Describe the management of Diabetic Ketoacidosis (DKA) focusing on potassium replacement.",
        "What is the diagnostic approach for a patient with a suspected pheochromocytoma?",
        "Outline the criteria for diagnosing Polycystic Ovary Syndrome (PCOS) using the Rotterdam Criteria.",

        # Gastroenterology
        "Describe the 'Light’s Criteria' for differentiating exudative from transudative pleural effusions.",
        "Outline the management strategy for suspected variceal bleeding in liver cirrhosis.",
        "Differentiate between Crohn’s Disease and Ulcerative Colitis based on colonoscopic findings.",
        "What are the diagnostic markers for Autoimmune Hepatitis?",
        "Describe the MELD score and its role in liver transplant prioritization.",
        "What are the complications of chronic Hepatitis C infection if left untreated?",
        "Explain the pathophysiology and clinical presentation of Acute Pancreatitis.",
        "What is the 'Child-Pugh Score' and how does it assess hepatic reserve?",
        "Describe the management of C. difficile infection (Initial vs. Recurrent).",
        "What are the indications for ERCP in a patient with suspected choledocholithiasis?",

        # Nephrology
        "What are the indications for urgent renal replacement therapy (dialysis) in AKI?",
        "Explain the diagnostic approach to metabolic acidosis using the Anion Gap.",
        "Differentiate between Nephrotic and Nephritic syndromes based on urinalysis.",
        "What is the pathophysiology of Hepatorenal Syndrome?",
        "Describe the management of hyperkalemia with EKG changes.",
        "Explain the stages of Chronic Kidney Disease (CKD) based on GFR.",
        "What are the characteristic biopsy findings in Minimal Change Disease?",
        "Describe the clinical features of Autosomal Dominant Polycystic Kidney Disease (ADPKD).",
        "How do you calculate and interpret the Fractional Excretion of Sodium (FeNa)?",
        "What are the complications of rapid correction of hyponatremia?",

        # Infectious Disease
        "Describe the 'Duke Criteria' for the diagnosis of Infective Endocarditis.",
        "Outline the current first-line antibiotic recommendations for Community-Acquired Pneumonia.",
        "What is the clinical significance of the CD4 count in HIV/AIDS management?",
        "Differentiate between Latent and Active Tuberculosis (TB) diagnosis.",
        "Describe the clinical presentation of Lyme Disease and the 'Erythema Migrans' rash.",
        "What are the risk factors for Multi-Drug Resistant (MDR) infections in hospitals?",
        "Explain the 'Window Period' in HIV testing and the role of 4th Gen assays.",
        "What is the management of Sepsis based on the 1-hour bundle guidelines?",
        "Describe the pathophysiology of Malaria and the 'Paroxysm' cycle.",
        "What are the isolation precautions required for Meningococcal Meningitis?",

        # Surgery
        "Describe the 'Alvarado Score' for the diagnosis of acute appendicitis.",
        "What are the indications for surgical intervention in a patient with a small bowel obstruction?",
        "Outline the 'Rule of Nines' for estimating total body surface area in burn patients.",
        "Describe the 'Triad of Death' in trauma surgery: Acidosis, Coagulopathy, Hypothermia.",
        "What is the 'Post-cholecystectomy Syndrome' and its common causes?",
        "Differentiate between incarcerated and strangulated hernias.",
        "Describe the pre-operative evaluation of a patient with significant cardiac history.",
        "What are the surgical margins required for a localized Malignant Melanoma?",
        "Explain the 'SIRS' criteria and its evolution toward the SOFA score.",
        "Describe the complications of a total thyroidectomy, focusing on recurrent laryngeal nerve injury.",

        # Obstetrics & Gynecology
        "Describe the management of Preeclampsia with severe features.",
        "What are the clinical stages of labor and the definition of 'arrest of descent'?",
        "Explain the pathophysiology of Ectopic Pregnancy and the role of Beta-hCG levels.",
        "Describe the management of Shoulder Dystocia using the HELPERR mnemonic.",
        "What are the screening guidelines for Cervical Cancer (Pap vs. HPV co-testing)?",
        "Differentiate between Placenta Previa and Placental Abruption.",
        "What is the 'HELLP Syndrome' and how is it definitively treated?",
        "Explain the management of Gestational Diabetes (Diet vs. Insulin).",
        "Describe the clinical markers for ovarian reserve (AMH vs. Antral Follicle Count).",
        "What are the contraindications for Combined Oral Contraceptive Pills?",

        # Pediatrics
        "Distinguish between physiological and pathological jaundice in a 3-day-old neonate.",
        "Describe the clinical features and management of Croup (Laryngotracheobronchitis).",
        "What are the radiographic signs of Necrotizing Enterocolitis (NEC) in premature infants?",
        "Outline the management of acute dehydration in a pediatric patient using the '4-2-1 Rule'.",
        "Describe the clinical presentation of Pyloric Stenosis and the 'Olive' mass.",
        "What are the red flags for developmental delay in an 18-month-old child?",
        "Describe the pathophysiology of Tetralogy of Fallot and the 'Tet Spell'.",
        "What are the diagnostic criteria for Kawasaki Disease?",
        "Describe the management of Febrile Seizures in children.",
        "Explain the 'Apgar Score' and its significance at 1 and 5 minutes.",

        # Additional specialized seeds
        "Explain the 'Virchow’s Triad' and its role in Venous Thromboembolism.",
        "Describe the clinical markers of Iron Deficiency Anemia vs. Anemia of Chronic Disease.",
        "What is the pathophysiology of Hemophilia A vs. B?",
        "Describe the management of Sickle Cell Vaso-occlusive Crisis.",
        "What is the diagnostic significance of anti-CCP antibodies in Rheumatoid Arthritis?",
        "Describe the 'CREST Syndrome' and its relationship to Scleroderma.",
        "What are the ACR criteria for Systemic Lupus Erythematosus (SLE)?",
        "Differentiate between Osteoarthritis and Rheumatoid Arthritis on hand X-ray.",
        "Describe the ABCDE criteria for the evaluation of melanoma suspicion.",
        "What is the 'Rule of Nines' in burn management?",
        "Distinguish between Stevens-Johnson Syndrome (SJS) and Toxic Epidermal Necrolysis (TEN).",
        "Explain the 'Beers Criteria' for medications generally avoided in the elderly.",
        "Describe the DSM-5 criteria for a Major Depressive Episode.",
        "What is the management of Opioid Overdose using Naloxone?",
        "Describe the clinical features of Serotonin Syndrome vs. Neuroleptic Malignant Syndrome.",
        "Explain the 'Ottawa Ankle Rules' for X-ray necessity in ankle injuries.",
        "What are the Hounsfield Unit (HU) ranges for blood, water, and fat on a CT scan?",
        "Explain the difference between ionizing radiation (CT) and non-ionizing radiation (MRI).",
        "Discuss the clinical challenges of managing MDR-TB in resource-limited settings.",
        "Define 'Sarcopenia' in the elderly and its impact on frailty.",
        "Describe the pathophysiology of Hereditary Angioedema (C1-esterase deficiency).",
        "Explain the 'window period' in HIV testing and why 4th Gen tests are preferred.",
        "Describe the components of the MELD score in transplant prioritization.",
        "Outline the 'Spasticity Management Ladder' for rehabilitation therapy.",
        "Explain the concept of 'informed consent' and its legal/ethical requirements.",
        "Describe the clinical signs of Vitamin B12 deficiency and MMA level testing.",
    ]

    specialty_fillers = [
        f"Explain the clinical management of {topic}."
        for topic in [
            "Psoriatic Arthritis",
            "Sarcoidosis",
            "Ankylosing Spondylitis",
            "Gouty Tophus",
            "Septic Arthritis",
            "Bullous Pemphigoid",
            "Pemphigus Vulgaris",
            "Atopic Dermatitis",
            "Erythema Nodosum",
            "Pyoderma Gangrenosum",
            "Giant Cell Arteritis",
            "Polymyalgia Rheumatica",
            "Behcet’s Disease",
            "Wegener's Granulomatosis",
            "Churg-Strauss Syndrome",
            "Multiple Sclerosis Relapse",
            "Guillain-Barre Treatment",
            "Myasthenia Crisis",
            "Lambert-Eaton Syndrome",
            "Trigeminal Neuralgia",
            "Normal Pressure Hydrocephalus",
            "Wernicke-Korsakoff Syndrome",
            "Huntington’s Pathophysiology",
            "Wilson’s Disease Diagnosis",
            "Hemochromatosis Screening",
            "Alpha-1 Antitrypsin Deficiency",
            "Bronchiectasis Management",
            "Idiopathic Pulmonary Fibrosis",
            "Asbestosis vs. Silicosis",
            "Hypersensitivity Pneumonitis",
            "Sleep Apnea Diagnosis",
            "Narcolepsy Features",
            "Restless Leg Syndrome",
            "Pulmonary Hypertension Stages",
            "Cor Pulmonale Signs",
            "Hypertrophic Cardiomyopathy Management",
            "Mitral Valve Prolapse",
            "Tricuspid Regurgitation Causes",
            "Restrictive Pericarditis",
            "Aortic Dissection Types",
            "Peripheral Artery Disease Screening",
            "Raynaud’s Phenomenon",
            "Buerger’s Disease",
            "Aneurysm Screening Guidelines",
            "Atrial Myxoma Features",
            "Von Willebrand Disease Types",
            "Thrombotic Thrombocytopenic Purpura (TTP)",
            "Immune Thrombocytopenia (ITP)",
            "Disseminated Intravascular Coagulation (DIC)",
            "Heparin-Induced Thrombocytopenia (HIT)",
            "Polycythemia Vera Treatment",
            "Essential Thrombocythemia",
            "Myelofibrosis Clinical Signs",
            "CML Blast Crisis",
            "CLL Staging",
            "Acute Myeloid Leukemia (AML) Morphology",
            "Aplastic Anemia Workup",
            "Paroxysmal Nocturnal Hemoglobinuria",
            "Thalassemia Alpha vs Beta",
            "Glucose-6-Phosphate Dehydrogenase Deficiency",
            "Gastroesophageal Reflux Disease (GERD)",
            "Barrett's Esophagus Surveillance",
            "Achalasia Diagnostic Workup",
            "Zenker's Diverticulum",
            "Gastroparesis Management",
            "Peptic Ulcer Disease Complications",
            "Zollinger-Ellison Syndrome",
            "Celiac Disease Serology",
            "Small Intestinal Bacterial Overgrowth (SIBO)",
            "Short Bowel Syndrome",
            "Irritable Bowel Syndrome (IBS) Subtypes",
            "Ischemic Colitis vs. Mesenteric Ischemia",
            "Diverticulitis Management",
            "Anal Fissure Treatment",
            "Hemorrhoid Grading",
        ]
    ]

    seeds.extend(specialty_fillers)
    return seeds[:200]


def load_seed_instructions(seed_file: Optional[str]) -> List[str]:
    """
    Load seed instructions from an optional external file.

    Supported formats:
      - .txt   : one instruction per line
      - .csv   : column named instruction, question, or text
      - .jsonl : field named instruction, question, or text
    """
    if not seed_file:
        return default_medical_seed_instructions()

    path = Path(seed_file)
    if not path.exists():
        raise FileNotFoundError(f"Seed file does not exist: {path}")

    if path.suffix.lower() == ".txt":
        seeds = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        return [s for s in seeds if s]

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        for col in ["instruction", "question", "text"]:
            if col in df.columns:
                return df[col].dropna().astype(str).str.strip().tolist()
        raise ValueError(f"CSV seed file must contain one of: instruction, question, text. Columns={df.columns.tolist()}")

    if path.suffix.lower() == ".jsonl":
        seeds: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                for key in ["instruction", "question", "text"]:
                    if key in obj and str(obj[key]).strip():
                        seeds.append(str(obj[key]).strip())
                        break
        return seeds

    raise ValueError(f"Unsupported seed file format: {path.suffix}")


# ---------------------------------------------------------------------
# M1 baseline: single-prompt direct generation
# ---------------------------------------------------------------------

def run_m1_single_prompt(
    args: argparse.Namespace,
    model_name: str,
    client: OpenAI,
    provider: str,
    output_path: Path,
) -> int:
    """
    M1: generate exactly one medical question per LLM call.

    Original notebook behavior:
      - randomly choose a broad topic and target audience;
      - ask model to generate one question;
      - clean and validate the result;
      - save incrementally.
    """
    logging.info("Running M1 single-prompt baseline | model=%s | target=%d", model_name, args.n)

    ensure_output_dir(output_path.parent)
    seen = existing_text_set(output_path) if args.resume else set()
    start_count = existing_record_count(output_path) if args.resume else 0

    if not args.resume and output_path.exists():
        output_path.unlink()

    if start_count >= args.n:
        logging.info("Output already contains %d records; skipping.", start_count)
        return start_count

    pbar_total = args.n
    pbar = tqdm(total=pbar_total, initial=start_count) if tqdm is not None else None

    count = start_count
    while count < args.n:
        topic = random.choice(MEDICAL_TOPICS)
        target = random.choice(TARGET_AUDIENCES)

        prompt = (
            f"Generate exactly ONE medical question related to {topic}. "
            f"Target audience: {target}. "
            "Rule: Output ONLY a single sentence ending with a question mark '?'. "
            "Do NOT provide explanations, answers, or multiple questions."
        )

        raw_text = call_llm(
            client=client,
            provider=provider,
            model_name=model_name,
            prompt=prompt,
            system_prompt="You are a helpful medical question generator.",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_m1,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )

        cleaned = clean_one_question(raw_text)

        if is_valid_question(cleaned, min_chars=args.min_question_chars) and cleaned not in seen:
            seen.add(cleaned)
            record = GenerationRecord(
                baseline="m1",
                model=model_name,
                sample_id=count,
                text=cleaned,
                raw_text=raw_text,
                metadata={
                    "topic": topic,
                    "target_audience": target,
                    "sampling_method": "single_prompt",
                },
            )
            append_jsonl(output_path, record)
            count += 1

            if pbar is not None:
                pbar.update(1)

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

    if pbar is not None:
        pbar.close()

    logging.info("Finished M1 | model=%s | saved=%d | path=%s", model_name, count, output_path)
    return count


# ---------------------------------------------------------------------
# M2 baseline: bulk-prompt direct generation
# ---------------------------------------------------------------------

def run_m2_bulk_prompt(
    args: argparse.Namespace,
    model_name: str,
    client: OpenAI,
    provider: str,
    output_path: Path,
) -> int:
    """
    M2: generate a batch of medical questions per LLM call.

    Note:
      This corresponds to the original notebook's "Method 4: Bulk Baseline".
      It is renamed here as M2 to match the paper-facing baseline naming.
    """
    logging.info(
        "Running M2 bulk-prompt baseline | model=%s | target=%d | batch_size=%d",
        model_name,
        args.n,
        args.batch_size,
    )

    ensure_output_dir(output_path.parent)
    seen = existing_text_set(output_path) if args.resume else set()
    start_count = existing_record_count(output_path) if args.resume else 0

    if not args.resume and output_path.exists():
        output_path.unlink()

    if start_count >= args.n:
        logging.info("Output already contains %d records; skipping.", start_count)
        return start_count

    pbar = tqdm(total=args.n, initial=start_count) if tqdm is not None else None

    count = start_count
    while count < args.n:
        remaining = args.n - count
        batch_size = min(args.batch_size, remaining)

        prompt = (
            f"Generate exactly {batch_size} distinct medical questions.\n"
            "Rules:\n"
            f"- Exactly {batch_size} lines.\n"
            "- One question per line.\n"
            "- No numbering.\n"
            "- Ends with '?'."
        )

        raw_text = call_llm(
            client=client,
            provider=provider,
            model_name=model_name,
            prompt=prompt,
            system_prompt="You are a helpful medical question generator.",
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_m2,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )

        if not raw_text:
            continue

        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        new_questions: List[str] = []

        for line in lines:
            question = clean_one_question(line)
            if not is_valid_question(question, min_chars=args.min_question_chars):
                continue
            if question in seen:
                continue
            seen.add(question)
            new_questions.append(question)

        if not new_questions:
            logging.debug("M2 call produced no valid new questions.")
            continue

        for question in new_questions:
            if count >= args.n:
                break
            record = GenerationRecord(
                baseline="m2",
                model=model_name,
                sample_id=count,
                text=question,
                raw_text=raw_text,
                metadata={
                    "sampling_method": "bulk_prompt",
                    "requested_batch_size": batch_size,
                },
            )
            append_jsonl(output_path, record)
            count += 1

            if pbar is not None:
                pbar.update(1)

        if args.sleep_between_calls > 0:
            time.sleep(args.sleep_between_calls)

    if pbar is not None:
        pbar.close()

    logging.info("Finished M2 | model=%s | saved=%d | path=%s", model_name, count, output_path)
    return count


# ---------------------------------------------------------------------
# Distilabel compatibility helpers
# ---------------------------------------------------------------------

def patch_openai_async_for_compatible_endpoints(
    force_max_tokens: int = 4096,
    debug_raw_outputs: int = 0,
) -> None:
    """
    Patch Distilabel's async OpenAI call path for strict OpenAI-compatible endpoints.

    This is mainly useful for Gemini-compatible and some third-party endpoints
    that reject OpenAI-specific fields such as tools, response_format, seed,
    or max_completion_tokens.

    The patch is intentionally narrow:
      - removes known unsupported kwargs;
      - converts structured message content into plain strings;
      - forces max_tokens to a sufficiently large value for SelfInstruct/EvolInstruct.
    """
    import asyncio
    import openai

    if not hasattr(openai, "_baseline_original_async_create"):
        openai._baseline_original_async_create = openai.resources.chat.completions.AsyncCompletions.create

    openai._baseline_debug_count = 0

    def deep_remove_none(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: deep_remove_none(v) for k, v in obj.items() if v is not None}
        if isinstance(obj, list):
            return [deep_remove_none(x) for x in obj if x is not None]
        return obj

    def sanitize_messages(messages: Any) -> List[Dict[str, str]]:
        sanitized: List[Dict[str, str]] = []

        if not isinstance(messages, list):
            return [{"role": "user", "content": str(messages)}]

        for msg in messages:
            if msg is None:
                continue
            msg = dict(msg)
            role = str(msg.get("role") or "user")
            content = msg.get("content")

            if content is None:
                content_text = ""
            elif isinstance(content, str):
                content_text = content
            elif isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict):
                        text = part.get("text") or part.get("content") or ""
                        if text:
                            parts.append(str(text))
                    elif part is not None:
                        parts.append(str(part))
                content_text = "\n".join(parts)
            else:
                content_text = str(content)

            sanitized.append({"role": role, "content": content_text})

        return sanitized

    async def patched_create(self: Any, *call_args: Any, **kwargs: Any) -> Any:
        unsupported_keys = [
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "top_logprobs",
            "response_format",
            "tool_choice",
            "tools",
            "parallel_tool_calls",
            "user",
            "seed",
            "service_tier",
            "reasoning_effort",
            "store",
            "metadata",
        ]

        for key in unsupported_keys:
            kwargs.pop(key, None)

        kwargs.pop("stop", None)
        kwargs.pop("max_completion_tokens", None)
        kwargs["max_tokens"] = force_max_tokens

        if "messages" in kwargs:
            kwargs["messages"] = sanitize_messages(kwargs["messages"])

        kwargs = deep_remove_none(kwargs)
        await asyncio.sleep(0.2)

        completion = await openai._baseline_original_async_create(self, *call_args, **kwargs)

        if debug_raw_outputs > 0 and openai._baseline_debug_count < debug_raw_outputs:
            logging.info("Distilabel raw output preview:")
            try:
                raw_text = completion.choices[0].message.content
                logging.info("%s", repr(raw_text)[:4000])
            except Exception as exc:
                logging.warning("Could not read raw output preview: %s", repr(exc))
            openai._baseline_debug_count += 1

        return completion

    openai.resources.chat.completions.AsyncCompletions.create = patched_create
    logging.info("Applied OpenAI-compatible async patch for Distilabel.")


def maybe_patch_distilabel(args: argparse.Namespace, provider: str) -> None:
    if args.disable_distilabel_patch:
        return

    if args.force_distilabel_patch or provider in {"gemini", "qwen"}:
        patch_openai_async_for_compatible_endpoints(
            force_max_tokens=args.distilabel_force_max_tokens,
            debug_raw_outputs=args.distilabel_debug_raw_outputs,
        )


def clear_distilabel_cache_if_requested(args: argparse.Namespace) -> None:
    if not args.clear_distilabel_cache:
        return

    cache_dir = Path(os.path.expanduser(args.distilabel_cache_dir))
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        logging.info("Cleared Distilabel cache directory: %s", cache_dir)


# ---------------------------------------------------------------------
# SelfInstruct baseline
# ---------------------------------------------------------------------

def ensure_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    if value is None:
        return []
    if hasattr(value, "tolist") and not isinstance(value, str):
        converted = value.tolist()
        if isinstance(converted, list):
            return [str(x) for x in converted if x is not None]
        return [str(converted)]
    return [str(value)]


def run_self_instruct(
    args: argparse.Namespace,
    model_name: str,
    provider_config: ProviderConfig,
    output_path: Path,
) -> int:
    """
    SelfInstruct baseline using Distilabel's SelfInstruct task.

    Original notebook behavior:
      - use a medical seed pool;
      - generate multiple instructions per seed;
      - explode the generated instruction list;
      - remove empty, too-short, duplicated, and seed-identical outputs.
    """
    try:
        from distilabel.llms import OpenAILLM
        from distilabel.pipeline import Pipeline
        from distilabel.steps.tasks import SelfInstruct
    except ImportError as exc:
        raise ImportError(
            "SelfInstruct baseline requires distilabel. Install with: pip install distilabel"
        ) from exc

    logging.info("Running SelfInstruct baseline | model=%s", model_name)

    ensure_output_dir(output_path.parent)
    if output_path.exists() and args.resume and existing_record_count(output_path) >= args.n:
        count = existing_record_count(output_path)
        logging.info("Output already contains %d records; skipping.", count)
        return count

    if not args.resume and output_path.exists():
        output_path.unlink()

    clear_distilabel_cache_if_requested(args)
    maybe_patch_distilabel(args, provider_config.provider)

    seeds = load_seed_instructions(args.seed_file)
    seeds = seeds[: args.max_seed_count]
    if not seeds:
        raise ValueError("No seed instructions available for SelfInstruct.")

    llm = OpenAILLM(
        model=model_name,
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
    )

    with Pipeline(name="medical-self-instruct") as pipeline:
        SelfInstruct(
            llm=llm,
            num_instructions=args.num_instructions,
            application_description=args.self_instruct_description,
            input_mappings={"input": "instruction"},
        )

    dataset = [{"instruction": seed} for seed in seeds]

    distiset = pipeline.run(
        dataset=dataset,
        use_cache=not args.no_cache,
    )

    results_df = distiset["default"]["train"].to_pandas()
    target_col = "instructions"

    if target_col not in results_df.columns:
        raise ValueError(
            f"SelfInstruct output column '{target_col}' not found. "
            f"Available columns: {results_df.columns.tolist()}"
        )

    results_df[target_col] = results_df[target_col].apply(ensure_list)
    results_df["num_generated"] = results_df[target_col].apply(len)

    exploded = results_df.explode(target_col, ignore_index=True)
    exploded = exploded.rename(columns={target_col: "medical_question"})

    output_df = exploded[["instruction", "medical_question"]].copy()
    output_df = output_df.dropna(subset=["medical_question"])
    output_df["medical_question"] = output_df["medical_question"].astype(str).str.strip()
    output_df = output_df[output_df["medical_question"].str.len() > args.min_question_chars]
    output_df = output_df[output_df["medical_question"] != output_df["instruction"]]
    output_df = output_df.drop_duplicates(subset=["medical_question"]).reset_index(drop=True)

    if args.n > 0:
        output_df = output_df.head(args.n)

    if output_df.empty:
        raise RuntimeError("SelfInstruct produced no valid medical questions.")

    count = 0
    for idx, row in output_df.iterrows():
        text = str(row["medical_question"]).strip()
        record = GenerationRecord(
            baseline="self_instruct",
            model=model_name,
            sample_id=count,
            text=text,
            raw_text=None,
            metadata={
                "seed_instruction": str(row["instruction"]),
                "num_instructions_per_seed": args.num_instructions,
                "method": "distilabel_self_instruct",
            },
        )
        append_jsonl(output_path, record)
        count += 1

    logging.info(
        "Finished SelfInstruct | model=%s | saved=%d | path=%s",
        model_name,
        count,
        output_path,
    )
    return count


# ---------------------------------------------------------------------
# WizardLM / EvolInstruct baseline
# ---------------------------------------------------------------------

def run_wizardlm(
    args: argparse.Namespace,
    model_name: str,
    provider_config: ProviderConfig,
    output_path: Path,
) -> int:
    """
    WizardLM-style baseline using Distilabel's EvolInstruct task.

    Original notebook behavior:
      - use the same medical seed pool;
      - expand seed dataset by a fixed factor;
      - run one evolution per expanded instruction;
      - save evolved_instruction as the generated medical question.
    """
    try:
        from distilabel.llms import OpenAILLM
        from distilabel.pipeline import Pipeline
        from distilabel.steps.tasks import EvolInstruct
    except ImportError as exc:
        raise ImportError(
            "WizardLM baseline requires distilabel. Install with: pip install distilabel"
        ) from exc

    logging.info("Running WizardLM/EvolInstruct baseline | model=%s", model_name)

    ensure_output_dir(output_path.parent)
    if output_path.exists() and args.resume and existing_record_count(output_path) >= args.n:
        count = existing_record_count(output_path)
        logging.info("Output already contains %d records; skipping.", count)
        return count

    if not args.resume and output_path.exists():
        output_path.unlink()

    clear_distilabel_cache_if_requested(args)
    maybe_patch_distilabel(args, provider_config.provider)

    seeds = load_seed_instructions(args.seed_file)
    seeds = seeds[: args.max_seed_count]
    if not seeds:
        raise ValueError("No seed instructions available for WizardLM.")

    expanded_dataset = (
        [{"instruction": seed} for seed in seeds]
        * args.wizardlm_expansion_factor
    )

    llm = OpenAILLM(
        model=model_name,
        base_url=provider_config.base_url,
        api_key=provider_config.api_key,
    )

    with Pipeline(name="medical-wizardlm-evol-instruct") as pipeline:
        EvolInstruct(
            llm=llm,
            num_evolutions=args.num_evolutions,
            input_mappings={"instruction": "instruction"},
            input_batch_size=args.distilabel_input_batch_size,
        )

    distiset = pipeline.run(
        dataset=expanded_dataset,
        use_cache=not args.no_cache,
        dataset_batch_size=args.distilabel_dataset_batch_size,
    )

    results_df = distiset["default"]["train"].to_pandas()

    if "evolved_instruction" not in results_df.columns:
        raise ValueError(
            "WizardLM/EvolInstruct output column 'evolved_instruction' not found. "
            f"Available columns: {results_df.columns.tolist()}"
        )

    final_df = results_df.rename(columns={"evolved_instruction": "medical_question"}).copy()
    final_df = final_df.dropna(subset=["medical_question"])
    final_df["medical_question"] = final_df["medical_question"].astype(str).str.strip()
    final_df = final_df[final_df["medical_question"].str.len() > args.min_question_chars]

    if "instruction" in final_df.columns:
        final_df = final_df[["instruction", "medical_question"]]
    else:
        final_df["instruction"] = ""

    final_df = final_df.drop_duplicates(subset=["medical_question"]).reset_index(drop=True)

    if args.n > 0:
        final_df = final_df.head(args.n)

    if final_df.empty:
        raise RuntimeError("WizardLM/EvolInstruct produced no valid medical questions.")

    count = 0
    for _, row in final_df.iterrows():
        text = str(row["medical_question"]).strip()
        record = GenerationRecord(
            baseline="wizardlm",
            model=model_name,
            sample_id=count,
            text=text,
            raw_text=None,
            metadata={
                "seed_instruction": str(row.get("instruction", "")),
                "method": "distilabel_evol_instruct",
                "num_evolutions": args.num_evolutions,
                "expansion_factor": args.wizardlm_expansion_factor,
            },
        )
        append_jsonl(output_path, record)
        count += 1

    logging.info(
        "Finished WizardLM/EvolInstruct | model=%s | saved=%d | path=%s",
        model_name,
        count,
        output_path,
    )
    return count


# ---------------------------------------------------------------------
# Med-RAG baseline
# ---------------------------------------------------------------------

def build_context_pool(
    pubmed_limit: int,
    medquad_limit: Optional[int],
    seed: int,
) -> List[Dict[str, str]]:
    """
    Build a retrieval/context pool from PubMedQA and MedQuAD.

    Original notebook behavior:
      - PubMedQA: use context.contexts joined as background text.
      - MedQuAD: use answer as background text.
      - shuffle the resulting pool.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Med-RAG baseline requires datasets. Install with: pip install datasets"
        ) from exc

    logging.info("Loading PubMedQA and MedQuAD datasets.")

    pubmed = load_dataset(
        "qiaojin/PubMedQA",
        "pqa_unlabeled",
        split="train",
        trust_remote_code=True,
    )
    medquad = load_dataset(
        "lavita/MedQuAD",
        split="train",
        trust_remote_code=True,
    )

    pool: List[Dict[str, str]] = []

    pubmed_sample_size = min(len(pubmed), pubmed_limit)
    pubmed_sample = pubmed.shuffle(seed=seed).select(range(pubmed_sample_size))

    for item in progress_bar(pubmed_sample, desc="Processing PubMedQA"):
        try:
            context = item.get("context")
            if context and context.get("contexts"):
                text = " ".join(context["contexts"])
                if isinstance(text, str) and text.strip():
                    pool.append({"text": text.strip(), "source": "PubMedQA"})
        except Exception:
            continue

    medquad_iterable = medquad
    if medquad_limit is not None and medquad_limit > 0:
        medquad_iterable = medquad.select(range(min(len(medquad), medquad_limit)))

    for item in progress_bar(medquad_iterable, desc="Processing MedQuAD"):
        try:
            answer = item.get("answer")
            if isinstance(answer, str) and answer.strip():
                pool.append({"text": answer.strip(), "source": "MedQuAD"})
        except Exception:
            continue

    rng = random.Random(seed)
    rng.shuffle(pool)

    if not pool:
        raise RuntimeError("Context pool is empty. Check dataset availability and fields.")

    logging.info("Built context pool with %d valid contexts.", len(pool))
    return pool


def med_rag_worker(
    category: str,
    ctx_item: Dict[str, str],
    client: OpenAI,
    provider: str,
    model_name: str,
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    system_prompt = (
        "You are a senior medical professor. "
        "Task: Generate ONE high-quality, open-ended clinical question based on the context. "
        "Requirements: Open-ended, professional reasoning, no options, no answers. "
        "Output JSON only in the format: {\"question\": \"...\"}"
    )

    user_prompt = (
        f"Category: {category}\n"
        f"Context: {ctx_item['text'][: args.max_context_chars]}"
    )

    raw_text = call_llm(
        client=client,
        provider=provider,
        model_name=model_name,
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens_med_rag,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    parsed = extract_json_object(raw_text)
    if not parsed:
        return None

    question = str(parsed.get("question", "")).strip()
    question = clean_one_question(question)

    if not is_valid_question(question, min_chars=args.min_question_chars):
        return None

    return {
        "category": category,
        "source": ctx_item.get("source", ""),
        "question": question,
        "raw_text": raw_text,
    }


def compute_category_targets(total_n: int, categories: Sequence[str]) -> Dict[str, int]:
    if total_n <= 0:
        raise ValueError("--n must be positive for med_rag.")

    base = total_n // len(categories)
    remainder = total_n % len(categories)

    targets: Dict[str, int] = {}
    for idx, category in enumerate(categories):
        targets[category] = base + (1 if idx < remainder else 0)

    return targets


def run_med_rag(
    args: argparse.Namespace,
    model_name: str,
    client: OpenAI,
    provider: str,
    output_path: Path,
) -> int:
    """
    Med-RAG baseline.

    Original notebook behavior:
      - build context pool from PubMedQA and MedQuAD;
      - define 10 broad medical categories with keywords;
      - retrieve keyword-matching contexts per category;
      - supplement with random contexts if needed;
      - generate one open-ended question per context;
      - save as JSONL.
    """
    logging.info("Running Med-RAG baseline | model=%s | target=%d", model_name, args.n)

    ensure_output_dir(output_path.parent)

    existing_records = read_existing_jsonl(output_path) if args.resume else []
    seen = {
        str(record.get("text", "")).strip()
        for record in existing_records
        if str(record.get("text", "")).strip()
    }

    if not args.resume and output_path.exists():
        output_path.unlink()
        existing_records = []
        seen = set()

    existing_count = len(existing_records)
    if existing_count >= args.n:
        logging.info("Output already contains %d records; skipping.", existing_count)
        return existing_count

    context_pool = build_context_pool(
        pubmed_limit=args.pubmed_limit,
        medquad_limit=args.medquad_limit,
        seed=args.seed,
    )

    categories = list(MEDICAL_CATEGORIES.keys())
    category_targets = compute_category_targets(args.n, categories)

    existing_by_category: Counter[str] = Counter()
    for record in existing_records:
        metadata = record.get("metadata", {}) or {}
        category = str(metadata.get("category", ""))
        if category:
            existing_by_category[category] += 1

    rng = random.Random(args.seed)
    total_saved = existing_count
    sample_id = total_saved

    pbar = tqdm(total=args.n, initial=existing_count) if tqdm is not None else None

    for category, keywords in MEDICAL_CATEGORIES.items():
        target_for_category = category_targets[category]
        already_have = existing_by_category.get(category, 0)
        remaining_for_category = max(0, target_for_category - already_have)

        if remaining_for_category <= 0:
            continue

        logging.info(
            "Med-RAG category=%s | target=%d | existing=%d | remaining=%d",
            category,
            target_for_category,
            already_have,
            remaining_for_category,
        )

        relevant = [
            ctx
            for ctx in context_pool
            if ctx.get("text")
            and any(keyword in ctx["text"].lower() for keyword in keywords)
        ]

        rng.shuffle(relevant)
        relevant = relevant[:remaining_for_category]

        if len(relevant) < remaining_for_category:
            gap = remaining_for_category - len(relevant)
            supplement = rng.sample(context_pool, min(gap, len(context_pool)))
            relevant.extend(supplement)

        if not relevant:
            logging.warning("No contexts available for category=%s", category)
            continue

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [
                executor.submit(
                    med_rag_worker,
                    category,
                    ctx_item,
                    client,
                    provider,
                    model_name,
                    args,
                )
                for ctx_item in relevant
            ]

            for future in concurrent.futures.as_completed(futures):
                if total_saved >= args.n:
                    break

                try:
                    result = future.result()
                except Exception as exc:
                    logging.warning("Med-RAG worker failed: %s", repr(exc))
                    continue

                if not result:
                    continue

                question = result["question"]
                if question in seen:
                    continue

                seen.add(question)
                record = GenerationRecord(
                    baseline="med_rag",
                    model=model_name,
                    sample_id=sample_id,
                    text=question,
                    raw_text=result.get("raw_text"),
                    metadata={
                        "category": result.get("category"),
                        "source": result.get("source"),
                        "method": "retrieval_augmented_generation",
                    },
                )
                append_jsonl(output_path, record)

                sample_id += 1
                total_saved += 1

                if pbar is not None:
                    pbar.update(1)

                if total_saved >= args.n:
                    break

    if pbar is not None:
        pbar.close()

    logging.info(
        "Finished Med-RAG | model=%s | saved=%d | path=%s",
        model_name,
        total_saved,
        output_path,
    )
    return total_saved


# ---------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------

def run_one_baseline_for_model(
    baseline: str,
    model_name: str,
    args: argparse.Namespace,
) -> int:
    output_path = get_output_path(args, baseline, model_name)
    provider_config = get_provider_config(model_name, args)

    if baseline in {"m1", "m2", "med_rag"}:
        client = make_openai_client(provider_config, timeout=args.timeout)

        if baseline == "m1":
            return run_m1_single_prompt(
                args=args,
                model_name=model_name,
                client=client,
                provider=provider_config.provider,
                output_path=output_path,
            )

        if baseline == "m2":
            return run_m2_bulk_prompt(
                args=args,
                model_name=model_name,
                client=client,
                provider=provider_config.provider,
                output_path=output_path,
            )

        if baseline == "med_rag":
            return run_med_rag(
                args=args,
                model_name=model_name,
                client=client,
                provider=provider_config.provider,
                output_path=output_path,
            )

    if baseline == "self_instruct":
        return run_self_instruct(
            args=args,
            model_name=model_name,
            provider_config=provider_config,
            output_path=output_path,
        )

    if baseline == "wizardlm":
        return run_wizardlm(
            args=args,
            model_name=model_name,
            provider_config=provider_config,
            output_path=output_path,
        )

    raise ValueError(f"Unsupported baseline: {baseline}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run medical benchmark generation baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Main experiment controls
    parser.add_argument(
        "--baselines",
        type=str,
        default="m1",
        help=(
            "Comma-separated baselines to run. "
            "Use any of: m1,m2,self_instruct,wizardlm,med_rag,all."
        ),
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help=(
            "Comma-separated model names, e.g. "
            "qwen3.5-flash,gpt-5.4,gemini-3-flash-preview"
        ),
    )
    parser.add_argument("--n", type=int, default=2000, help="Target number of generated samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output_dir", type=str, default="outputs/baselines", help="Output directory.")
    parser.add_argument("--output_file", type=str, default=None, help="Optional explicit output JSONL path.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output JSONL if present.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level.")

    # Provider controls
    parser.add_argument(
        "--provider",
        type=str,
        default="auto",
        choices=["auto", "openai", "qwen", "gemini"],
        help="Provider. Use auto to infer from model name.",
    )
    parser.add_argument("--openai_api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--qwen_api_key_env", type=str, default="QWEN_API_KEY")
    parser.add_argument("--gemini_api_key_env", type=str, default="GEMINI_API_KEY")

    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--qwen_base_url", type=str, default=None)
    parser.add_argument("--gemini_base_url", type=str, default=None)

    # General generation controls
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry_sleep", type=float, default=2.0)
    parser.add_argument("--sleep_between_calls", type=float, default=0.0)
    parser.add_argument("--min_question_chars", type=int, default=20)

    # M1 / M2 controls
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size for M2 bulk baseline.")
    parser.add_argument("--max_tokens_m1", type=int, default=1024)
    parser.add_argument("--max_tokens_m2", type=int, default=2048)

    # Seed controls for SelfInstruct and WizardLM
    parser.add_argument(
        "--seed_file",
        type=str,
        default=None,
        help="Optional external seed file: .txt, .csv, or .jsonl.",
    )
    parser.add_argument("--max_seed_count", type=int, default=200)

    # SelfInstruct controls
    parser.add_argument("--num_instructions", type=int, default=13)
    parser.add_argument(
        "--self_instruct_description",
        type=str,
        default=(
            "You are a Senior Medical Professor. Generate advanced, clinically "
            "professional medical questions. Focus on reasoning, guidelines, "
            "and complex pathophysiology. Avoid trivial questions."
        ),
    )

    # WizardLM / EvolInstruct controls
    parser.add_argument("--wizardlm_expansion_factor", type=int, default=13)
    parser.add_argument("--num_evolutions", type=int, default=1)

    # Distilabel controls
    parser.add_argument("--no_cache", action="store_true", help="Disable Distilabel cache.")
    parser.add_argument("--clear_distilabel_cache", action="store_true")
    parser.add_argument(
        "--distilabel_cache_dir",
        type=str,
        default="~/.cache/distilabel",
        help="Distilabel cache directory.",
    )
    parser.add_argument("--distilabel_input_batch_size", type=int, default=8)
    parser.add_argument("--distilabel_dataset_batch_size", type=int, default=32)
    parser.add_argument(
        "--disable_distilabel_patch",
        action="store_true",
        help="Disable compatibility patch for Distilabel OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--force_distilabel_patch",
        action="store_true",
        help="Force compatibility patch even for OpenAI provider.",
    )
    parser.add_argument("--distilabel_force_max_tokens", type=int, default=4096)
    parser.add_argument("--distilabel_debug_raw_outputs", type=int, default=0)

    # Med-RAG controls
    parser.add_argument("--pubmed_limit", type=int, default=15000)
    parser.add_argument(
        "--medquad_limit",
        type=int,
        default=None,
        help="Optional limit for MedQuAD rows. None means use all.",
    )
    parser.add_argument("--max_context_chars", type=int, default=1800)
    parser.add_argument("--max_tokens_med_rag", type=int, default=512)
    parser.add_argument("--max_workers", type=int, default=10)

    return parser


def validate_args(args: argparse.Namespace) -> Tuple[List[str], List[str]]:
    baselines = parse_csv_list(args.baselines)
    models = parse_csv_list(args.models)

    if not baselines:
        raise ValueError("At least one baseline must be provided.")
    if not models:
        raise ValueError("At least one model must be provided.")

    if "all" in baselines:
        baselines = ["m1", "m2", "self_instruct", "wizardlm", "med_rag"]

    unknown = sorted(set(baselines) - SUPPORTED_BASELINES)
    if unknown:
        raise ValueError(f"Unsupported baselines: {unknown}. Supported: {sorted(SUPPORTED_BASELINES)}")

    if args.n <= 0:
        raise ValueError("--n must be positive.")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    if args.max_workers <= 0:
        raise ValueError("--max_workers must be positive.")

    return baselines, models


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)
    set_global_seed(args.seed)
    ensure_output_dir(Path(args.output_dir))

    baselines, models = validate_args(args)

    summary: Dict[str, Dict[str, int]] = {}

    for model_name in models:
        summary[model_name] = {}

        for baseline in baselines:
            logging.info("=" * 80)
            logging.info("Starting baseline=%s | model=%s", baseline, model_name)
            logging.info("=" * 80)

            try:
                count = run_one_baseline_for_model(
                    baseline=baseline,
                    model_name=model_name,
                    args=args,
                )
                summary[model_name][baseline] = count
            except Exception as exc:
                logging.exception(
                    "Failed baseline=%s | model=%s | error=%s",
                    baseline,
                    model_name,
                    repr(exc),
                )
                if not args.resume:
                    logging.info(
                        "Continuing to the next task. Use --resume to continue partially completed outputs."
                    )
                summary[model_name][baseline] = -1

    logging.info("=" * 80)
    logging.info("Generation summary")
    logging.info("=" * 80)

    for model_name, counts in summary.items():
        for baseline, count in counts.items():
            status = "FAILED" if count < 0 else str(count)
            logging.info("model=%s | baseline=%s | saved=%s", model_name, baseline, status)


if __name__ == "__main__":
    main()