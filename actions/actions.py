from typing import Any, Text, Dict, List
from datetime import datetime
from pathlib import Path
import csv
import json
import os

from rasa_sdk import Action, Tracker
from rasa_sdk.events import FollowupAction, SlotSet
from rasa_sdk.executor import CollectingDispatcher

# === PERBAIKAN IMPORT LOKAL (RAG & FAISS) ===
# Menggunakan tanda titik (.) karena filenya sekarang sudah satu folder di dalam 'actions'
from .search_faiss import search_by_intent
from .rag_engine import generate_answer_with_gemini_with_context

SLOTS_TO_RESET = [
    "jalur_seleksi_jenjang",
    "prodi",
    "jenis_beasiswa",
    "fakultas",
    "jenis_biaya",
    "metode_pembayaran",
    "unit_kontak",
    "active_intent",
]
# Mencari folder 'vectorstores' yang berada di dalam folder yang sama dengan file ini
VECTORSTORE_DIR = Path(__file__).parent / "vectorstores"

MAX_CONTEXT_TURNS = 3  # berapa turn sebelumnya yang dibawa ke RAG
EVENT_TAIL = 100       # batas event yang di-scan (hindari iterasi penuh)


def reset_context_slots():
    return [SlotSet(slot_name, None) for slot_name in SLOTS_TO_RESET]


def get_last_real_intent(tracker: Tracker) -> Text:
    ignored_intents = {"inform", "nlu_fallback"}

    for event in reversed(tracker.events):
        if event.get("event") == "user":
            intent_name = (
                event.get("parse_data", {})
                .get("intent", {})
                .get("name")
            )

            if intent_name and intent_name not in ignored_intents:
                return intent_name

    return tracker.latest_message.get("intent", {}).get("name", "")

def get_previous_real_intent(tracker: Tracker) -> Text:
    ignored_intents = {"inform", "nlu_fallback"}

    user_events = [
        event for event in tracker.events
        if event.get("event") == "user"
    ]

    # Abaikan pesan user yang sedang diproses sekarang
    previous_user_events = user_events[:-1]

    for event in reversed(previous_user_events):
        intent_name = (
            event.get("parse_data", {})
            .get("intent", {})
            .get("name")
        )

        if intent_name and intent_name not in ignored_intents:
            return intent_name

    return ""

def is_intent_switch(detected_intent, active_intent):
    ignored_intents = {"inform", "nlu_fallback"}

    return bool(
        detected_intent
        and detected_intent not in ignored_intents
        and active_intent
        and detected_intent != active_intent
    )

def text_value(value):
    if isinstance(value, list):
        return " ".join(str(v) for v in value)

    return str(value or "")


def normalize_text(value):
    if isinstance(value, list):
        value = " ".join(str(v) for v in value)

    return str(value or "").lower().strip()

def latest_entity_value(tracker, entity_name):
    entities = tracker.latest_message.get("entities", [])
    for ent in entities:
        if ent.get("entity") == entity_name:
            return ent.get("value")
    return None

def rerank_results(results, slots):
    reranked = []

    for result in results:
        base_score = float(result.get("score", 0.0))

        searchable_text = normalize_text(" ".join([
            text_value(result.get("topik", "")),
            text_value(result.get("alias_topik", "")),
            text_value(result.get("search_text", "")),
            text_value(result.get("parent_context", "")),
            text_value(result.get("heading", "")),
            text_value(result.get("content", "")),
        ]))

        bonus = 0.0
        penalty = 0.0

        prodi           = normalize_text(slots.get("prodi"))
        jalur_jenjang   = normalize_text(slots.get("jalur_seleksi_jenjang"))
        jenis_biaya     = normalize_text(slots.get("jenis_biaya"))
        jenis_beasiswa  = normalize_text(slots.get("jenis_beasiswa"))
        fakultas        = normalize_text(slots.get("fakultas"))
        metode          = normalize_text(slots.get("metode_pembayaran"))
        unit_kontak     = normalize_text(slots.get("unit_kontak"))

        if prodi and prodi in searchable_text:
            bonus += 0.18

        if jalur_jenjang and jalur_jenjang in searchable_text:
            bonus += 0.18

        if jenis_biaya and jenis_biaya in searchable_text:
            bonus += 0.15

        if jenis_beasiswa and jenis_beasiswa in searchable_text:
            bonus += 0.18

        if fakultas and fakultas in searchable_text:
            bonus += 0.10

        if metode and metode in searchable_text:
            bonus += 0.12

        if unit_kontak and unit_kontak in searchable_text:
            bonus += 0.18

        if jenis_biaya == "ukt" and "pendaftaran" in searchable_text:
            penalty += 0.35

        if jenis_biaya == "pendaftaran" and "ukt" in searchable_text:
            penalty += 0.35

        item = result.copy()
        item["rerank_bonus"] = bonus
        item["rerank_penalty"] = penalty
        item["rerank_score"] = max(0.0, base_score + bonus - penalty)
        reranked.append(item)

    return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)


class ActionHandleFallback(Action):
    def name(self) -> Text:
        return "action_handle_fallback"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        fallback_count = tracker.get_slot("fallback_count") or 0

        if fallback_count < 1:
            dispatcher.utter_message(response="utter_fallback")

            return [
                SlotSet("fallback_count", fallback_count + 1)
            ]

        return [
            SlotSet("fallback_count", 0),
            SlotSet("rag_search_mode", "global"),
            FollowupAction("action_rag_search")
        ]
    
def search_all_vector_stores(query, top_k=15, min_score=0.40):
    all_results = []

    for vectorstore_path in VECTORSTORE_DIR.iterdir():
        if not vectorstore_path.is_dir():
            continue

        intent_name = vectorstore_path.name

        results = search_by_intent(
            query=query,
            intent=intent_name,
            top_k=top_k,
            min_score=min_score,
        )

        for result in results:
            result["source_intent"] = intent_name

        all_results.extend(results)

    all_results = sorted(
        all_results,
        key=lambda x: x.get("score", 0),
        reverse=True
    )

    return all_results[:top_k]


class ActionTanyaBiaya(Action):
    def name(self):
        return "action_tanya_biaya"

    def run(self, dispatcher, tracker, domain):
        active_intent = tracker.get_slot("active_intent")
        user_text = (tracker.latest_message.get("text") or "").lower().strip()

        jenis_biaya = (
            latest_entity_value(tracker, "jenis_biaya")
            or tracker.get_slot("jenis_biaya")
        )

        prodi = (
            latest_entity_value(tracker, "prodi")
            or tracker.get_slot("prodi")
        )

        jalur = (
            latest_entity_value(tracker, "jalur_seleksi_jenjang")
            or tracker.get_slot("jalur_seleksi_jenjang")
        )

        jenis_biaya = (jenis_biaya or "").lower().strip()

        # Fallback keyword dari teks terbaru.
        # Ini penting supaya slot lama seperti "pendaftaran" bisa ketimpa
        # ketika user bertanya topik biaya baru seperti "matrikulasi".
        if "matrikulasi" in user_text or "martikulasi" in user_text:
            jenis_biaya = "matrikulasi"
        elif (
            "pendaftaran" in user_text
            or "daftar" in user_text
            or "formulir" in user_text
            or "seleksi" in user_text
        ):
            jenis_biaya = "pendaftaran"
        elif "ukt" in user_text or "uang kuliah tunggal" in user_text:
            jenis_biaya = "ukt"

        invalid_prodi = {"informasi", "prodi", "jurusan", "fakultas"}
        invalid_jalur = {"jalur", "seleksi"}

        if prodi and prodi.lower().strip() in invalid_prodi:
            prodi = None

        if jalur and jalur.lower().strip() in invalid_jalur:
            jalur = None

        # Recovery follow-up:
        # Kalau user sedang dalam konteks tanya_biaya dan hanya jawab prodi/jalur,
        # sistem bantu tebak jenis biaya dengan aturan ringan.
        if active_intent == "tanya_biaya" and not jenis_biaya:
            if prodi and not jalur:
                jenis_biaya = "ukt"
            elif jalur and not prodi:
                jenis_biaya = "pendaftaran"

        is_ukt = (
            "ukt" in jenis_biaya
            or "uang kuliah tunggal" in jenis_biaya
        )

        is_pendaftaran = (
            "pendaftaran" in jenis_biaya
            or "daftar" in jenis_biaya
            or "formulir" in jenis_biaya
        )

        is_matrikulasi = (
            "matrikulasi" in jenis_biaya
            or "martikulasi" in jenis_biaya
        )

        is_biaya_kuliah = jenis_biaya == "biaya kuliah"

        print("DEBUG BIAYA:")
        print("active_intent:", active_intent)
        print("user_text:", user_text)
        print("jenis_biaya:", jenis_biaya)
        print("jalur:", jalur)
        print("prodi:", prodi)

        if not jenis_biaya:
            dispatcher.utter_message(response="utter_klarifikasi_biaya_kuliah")
            return [
                SlotSet("active_intent", "tanya_biaya"),
            ]

        elif is_ukt and not prodi:
            dispatcher.utter_message(response="utter_minta_prodi_ukt")
            return [
                SlotSet("jenis_biaya", "ukt"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("prodi", prodi),
                SlotSet("active_intent", "tanya_biaya"),
            ]

        elif is_ukt and prodi:
            return [
                SlotSet("jenis_biaya", "ukt"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("prodi", prodi),
                SlotSet("active_intent", "tanya_biaya"),
                FollowupAction("action_rag_search"),
            ]

        elif is_pendaftaran and not jalur:
            dispatcher.utter_message(response="utter_minta_jalur_pendaftaran")
            return [
                SlotSet("jenis_biaya", "pendaftaran"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("prodi", prodi),
                SlotSet("active_intent", "tanya_biaya"),
            ]

        elif is_pendaftaran and jalur:
            return [
                SlotSet("jenis_biaya", "pendaftaran"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("prodi", prodi),
                SlotSet("active_intent", "tanya_biaya"),
                FollowupAction("action_rag_search"),
            ]

        elif is_matrikulasi:
            return [
                SlotSet("jenis_biaya", "matrikulasi"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("prodi", prodi),
                SlotSet("active_intent", "tanya_biaya"),
                FollowupAction("action_rag_search"),
            ]

        elif is_biaya_kuliah:
            dispatcher.utter_message(response="utter_klarifikasi_biaya_kuliah")
            return [
                SlotSet("active_intent", "tanya_biaya"),
            ]

        else:
            dispatcher.utter_message(response="utter_fallback_biaya")
            return [
                SlotSet("active_intent", "tanya_biaya"),
            ]
        
        
class ActionCaraBayar(Action):
    def name(self):
        return "action_cara_bayar"

    def _latest_entity_value(self, tracker, entity_name):
        entities = tracker.latest_message.get("entities", [])

        for ent in entities:
            if ent.get("entity") == entity_name:
                return ent.get("value")

        return None

    def run(self, dispatcher, tracker, domain):
        jenis_biaya_slot = tracker.get_slot("jenis_biaya")
        jalur_slot = tracker.get_slot("jalur_seleksi_jenjang")

        jenis_biaya_entity = self._latest_entity_value(tracker, "jenis_biaya")
        jalur_entity = self._latest_entity_value(
            tracker,
            "jalur_seleksi_jenjang"
        )

        jenis_biaya = (
            jenis_biaya_entity
            or jenis_biaya_slot
            or ""
        )

        jalur = jalur_entity or jalur_slot

        jenis_biaya = jenis_biaya.lower().strip()

        invalid_jalur = {"jalur", "seleksi"}

        if jalur and jalur.lower().strip() in invalid_jalur:
            jalur = None

        is_ukt = (
            "ukt" in jenis_biaya
            or "uang kuliah tunggal" in jenis_biaya
        )

        is_pendaftaran = (
            "pendaftaran" in jenis_biaya
            or "daftar" in jenis_biaya
            or "formulir" in jenis_biaya
        )

        print("DEBUG CARA BAYAR:")
        print("jenis_biaya_slot:", jenis_biaya_slot)
        print("jenis_biaya_entity:", jenis_biaya_entity)
        print("jenis_biaya_final:", jenis_biaya)
        print("jalur_slot:", jalur_slot)
        print("jalur_entity:", jalur_entity)
        print("jalur_final:", jalur)

        if not jenis_biaya:
            dispatcher.utter_message(response="utter_klarifikasi_cara_bayar")
            return [
                SlotSet("active_intent", "cara_bayar"),
            ]

        elif is_ukt:
            return [
                SlotSet("jenis_biaya", "ukt"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("active_intent", "cara_bayar"),
                FollowupAction("action_rag_search"),
            ]

        elif is_pendaftaran and not jalur:
            dispatcher.utter_message(response="utter_minta_jalur_cara_bayar")
            return [
                SlotSet("jenis_biaya", "pendaftaran"),
                SlotSet("active_intent", "cara_bayar"),
            ]

        elif is_pendaftaran and jalur:
            return [
                SlotSet("jenis_biaya", "pendaftaran"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("active_intent", "cara_bayar"),
                FollowupAction("action_rag_search"),
            ]

        else:
            dispatcher.utter_message(response="utter_klarifikasi_cara_bayar")
            return [
                SlotSet("active_intent", "cara_bayar"),
            ]


class ActionTanyaKuota(Action):
    def name(self):
        return "action_tanya_kuota"

    def _last_slot_value(self, tracker, slot_name):
        for event in reversed(tracker.events):
            if event.get("event") == "slot" and event.get("name") == slot_name:
                value = event.get("value")
                if value:
                    return value
        return None

    def _latest_entity_value(self, tracker, entity_name):
        entities = tracker.latest_message.get("entities", [])
        for ent in entities:
            if ent.get("entity") == entity_name:
                return ent.get("value")
        return None

    def run(self, dispatcher, tracker, domain):
        active_intent = tracker.get_slot("active_intent")

        jalur_slot = tracker.get_slot("jalur_seleksi_jenjang")
        prodi_slot = tracker.get_slot("prodi")

        jalur_entity = self._latest_entity_value(tracker, "jalur_seleksi_jenjang")
        prodi_entity = self._latest_entity_value(tracker, "prodi")

        jalur_history = self._last_slot_value(tracker, "jalur_seleksi_jenjang")
        prodi_history = self._last_slot_value(tracker, "prodi")

        # History hanya boleh dipakai kalau memang sedang dalam konteks tanya_kuota.
        # Kalau bukan, jangan ambil slot lama supaya tidak nyasar dari kontak/jadwal/biaya ke kuota.
        if active_intent == "tanya_kuota":
            jalur = jalur_entity or jalur_slot or jalur_history
            prodi = prodi_entity or prodi_slot or prodi_history
        else:
            jalur = jalur_entity or jalur_slot
            prodi = prodi_entity or prodi_slot

        invalid_prodi = {"informasi", "prodi", "jurusan", "fakultas"}
        invalid_jalur = {"jalur", "seleksi"}

        if prodi and prodi.lower().strip() in invalid_prodi:
            prodi = None

        if jalur and jalur.lower().strip() in invalid_jalur:
            jalur = None

        print("DEBUG KUOTA:")
        print("active_intent:", active_intent)
        print("jalur_slot:", jalur_slot)
        print("jalur_entity:", jalur_entity)
        print("jalur_history:", jalur_history)
        print("jalur_final:", jalur)
        print("prodi_slot:", prodi_slot)
        print("prodi_entity:", prodi_entity)
        print("prodi_history:", prodi_history)
        print("prodi_final:", prodi)

        if not jalur and not prodi:
            dispatcher.utter_message(
                text=(
                    "Kak, ingin mengetahui kuota/daya tampung untuk jalur dan prodi apa ya?\n"
                    "Misalnya: kuota Kedokteran jalur SNBP atau kuota Farmasi jalur Mandiri."
                )
            )
            return [
                SlotSet("active_intent", "tanya_kuota"),
                SlotSet("jalur_seleksi_jenjang", None),
                SlotSet("prodi", None),
            ]

        elif jalur and not prodi:
            dispatcher.utter_message(
                text="Kak, boleh sebutkan prodi yang ingin dicek kuotanya?"
            )
            return [
                SlotSet("active_intent", "tanya_kuota"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("prodi", None),
            ]

        elif prodi and not jalur:
            dispatcher.utter_message(
                text="Kak, kuota yang dimaksud untuk jalur apa ya? SNBP, SNBT, SPAN-PTKIN, UM-PTKIN, atau Mandiri?"
            )
            return [
                SlotSet("active_intent", "tanya_kuota"),
                SlotSet("prodi", prodi),
                SlotSet("jalur_seleksi_jenjang", None),
            ]

        elif jalur and prodi:
            return [
                SlotSet("active_intent", "tanya_kuota"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("prodi", prodi),
                FollowupAction("action_rag_search"),
            ]


class ActionTanyaPersyaratan(Action):
    def name(self):
        return "action_tanya_persyaratan"

    def _latest_entity_value(self, tracker, entity_name):
        entities = tracker.latest_message.get("entities", [])

        for ent in entities:
            if ent.get("entity") == entity_name:
                return ent.get("value")

        return None

    def run(self, dispatcher, tracker, domain):
        active_intent = tracker.get_slot("active_intent")

        jalur_slot = tracker.get_slot("jalur_seleksi_jenjang")
        jenis_beasiswa_slot = tracker.get_slot("jenis_beasiswa")

        jalur_entity = self._latest_entity_value(
            tracker,
            "jalur_seleksi_jenjang"
        )

        jenis_beasiswa_entity = self._latest_entity_value(
            tracker,
            "jenis_beasiswa"
        )

        # prioritaskan entity terbaru
        jalur = jalur_entity or jalur_slot
        jenis_beasiswa = (
            jenis_beasiswa_entity
            or jenis_beasiswa_slot
        )

        invalid_jalur = {"jalur", "seleksi"}
        invalid_beasiswa = {"beasiswa"}

        if jalur and jalur.lower().strip() in invalid_jalur:
            jalur = None

        if (
            jenis_beasiswa
            and jenis_beasiswa.lower().strip() in invalid_beasiswa
        ):
            jenis_beasiswa = None

        print("DEBUG PERSYARATAN:")
        print("active_intent:", active_intent)
        print("jalur:", jalur)
        print("jenis_beasiswa:", jenis_beasiswa)

        # kalau ada konteks → langsung RAG
        if jalur or jenis_beasiswa:
            return [
                SlotSet("active_intent", "tanya_persyaratan"),
                SlotSet("jalur_seleksi_jenjang", jalur),
                SlotSet("jenis_beasiswa", jenis_beasiswa),
                FollowupAction("action_rag_search"),
            ]

        dispatcher.utter_message(
            response="utter_minta_konteks_persyaratan"
        )

        return [
            SlotSet("active_intent", "tanya_persyaratan"),
        ]


class ActionHandleInform(Action):
    def name(self):
        return "action_handle_inform"

    def run(self, dispatcher, tracker, domain):
        active_intent = tracker.get_slot("active_intent")
        fallback_count = tracker.get_slot("fallback_count") or 0

        if fallback_count >= 1:
            return [
                SlotSet("fallback_count", 0),
                SlotSet("rag_search_mode", "global"),
                FollowupAction("action_rag_search"),
            ]

        intent_action_map = {
            "tanya_kontak": "action_rag_search",
            "tanya_jadwal": "action_rag_search",
            "daftar_ulang": "action_rag_search",
            "tanya_prosedur_pendaftaran": "action_rag_search",
            "tanya_ujian": "action_rag_search",
            "tanya_beasiswa": "action_rag_search",
            "tanya_biaya": "action_tanya_biaya",
            "cara_bayar": "action_cara_bayar",
            "tanya_kuota": "action_tanya_kuota",
            "tanya_persyaratan": "action_tanya_persyaratan",
            "tanya_prodi_fakultas": "action_rag_search",
        }

        print("DEBUG HANDLE INFORM:")
        print("active_intent:", active_intent)
        print("latest_text:", tracker.latest_message.get("text"))
        print("entities:", tracker.latest_message.get("entities", []))

        if active_intent in intent_action_map:
            return [FollowupAction(intent_action_map[active_intent])]

        dispatcher.utter_message(response="utter_fallback")
        return []

# ── Helper functions (di luar class, Rasa-friendly) ───────────────────────────

def resolve_intent(detected, active, previous=None):
    ignored_intents = {"inform", "nlu_fallback"}

    if detected in ignored_intents:
        return active or previous

    if detected:
        return detected

    return active or previous

def extract_entities(tracker, inherit_slots):
    # inherit_slots=True  → slot lama ikut (percakapan lanjut)
    # inherit_slots=False → slot lama diabaikan (topic switch)
    entity_keys = [
        "prodi",
        "jalur_seleksi_jenjang",
        "jenis_biaya",
        "jenis_beasiswa",
        "fakultas",
        "metode_pembayaran",
    ]
    result = {}
    for key in entity_keys:
        from_message = latest_entity_value(tracker, key)
        if from_message:
            result[key] = from_message
        elif inherit_slots:
            result[key] = tracker.get_slot(key)
        else:
            result[key] = None  # slot lama sengaja dikosongkan
    return result


def build_enriched_query(user_query, intent, entities, context_turns, tracker=None):
    parts = []

    # Riwayat percakapan
    for i, turn in enumerate(context_turns, start=1):
        parts.append(f"Konteks percakapan [{i}]: {turn}")

    # Pertanyaan user
    if user_query:
        parts.append(f"Pertanyaan user: {user_query}")

    # Intent
    if intent:
        parts.append(f"Intent: {intent}")

    # Inject slot helper manual
    if tracker:
        entities["unit_kontak"] = tracker.get_slot("unit_kontak")

    label_map = {
        "jenis_biaya":           "Jenis biaya",
        "jenis_beasiswa":        "Jenis beasiswa",
        "jalur_seleksi_jenjang": "Jalur seleksi / jenjang",
        "prodi":                 "Program studi",
        "fakultas":              "Fakultas",
        "metode_pembayaran":     "Metode pembayaran",
        "unit_kontak":           "Unit kontak",
    }

    for key, label in label_map.items():
        val = entities.get(key)
        if val:
            parts.append(f"{label}: {val}")

    return "\n".join(parts).strip()


def build_context(top_results):
    parts = []
    for i, r in enumerate(top_results, start=1):
        topik = (
            r.get("topik") or r.get("topic") or r.get("title")
            or r.get("intent") or "Informasi"
        )
        content = (
            r.get("content") or r.get("text") or r.get("answer")
            or r.get("document")
            or r.get("metadata", {}).get("content")
            or r.get("metadata", {}).get("text")
            or ""
        )
        if str(content).strip():
            parts.append(f"# {i}. {topik}\n{str(content).strip()}")
    return "\n\n".join(parts).strip()


def make_slot_events(intent, entities, rag_search_mode):
    return [
        SlotSet("active_intent",         intent),
        SlotSet("rag_search_mode",       rag_search_mode),
        SlotSet("prodi",                 entities.get("prodi")),
        SlotSet("jalur_seleksi_jenjang", entities.get("jalur_seleksi_jenjang")),
        SlotSet("jenis_biaya",           entities.get("jenis_biaya")),
        SlotSet("jenis_beasiswa",        entities.get("jenis_beasiswa")),
        SlotSet("fakultas",              entities.get("fakultas")),
        SlotSet("metode_pembayaran",     entities.get("metode_pembayaran")),
        SlotSet("unit_kontak",           entities.get("unit_kontak")),
    ]

RAGAS_LOG_PATH = "ragas_eval_log.csv"


def top_results_to_contexts(top_results):
    contexts = []

    for r in top_results:
        text = (
            r.get("text")
            or r.get("content")
            or r.get("page_content")
            or r.get("dokumen")
            or r.get("jawaban")
            or ""
        )

        if text:
            contexts.append(text)

    return contexts


def clean_text(value):
    """Biar CSV tidak berantakan karena enter, koma, atau None."""
    if value is None:
        return ""
    return str(value).replace("\r", " ").replace("\n", " ").strip()


def append_ragas_csv(question, answer, contexts, ground_truth="", filename="ragas_eval_log.csv"):
    file_exists = os.path.exists(filename)

    # contexts harus aman untuk CSV
    if isinstance(contexts, list):
        contexts = json.dumps(contexts, ensure_ascii=False)
    else:
        contexts = clean_text(contexts)

    row = {
        "question": clean_text(question),
        "answer": clean_text(answer),
        "contexts": clean_text(contexts),
        "ground_truth": clean_text(ground_truth),
    }

    with open(filename, mode="a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["question", "answer", "contexts", "ground_truth"],
            quoting=csv.QUOTE_ALL
        )

        if not file_exists or os.path.getsize(filename) == 0:
            writer.writeheader()

        writer.writerow(row)

def get_previous_action_name(tracker: Tracker) -> Text:
    for event in reversed(tracker.events):
        if event.get("event") == "action":
            action_name = event.get("name")

            if action_name not in {
                "action_listen",
                "action_extract_slots",
            }:
                return action_name

    return ""

# ── Action class ──────────────────────────────────────────────────────────────

class ActionRagSearch(Action):
    def name(self) -> Text:
        return "action_rag_search"

    def run(self, dispatcher, tracker, domain):
        user_query = tracker.latest_message.get("text", "")
        rag_search_mode = tracker.get_slot("rag_search_mode")

        # 1. Tentukan intent
        # 1. Tentukan intent
        detected_intent = tracker.latest_message.get("intent", {}).get("name")
        active_intent = tracker.get_slot("active_intent")
        previous_intent = get_previous_real_intent(tracker)
        previous_action = get_previous_action_name(tracker)

        form_intent_map = {
            "form_tanya_jadwal": "tanya_jadwal",
            "form_daftar_ulang": "daftar_ulang",
            "form_tanya_prosedur": "tanya_prosedur_pendaftaran",
            "form_tanya_ujian": "tanya_ujian",
            "form_tanya_beasiswa": "tanya_beasiswa",
            "form_tanya_kontak": "tanya_kontak",
        }

        forced_intent = form_intent_map.get(previous_action)

        if forced_intent:
            intent = forced_intent
            is_topic_switch = False
        else:
            intent = resolve_intent(
                detected=detected_intent,
                active=active_intent,
                previous=previous_intent,
            )

            is_topic_switch = is_intent_switch(
                detected_intent=detected_intent,
                active_intent=active_intent,
            )
        # 2. Ekstrak entitas
        # Kalau topic switch, slot lama tidak diwarisi supaya tidak kontaminasi topik baru
        entities = extract_entities(
            tracker,
            inherit_slots=not is_topic_switch
        )

        if intent == "tanya_kontak":
            entities["unit_kontak"] = (
                tracker.get_slot("unit_kontak")
                or entities.get("prodi")
                or entities.get("fakultas")
                or entities.get("jalur_seleksi_jenjang")
            )
        else:
            entities["unit_kontak"] = tracker.get_slot("unit_kontak")

        # 3. Ambil riwayat percakapan (rolling window, MAX_CONTEXT_TURNS turn terakhir)
        recent_tail    = tracker.events[-EVENT_TAIL:]
        all_user_texts = [
            e["text"]
            for e in recent_tail
            if e.get("event") == "user" and e.get("text")
        ]
        # all_user_texts[-1] adalah pesan saat ini, jadi kita ambil sebelumnya
        context_turns = all_user_texts[-(MAX_CONTEXT_TURNS + 1):-1]

        # 4. Bangun enriched query
        # Kalau topic switch, context_turns dikosongkan supaya konteks lama tidak ikut
        enriched_query = build_enriched_query(
            user_query=user_query,
            intent=intent,
            entities=entities,
            context_turns=context_turns if not is_topic_switch else [],
            tracker=tracker,
        )

        # 5. Debug logs
        print("=" * 60)
        print("LATEST TEXT      :", user_query)
        print("DETECTED INTENT  :", detected_intent)
        print("ACTIVE INTENT    :", active_intent)
        print("FINAL INTENT     :", intent)
        print("IS TOPIC SWITCH  :", is_topic_switch)
        print("ENTITIES         :", entities)
        print("CONTEXT TURNS    :", context_turns)
        print("RAG QUERY:\n", enriched_query)
        print("=" * 60)

        if not enriched_query or (rag_search_mode != "global" and not intent):
            dispatcher.utter_message(
                text="Mohon maaf, saya belum dapat memahami pertanyaan Anda dengan baik 🙏"
            )
            return []

        # 6. Vector search
        print("RAG SEARCH MODE:", rag_search_mode)

        if rag_search_mode == "global":
            results = search_all_vector_stores(
                query=enriched_query,
                top_k=15,
                min_score=0.30,
            )
        else:
            results = search_by_intent(
                query=enriched_query,
                intent=intent,
                top_k=15,
                min_score=0.30,
            )

        if not results:
            if rag_search_mode == "global":
                dispatcher.utter_message(response="utter_hubungi_cs")
            else:
                dispatcher.utter_message(
                    text=(
                        "Mohon maaf 🙏 saya belum menemukan informasi yang sesuai. "
                        "Bisa dituliskan kembali dengan kalimat yang lebih spesifik?"
                    )
                )
            return make_slot_events(intent, entities, rag_search_mode=None)

        # 7. Rerank & ambil top 5
        results     = rerank_results(results, entities)
        top_results = results[:5]

        print("DEBUG RAG RESULTS:")
        for r in results:
            print(
                "SCORE:", r.get("score"),
                "| RERANK:", r.get("rerank_score"),
                "| BONUS:", r.get("rerank_bonus"),
                "| PENALTY:", r.get("rerank_penalty"),
                "| TOPIK:", r.get("topik"),
                "| ALIAS:", r.get("alias_topik"),
            )

        context = build_context(top_results)

        if not context:
            dispatcher.utter_message(
                text=(
                    "Mohon maaf 🙏 saya belum menemukan informasi yang sesuai. "
                    "Bisa dituliskan kembali dengan kalimat yang lebih spesifik?"
                )
            )
            return make_slot_events(intent, entities, rag_search_mode=None)

        # 8. Generate jawaban via Gemini
        answer = generate_answer_with_gemini_with_context(
            query=enriched_query,
            context=context,
            conversation_history=context_turns,
        )

        # Simpan data untuk evaluasi RAGAS
        ragas_contexts = top_results_to_contexts(top_results)

        append_ragas_csv(
            question=user_query,
            answer=answer,
            contexts=ragas_contexts,
            ground_truth="",  # nanti isi manual jawaban idealnya
        )

        print("ANSWER SENT TO USER:\n", answer)
        dispatcher.utter_message(text=answer)

        return make_slot_events(intent, entities, rag_search_mode=None)
