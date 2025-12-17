import os
import json
import textwrap
import time
import urllib.parse

import requests
import pandas as pd
from bs4 import BeautifulSoup

from neo4j import GraphDatabase

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


# ==================================================
# 1. KONFIGURASI API KEY & NEO4J
# ==================================================

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-YOUR-DEEPSEEK-KEY"):
    print("⚠️ Peringatan: DEEPSEEK_API_KEY belum diisi dengan benar.")

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.0,
)

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "12345678")

# driver global supaya bisa dipakai tool agent ke-2, 4, dan 5
driver = None

CSV_ENRICHED_PATH = "anggota_dpr_enriched.csv"
CSV_RAW_PATH = "anggota_dpr.csv"


# ==================================================
# 2. UTIL: BANGUN URL WIKIPEDIA & SCRAPE TEKS
# ==================================================

def build_wikipedia_url_from_name(name: str) -> str:
    title = name.replace(" ", "_")
    encoded_title = urllib.parse.quote(title)
    return f"https://id.wikipedia.org/wiki/{encoded_title}"


def fetch_wikipedia_text_with_infobox(url: str) -> str:
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; RafyBot/1.0; +https://example.com/bot)"
    }
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Infobox
    infobox_text = ""
    infobox = soup.find("table", class_="infobox")
    if infobox:
        rows = infobox.find_all("tr")
        lines = []
        for row in rows:
            header = row.find("th")
            value = row.find("td")
            if header and value:
                h = header.get_text(" ", strip=True)
                v = value.get_text(" ", strip=True)
                lines.append(f"{h}: {v}")
        infobox_text = "\n".join(lines)

    # Paragraf
    content_div = soup.find("div", id="mw-content-text")
    article_text = ""
    if content_div:
        paragraphs = content_div.find_all("p")
        text_chunks = [p.get_text(" ", strip=True) for p in paragraphs]
        article_text = "\n".join(ch for ch in text_chunks if ch)
    else:
        article_text = soup.get_text(separator="\n", strip=True)

    combined_parts = []
    if infobox_text:
        combined_parts.append("INFORMASI PRIBADI (INFOBOX)\n" + infobox_text)
    if article_text:
        combined_parts.append("TEKS ARTIKEL\n" + article_text)

    combined_text = "\n".join(combined_parts)
    return combined_text


# ==================================================
# 3. TOOL UNTUK AGENT 1: get_wikipedia_biography
# ==================================================

@tool
def get_wikipedia_biography(name: str) -> str:
    """
    Tool Agent 1: ambil teks biografi Wikipedia (infobox + artikel).
    Output: "SOURCE_URL::<url>\\n\\n<teks>"
    """
    url = build_wikipedia_url_from_name(name)
    text = fetch_wikipedia_text_with_infobox(url)
    if not text.strip():
        raise ValueError(f"Tidak ada teks Wikipedia untuk {name} di URL: {url}")
    return f"SOURCE_URL::{url}\n\n{text}"


# ==================================================
# 4. AGENT 1: FAMILY EXTRACTION AGENT
# ==================================================

SYSTEM_PROMPT_A1 = """
Kamu adalah asisten ekstraksi informasi yang sangat teliti.

Tugasmu:
- Untuk tokoh dengan nama tertentu, kamu HARUS memanggil tool `get_wikipedia_biography`
  dengan argumen `name` yang sama dengan nama tokoh tersebut.
- Tool akan mengembalikan teks biografi (infobox + artikel) dengan format:
  
  SOURCE_URL::<url_wikipedia>

  <teks biografi>

- Dari teks tersebut, ekstrak *relasi keluarga*.

Relasi yang dicari misalnya:
- suami / istri / pasangan
- anak (putra/putri)
- menantu
- orang tua (ayah, ibu)
- saudara kandung, cucu, mertua jika ada

PERHATIKAN:
- Gunakan HANYA informasi yang ada di teks yang dikembalikan tool.
- Jangan mengarang informasi di luar teks.
- Ambil nilai URL dari baris yang diawali "SOURCE_URL::" sebagai `source_url`.

Jawab SELALU dalam FORMAT JSON murni, tanpa teks lain, dengan struktur:

{
  "person": "Nama tokoh",
  "source_url": "https://id.wikipedia.org/...",
  "families": [
    {
      "relation": "istri",
      "name": "Nama Lengkap",
      "note": "keterangan tambahan jika ada (opsional)"
    }
  ]
}

Jika tidak ditemukan informasi keluarga di teks, berikan:

{
  "person": "<nama tokoh>",
  "source_url": "<url wikipedia>",
  "families": []
}

JANGAN menambahkan komentar lain di luar JSON.
""".strip()

family_agent = create_react_agent(
    llm,
    tools=[get_wikipedia_biography],
    prompt=SYSTEM_PROMPT_A1,
    name="family_extraction_agent",
)


def run_family_agent(person_name: str) -> dict:
    user_prompt = textwrap.dedent(f"""
    Ekstrak relasi keluarga untuk tokoh bernama: "{person_name}".

    Langkah yang HARUS kamu lakukan:
    1. Panggil tool `get_wikipedia_biography` dengan argumen `name="{person_name}"`.
    2. Baca teks biografi yang dikembalikan tool (infobox + artikel).
    3. Berdasarkan teks tersebut, bentuk JSON sesuai format yang sudah dijelaskan
       di sistem prompt, TANPA teks tambahan di luar JSON.
    """)

    state = family_agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    messages = state["messages"]
    last_msg = messages[-1]
    content = last_msg.content

    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        content_str = "\n".join(parts)
    else:
        content_str = str(content)

    content_str = content_str.strip()

    try:
        data = json.loads(content_str)
        return data
    except json.JSONDecodeError:
        first_brace = content_str.find("{")
        last_brace = content_str.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_sub = content_str[first_brace:last_brace + 1]
            data = json.loads(json_sub)
            return data
        else:
            raise RuntimeError("Output agent 1 bukan JSON valid:\n" + content_str)


# ==================================================
# 5. FUNGSI DASAR TULIS KE NEO4J
# ==================================================

def write_family_to_neo4j(neo4j_driver, person_name, families, source_url=None):
    """
    Buat node Person dan relasi keluarga ke Neo4j.
    """
    if not families:
        return

    relation_mapping = {
        "suami": ("SPOUSE_OF", "undirected"),
        "istri": ("SPOUSE_OF", "undirected"),
        "pasangan": ("SPOUSE_OF", "undirected"),
        "suami/istri": ("SPOUSE_OF", "undirected"),

        "anak": ("PARENT_OF", "outgoing"),      # person -> child
        "putra": ("PARENT_OF", "outgoing"),
        "putri": ("PARENT_OF", "outgoing"),

        "ayah": ("PARENT_OF", "incoming"),      # parent -> person
        "ibu": ("PARENT_OF", "incoming"),
        "orang tua": ("PARENT_OF", "incoming"),

        "saudara": ("SIBLING_OF", "undirected"),
        "saudara kandung": ("SIBLING_OF", "undirected"),

        "menantu": ("IN_LAW_OF", "undirected"),
        "mertua": ("IN_LAW_OF", "undirected"),
        "cucu": ("FAMILY_OF", "undirected"),
    }

    with neo4j_driver.session() as session:
        session.run(
            """
            MERGE (p:Person {name: $name})
            ON CREATE SET p.created_at = timestamp()
            SET p.last_seen = timestamp()
            """,
            name=person_name,
        )

        for fam in families:
            rel_raw = (fam.get("relation") or "").strip().lower()
            rel_name = (fam.get("name") or "").strip()
            note = (fam.get("note") or "").strip()

            if not rel_name:
                continue

            rel_type, direction = relation_mapping.get(rel_raw, ("FAMILY_OF", "undirected"))

            session.run(
                """
                MERGE (f:Person {name: $rel_name})
                ON CREATE SET f.created_at = timestamp()
                SET f.last_seen = timestamp()
                """,
                rel_name=rel_name,
            )

            props = {
                "relation_label": rel_raw,
                "note": note if note else None,
                "source": source_url,
                "created_at": int(time.time() * 1000),
            }

            if direction == "outgoing":
                cypher_rel = f"""
                MATCH (p:Person {{name: $person_name}}),
                      (f:Person {{name: $rel_name}})
                MERGE (p)-[r:{rel_type}]->(f)
                SET r.relation_label = $relation_label,
                    r.note = $note,
                    r.source = $source,
                    r.last_seen = $created_at
                """
            elif direction == "incoming":
                cypher_rel = f"""
                MATCH (p:Person {{name: $person_name}}),
                      (f:Person {{name: $rel_name}})
                MERGE (f)-[r:{rel_type}]->(p)
                SET r.relation_label = $relation_label,
                    r.note = $note,
                    r.source = $source,
                    r.last_seen = $created_at
                """
            else:  # undirected -> simpan p -> f
                cypher_rel = f"""
                MATCH (p:Person {{name: $person_name}}),
                      (f:Person {{name: $rel_name}})
                MERGE (p)-[r:{rel_type}]->(f)
                SET r.relation_label = $relation_label,
                    r.note = $note,
                    r.source = $source,
                    r.last_seen = $created_at
                """

            session.run(
                cypher_rel,
                person_name=person_name,
                rel_name=rel_name,
                **props,
            )


# ==================================================
# 6. TOOL UNTUK AGENT 2: store_family_in_neo4j
# ==================================================

@tool
def store_family_in_neo4j(person: str, families: list, source_url: str = None) -> str:
    """
    Tool Agent 2: simpan satu orang & relasi keluarganya ke Neo4j.
    """
    global driver
    if driver is None:
        raise RuntimeError("Neo4j driver belum diinisialisasi.")

    write_family_to_neo4j(driver, person, families, source_url)
    return f"Stored {len(families)} relations for {person} in Neo4j"


# ==================================================
# 7. AGENT 2: KG BUILDER AGENT
# ==================================================

SYSTEM_PROMPT_A2 = """
Kamu adalah agen pembangun knowledge graph (Neo4j).

Tugasmu:
- Menerima JSON hasil ekstraksi keluarga dengan struktur:
  {
    "person": "...",
    "source_url": "...",
    "families": [
      {"relation": "...", "name": "...", "note": "..."},
      ...
    ]
  }

Instruksi:
- JANGAN mengubah isi JSON (nama, relasi, dll.).
- PANGGIL tool `store_family_in_neo4j` tepat SATU kali dengan:
  - person = field "person"
  - families = field "families"
  - source_url = field "source_url"

Setelah tool dipanggil:
- Balas lagi dengan JSON ASLI yang sama persis seperti input.
- Jangan tambah komentar atau teks lain di luar JSON.
""".strip()

kg_agent = create_react_agent(
    llm,
    tools=[store_family_in_neo4j],
    prompt=SYSTEM_PROMPT_A2,
    name="kg_builder_agent",
)


def run_kg_agent(result_json: dict) -> dict:
    json_str = json.dumps(result_json, ensure_ascii=False)
    user_prompt = (
        "Berikut JSON hasil ekstraksi keluarga:\n"
        f"{json_str}\n\n"
        "Gunakan JSON ini persis seperti instruksi di sistem prompt."
    )

    state = kg_agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    messages = state["messages"]
    last_msg = messages[-1]
    content = last_msg.content

    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        content_str = "\n".join(parts)
    else:
        content_str = str(content)

    content_str = content_str.strip()

    try:
        data = json.loads(content_str)
        return data
    except json.JSONDecodeError:
        first_brace = content_str.find("{")
        last_brace = content_str.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_sub = content_str[first_brace:last_brace + 1]
            data = json.loads(json_sub)
            return data
        else:
            raise RuntimeError("Output agent 2 bukan JSON valid:\n" + content_str)


# ==================================================
# 8. PROSES CSV UNTUK AGENT 1 + 2
# ==================================================

def process_csv_with_agents_1_2(csv_path: str, max_rows: int = 10):
    df = pd.read_csv(csv_path)

    for col in ["Pasangan", "Keluarga"]:
        if col not in df.columns:
            df[col] = pd.Series([""] * len(df), dtype="string")
        else:
            df[col] = df[col].astype("string")

    n = min(max_rows, len(df))
    print(f"📄 Membaca {n} baris pertama dari: {csv_path}\n")

    for idx, row in df.head(n).iterrows():
        nama = str(row.get("Nama", "")).strip()
        if not nama:
            print(f"Baris {idx}: kolom 'Nama' kosong, dilewati.")
            continue

        print(f"=== [{idx}] Memproses: {nama} ===")

        # Agent 1: ekstraksi keluarga
        try:
            result = run_family_agent(nama)
        except Exception as e:
            print(f"  ❌ Error dari agent 1 (ekstraksi) untuk {nama}: {e}")
            continue

        # Agent 2: simpan ke Neo4j
        try:
            result_kg = run_kg_agent(result)
            print("  🟢 Relasi keluarga ditulis ke Neo4j via agent kedua.")
        except Exception as e:
            print(f"  ⚠️ Gagal tulis ke Neo4j via agent 2 untuk {nama}: {e}")
            result_kg = result  # tetap pakai hasil ekstraksi untuk CSV

        families = result_kg.get("families", [])
        if not isinstance(families, list):
            print(f"  ⚠️ Format 'families' tidak list untuk {nama}, dilewati.")
            continue

        if not families:
            print(f"  ℹ️ Tidak ada keluarga ditemukan untuk {nama}")
            continue

        print(f"  ✅ Ditemukan {len(families)} relasi keluarga")

        existing_pasangan = row.get("Pasangan", "")
        existing_keluarga = row.get("Keluarga", "")

        if pd.isna(existing_pasangan):
            existing_pasangan = ""
        if pd.isna(existing_keluarga):
            existing_keluarga = ""

        pasangan_list = []
        keluarga_list = []

        if existing_pasangan.strip():
            pasangan_list.append(existing_pasangan.strip())
        if existing_keluarga.strip():
            keluarga_list.append(existing_keluarga.strip())

        for fam in families:
            rel = (fam.get("relation") or "").strip().lower()
            obj_name = (fam.get("name") or "").strip()
            note = (fam.get("note") or "").strip()

            if not obj_name:
                continue

            if note:
                label = f"{obj_name} ({rel}, {note})"
            else:
                label = f"{obj_name} ({rel})"

            if rel in ["suami", "istri", "pasangan", "suami/istri"]:
                if label not in pasangan_list:
                    pasangan_list.append(label)
            else:
                if label not in keluarga_list:
                    keluarga_list.append(label)

        df.at[idx, "Pasangan"] = "; ".join(pasangan_list)
        df.at[idx, "Keluarga"] = "; ".join(keluarga_list)

        time.sleep(1)

    out_csv = CSV_ENRICHED_PATH
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 File hasil CSV disimpan ke: {out_csv}")


# ==================================================
# 9. TOOL UNTUK AGENT 3: RINGKASAN STRATEGIC MARRIAGE
# ==================================================

@tool
def get_strategic_marriage_summary(csv_path: str, max_rows: int = 1000) -> str:
    """
    Tool Agent 3:
    - Baca anggota_dpr_enriched.csv
    - Bangun relasi (Nama, Dapil, Partai, Jabatan, Pendidikan, Pasangan, Keluarga)
    - Deteksi calon "pernikahan politik" (cross-family, cross-party)
    - Return JSON string ringkasan.
    """
    df = pd.read_csv(csv_path)
    if max_rows < len(df):
        df = df.head(max_rows)

    # Pastikan kolom-kolom kunci ada
    for col in ["Nama", "Dapil", "Partai", "Jabatan", "Pendidikan", "Pasangan", "Keluarga"]:
        if col not in df.columns:
            df[col] = ""

    # Map nama -> info
    persons = {}
    for _, row in df.iterrows():
        nama = str(row["Nama"]).strip()
        if not nama:
            continue
        keluarga_label = nama.split()[-1] if nama.split() else nama
        persons[nama] = {
            "nama": nama,
            "dapil": str(row["Dapil"]).strip(),
            "partai": str(row["Partai"]).strip(),
            "jabatan": str(row["Jabatan"]).strip(),
            "pendidikan": str(row["Pendidikan"]).strip(),
            "keluarga_label": keluarga_label,
        }

    marriages = []
    cross_party_counts = {}
    cross_family_counts = {}

    for _, row in df.iterrows():
        nama = str(row["Nama"]).strip()
        if not nama:
            continue

        info = persons.get(nama, {})
        person_family = info.get("keluarga_label", "")
        person_party = info.get("partai", "")
        person_dapil = info.get("dapil", "")

        pasangan_raw = str(row["Pasangan"]) if not pd.isna(row["Pasangan"]) else ""
        if not pasangan_raw.strip():
            continue

        pasangan_entries = [e.strip() for e in pasangan_raw.split(";") if e.strip()]
        for entry in pasangan_entries:
            # Format: "Nama Pasangan (relation, note)" atau "Nama Pasangan"
            spouse_name = entry
            relation_label = ""
            if "(" in entry:
                name_part, rel_part = entry.split("(", 1)
                spouse_name = name_part.strip()
                rel_part = rel_part.rstrip(")")
                relation_label = rel_part.split(",")[0].strip()

            spouse_info = persons.get(spouse_name, None)
            spouse_party = spouse_info.get("partai", "") if spouse_info else ""
            spouse_dapil = spouse_info.get("dapil", "") if spouse_info else ""
            spouse_family = (
                spouse_info.get("keluarga_label")
                if spouse_info
                else (spouse_name.split()[-1] if spouse_name.split() else spouse_name)
            )

            cross_family = (
                person_family.lower() != spouse_family.lower()
                if person_family and spouse_family
                else False
            )

            cross_party = False
            if person_party and spouse_party and person_party != spouse_party:
                cross_party = True

            if cross_party:
                key = tuple(sorted([person_party, spouse_party]))
                cross_party_counts[key] = cross_party_counts.get(key, 0) + 1

            if cross_family:
                key_f = tuple(sorted([person_family, spouse_family]))
                cross_family_counts[key_f] = cross_family_counts.get(key_f, 0) + 1

            marriages.append(
                {
                    "person": nama,
                    "person_family": person_family,
                    "person_party": person_party,
                    "person_dapil": person_dapil,
                    "spouse": spouse_name,
                    "spouse_family": spouse_family,
                    "spouse_party": spouse_party,
                    "spouse_dapil": spouse_dapil,
                    "relation_label": relation_label,
                    "cross_family": cross_family,
                    "cross_party": cross_party,
                }
            )

    # Deteksi orang yang menikah ke lebih dari satu keluarga berbeda
    multi_family_bridge = []
    person_to_families = {}
    for m in marriages:
        fam = m["spouse_family"]
        if not fam:
            continue
        person_to_families.setdefault(m["person"], set()).add(fam)

    for person, fams in person_to_families.items():
        if len(fams) > 1:
            multi_family_bridge.append(
                {
                    "person": person,
                    "families": list(fams),
                    "partai": persons.get(person, {}).get("partai", ""),
                    "dapil": persons.get(person, {}).get("dapil", ""),
                }
            )

    summary = {
        "total_persons": len(persons),
        "total_marriages": len(marriages),
        "total_cross_family_marriages": sum(1 for m in marriages if m["cross_family"]),
        "total_cross_party_marriages": sum(1 for m in marriages if m["cross_party"]),
        "marriages": marriages,
        "cross_party_pairs": [
            {"partai_a": k[0], "partai_b": k[1], "count": v}
            for k, v in sorted(cross_party_counts.items(), key=lambda x: -x[1])
        ],
        "cross_family_pairs": [
            {"keluarga_a": k[0], "keluarga_b": k[1], "count": v}
            for k, v in sorted(cross_family_counts.items(), key=lambda x: -x[1])
        ],
        "multi_family_bridge_persons": multi_family_bridge,
    }

    return json.dumps(summary, ensure_ascii=False)


# ==================================================
# 10. AGENT 3: STRATEGIC MARRIAGE ANALYSIS
# ==================================================

SYSTEM_PROMPT_A3 = """
Kamu adalah analis politik yang fokus pada konsep "Strategic Marriage" (pernikahan politik).

Konsep:
- Strategic marriage = pernikahan antara dua keluarga politik yang berbeda,
  terutama jika:
  - kedua belah pihak aktif di politik (anggota DPR, elit partai, dsb.), atau
  - pernikahan menghubungkan dua partai berbeda, dua dapil/daerah kekuasaan,
    atau dua "klan" keluarga yang kuat.

Data yang akan kamu terima dari tool:
- JSON ringkasan berisi:
  - total_persons, total_marriages
  - marriages[]: list relasi pernikahan dengan:
      person, person_family, person_party, person_dapil,
      spouse, spouse_family, spouse_party, spouse_dapil,
      cross_family, cross_party, relation_label
  - cross_party_pairs[]: pasangan partai dan jumlah pernikahan
  - cross_family_pairs[]: pasangan keluarga dan jumlah pernikahan
  - multi_family_bridge_persons[]: orang yang menikah ke >1 keluarga berbeda

TUGASMU:
1. Jelaskan secara naratif:
   - seberapa banyak pernikahan politik (cross_family dan cross_party),
   - partai mana yang paling sering terhubung lewat pernikahan,
   - keluarga mana yang sering muncul dalam pasangan keluarga (cross_family_pairs).

2. Berikan contoh konkret:
   - beberapa contoh pernikahan cross-party,
   - beberapa contoh tokoh yang menjadi "jembatan" banyak keluarga (multi_family_bridge_persons).

3. Kaitkan dengan konsep "Strategic Marriage — Mendeteksi Pernikahan Politik":
   - bagaimana pola-pola tersebut bisa memperluas jaringan kekuasaan?
   - apa implikasinya bagi konsolidasi kekuasaan atau koalisi informal?

PENTING:
- Jangan mengarang data baru di luar JSON.
- Jika data terbatas (misalnya sedikit pernikahan politik), jelaskan keterbatasan itu.
- Jawaban dalam bahasa Indonesia, gaya analitis tapi mudah dipahami, 4–8 paragraf.
""".strip()

strategic_agent = create_react_agent(
    llm,
    tools=[get_strategic_marriage_summary],
    prompt=SYSTEM_PROMPT_A3,
    name="strategic_marriage_agent",
)


def run_strategic_marriage_agent(csv_path: str):
    user_prompt = textwrap.dedent(f"""
    Lakukan analisis "Strategic Marriage — Mendeteksi Pernikahan Politik"
    dengan membaca file CSV berikut:

    {csv_path}

    Langkah:
    1. Panggil tool `get_strategic_marriage_summary` dengan csv_path ini.
    2. Gunakan JSON yang dikembalikan untuk membuat analisis seperti di sistem prompt.
    """)

    state = strategic_agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    messages = state["messages"]
    last_msg = messages[-1]
    content = last_msg.content

    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        content_str = "\n".join(parts)
    else:
        content_str = str(content)

    print("\n===== HASIL ANALISIS STRATEGIC MARRIAGE =====\n")
    print(content_str)
    print("\n=============================================\n")


# ==================================================
# 12. TOOL & AGENT 4: BANGUN RELASI NAMA–DAPIL–PARTAI–JABATAN–PENDIDIKAN–PASANGAN–KELUARGA
# ==================================================

@tool
def build_kg_from_enriched_csv(csv_path: str, max_rows: int = 1000) -> str:
    """
    Tool Agent 4:
    - Baca CSV yang berisi kolom:
      Nama, Dapil, Partai, Jabatan, Pendidikan, Pasangan, Keluarga
    - Bangun node & relasi dasar di Neo4j:
      (Person)-[:REPRESENTS]->(Dapil)
      (Person)-[:MEMBER_OF]->(Party)
      (Person)-[:HOLDS_POSITION]->(Position)
      (Person)-[:ALUMNI_OF]->(Education)
      (Person)-[:SPOUSE_OF]->(Person)
      (Person)-[:FAMILY_OF]->(Person)
    """
    global driver
    if driver is None:
        raise RuntimeError("Neo4j driver belum diinisialisasi.")

    df = pd.read_csv(csv_path)
    if max_rows < len(df):
        df = df.head(max_rows)

    with driver.session() as session:
        for _, row in df.iterrows():
            nama = str(row.get("Nama", "")).strip()
            if not nama:
                continue

            dapil = str(row.get("Dapil", "")).strip()
            partai = str(row.get("Partai", "")).strip()
            jabatan = str(row.get("Jabatan", "")).strip()
            pendidikan = str(row.get("Pendidikan", "")).strip()

            pasangan_val = row.get("Pasangan", "")
            keluarga_val = row.get("Keluarga", "")

            pasangan = "" if pd.isna(pasangan_val) else str(pasangan_val).strip()
            keluarga = "" if pd.isna(keluarga_val) else str(keluarga_val).strip()

            # --- Person utama ---
            session.run(
                """
                MERGE (p:Person {name: $nama})
                ON CREATE SET p.created_at = timestamp()
                SET p.last_seen = timestamp(),
                    p.dapil = $dapil,
                    p.partai = $partai,
                    p.jabatan_raw = $jabatan,
                    p.pendidikan_raw = $pendidikan
                """,
                nama=nama,
                dapil=dapil,
                partai=partai,
                jabatan=jabatan,
                pendidikan=pendidikan,
            )

            # --- Dapil ---
            if dapil:
                session.run(
                    """
                    MERGE (d:Dapil {name: $dapil})
                    MERGE (p:Person {name: $nama})
                    MERGE (p)-[:REPRESENTS]->(d)
                    """,
                    dapil=dapil,
                    nama=nama,
                )

            # --- Partai ---
            if partai:
                session.run(
                    """
                    MERGE (par:Party {name: $partai})
                    MERGE (p:Person {name: $nama})
                    MERGE (p)-[:MEMBER_OF]->(par)
                    """,
                    partai=partai,
                    nama=nama,
                )

            # --- Jabatan (bisa banyak, dipisah ; ) ---
            if jabatan:
                for j in [x.strip() for x in jabatan.split(";") if x.strip()]:
                    session.run(
                        """
                        MERGE (pos:Position {name: $jabatan})
                        MERGE (p:Person {name: $nama})
                        MERGE (p)-[:HOLDS_POSITION]->(pos)
                        """,
                        jabatan=j,
                        nama=nama,
                    )

            # --- Pendidikan (bisa banyak, dipisah ; ) ---
            if pendidikan:
                for edu in [x.strip() for x in pendidikan.split(";") if x.strip()]:
                    session.run(
                        """
                        MERGE (u:Education {name: $edu})
                        MERGE (p:Person {name: $nama})
                        MERGE (p)-[:ALUMNI_OF]->(u)
                        """,
                        edu=edu,
                        nama=nama,
                    )

            # --- Pasangan -> SPOUSE_OF ---
            if pasangan:
                pasangan_entries = [e.strip() for e in pasangan.split(";") if e.strip()]
                for entry in pasangan_entries:
                    spouse_name = entry
                    rel_label = "pasangan"

                    # Format umum: "Jo Lin Sumbardi (istri)" dst.
                    if "(" in entry:
                        name_part, rel_part = entry.split("(", 1)
                        spouse_name = name_part.strip()
                        rel_part = rel_part.rstrip(")")
                        rel_label = rel_part.strip()  # misalnya "istri", "suami"

                    session.run(
                        """
                        MERGE (s:Person {name: $spouse_name})
                        ON CREATE SET s.created_at = timestamp()
                        SET s.last_seen = timestamp()
                        """,
                        spouse_name=spouse_name,
                    )

                    session.run(
                        """
                        MATCH (p:Person {name: $nama}),
                              (s:Person {name: $spouse_name})
                        MERGE (p)-[r:SPOUSE_OF]->(s)
                        SET r.relation_label = $rel_label,
                            r.created_at = coalesce(r.created_at, timestamp()),
                            r.last_seen = timestamp()
                        """,
                        nama=nama,
                        spouse_name=spouse_name,
                        rel_label=rel_label,
                    )

            # --- Keluarga lain -> FAMILY_OF (sederhana) ---
            if keluarga:
                keluarga_entries = [e.strip() for e in keluarga.split(";") if e.strip()]
                for entry in keluarga_entries:
                    fam_name = entry
                    note = ""

                    if "(" in entry:
                        name_part, note_part = entry.split("(", 1)
                        fam_name = name_part.strip()
                        note_part = note_part.rstrip(")")
                        note = note_part.strip()

                    session.run(
                        """
                        MERGE (f:Person {name: $fam_name})
                        ON CREATE SET f.created_at = timestamp()
                        SET f.last_seen = timestamp()
                        """,
                        fam_name=fam_name,
                    )

                    session.run(
                        """
                        MATCH (p:Person {name: $nama}),
                              (f:Person {name: $fam_name})
                        MERGE (p)-[r:FAMILY_OF]->(f)
                        SET r.note = $note,
                            r.created_at = coalesce(r.created_at, timestamp()),
                            r.last_seen = timestamp()
                        """,
                        nama=nama,
                        fam_name=fam_name,
                        note=note,
                    )

    return f"Berhasil membangun KG dari {len(df)} baris di {csv_path}"


SYSTEM_PROMPT_A4 = """
Kamu adalah agen pembangun knowledge graph (Agent 4) dari data CSV anggota DPR.

Tugas:
- Menerima path file CSV yang berisi kolom:
  Nama, Dapil, Partai, Jabatan, Pendidikan, Pasangan, Keluarga.
- PANGGIL tool `build_kg_from_enriched_csv` TEPAT SATU KALI dengan:
  - csv_path = path CSV
  - max_rows = sesuai instruksi di pesan user (misalnya 1000).

Setelah tool dipanggil:
- Balas singkat dalam bahasa Indonesia, misalnya:
  "KG berhasil dibangun dari N baris."
- Jangan mengarang data di luar hasil tool.
""".strip()

kg_rel_agent = create_react_agent(
    llm,
    tools=[build_kg_from_enriched_csv],
    prompt=SYSTEM_PROMPT_A4,
    name="relation_kg_agent",
)


def run_relation_kg_agent(csv_path: str, max_rows: int = 1000):
    user_prompt = textwrap.dedent(f"""
    Bangun knowledge graph Neo4j dari file CSV berikut:

    {csv_path}

    CSV ini memiliki kolom:
    Nama,Dapil,Partai,Jabatan,Pendidikan,Pasangan,Keluarga

    Panggil tool `build_kg_from_enriched_csv` dengan:
    - csv_path = "{csv_path}"
    - max_rows = {max_rows}
    """)
    state = kg_rel_agent.invoke(
        {"messages": [{"role": "user", "content": user_prompt}]}
    )
    messages = state["messages"]
    last_msg = messages[-1]
    content = last_msg.content

    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        content_str = "\n".join(parts)
    else:
        content_str = str(content)

    print("\n===== HASIL AGENT 4 (RELATIONAL KG) =====\n")
    print(content_str)
    print("\n=========================================\n")


# ==================================================
# 13. TOOL & AGENT 5: QA NEO4J (BAHASA INDONESIA -> CYPHER)
# ==================================================

@tool
def run_cypher_query(cypher: str) -> str:
    """
    Tool Agent 5:
    Jalankan query Cypher ke Neo4j dan kembalikan hasilnya sebagai JSON string.
    """
    global driver
    if driver is None:
        raise RuntimeError("Neo4j driver belum diinisialisasi.")

    with driver.session() as session:
        result = session.run(cypher)
        rows = [dict(r) for r in result]

    return json.dumps(
        {
            "cypher": cypher,
            "rows": rows,
        },
        ensure_ascii=False,
    )


SYSTEM_PROMPT_A5 = """
Kamu adalah Agent 5, asisten tanya-jawab untuk database graf Neo4j yang berisi data anggota DPR dan relasi keluarganya.

Skema graf (ringkas):

Node:
- (:Person {name, dapil, partai, jabatan_raw, pendidikan_raw, ...})
- (:Dapil {name})
- (:Party {name})
- (:Position {name})
- (:Education {name})

Relasi penting:
- (p:Person)-[:REPRESENTS]->(d:Dapil)
- (p:Person)-[:MEMBER_OF]->(par:Party)
- (p:Person)-[:HOLDS_POSITION]->(pos:Position)
- (p:Person)-[:ALUMNI_OF]->(u:Education)
- (p:Person)-[:SPOUSE_OF]->(s:Person)
- (p:Person)-[:FAMILY_OF]->(f:Person)
- (p:Person)-[:PARENT_OF]->(c:Person)   -- jika ada
- (p:Person)-[:IN_LAW_OF]->(x:Person)   -- jika ada

TUGASMU:
- Menerjemahkan pertanyaan dalam bahasa Indonesia menjadi query Cypher terhadap graf di atas.
- SELALU panggil tool `run_cypher_query` TEPAT SATU KALI dengan parameter:
  - cypher = string query Cypher yang kamu susun.

Langkah berpikir (di kepalamu, jangan ditulis eksplisit):
1. Pahami maksud pertanyaan (misalnya: "siapa anggota DPR dari Lampung I yang punya pasangan dari partai berbeda?").
2. Tentukan node/relasi yang relevan.
3. Susun query Cypher yang valid dan efisien.
4. Panggil tool `run_cypher_query` dengan query tersebut.
5. Gunakan hasil tool (JSON) untuk membuat jawaban yang rapi.

OUTPUT KE USER:
- Tampilkan dulu query Cypher yang kamu gunakan dalam blok kode.
- Setelah itu, jelaskan hasilnya dalam bahasa Indonesia yang jelas, berupa daftar/tabel ringkas:
  - Jika hasil berupa daftar orang: sebutkan minimal name, partai, dapil.
  - Jika hasil kosong, jelaskan bahwa tidak ditemukan data yang cocok.
- Jangan menuliskan langkah-langkah berpikir internalmu.
""".strip()

qa_agent = create_react_agent(
    llm,
    tools=[run_cypher_query],
    prompt=SYSTEM_PROMPT_A5,
    name="cypher_qa_agent",
)


def run_agent5_qa(question: str):
    """
    Jalankan Agent 5 untuk satu pertanyaan bahasa Indonesia.
    """
    state = qa_agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    messages = state["messages"]
    last_msg = messages[-1]
    content = last_msg.content

    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            else:
                parts.append(str(c))
        content_str = "\n".join(parts)
    else:
        content_str = str(content)

    print("\n===== HASIL AGENT 5 (QA NEO4J) =====\n")
    print(content_str)
    print("\n====================================\n")


# ==================================================
# 14. MAIN: PILIH MODE (AGENT 1+2, 3, 4, 5)
# ==================================================

if __name__ == "__main__":
    mode = input(
        "Pilih mode:\n"
        "  1 = Jalankan Agent 1 + 2 (scrape Wikipedia + tulis Neo4j + update CSV)\n"
        "  3 = Jalankan Agent 3 saja (analisis Strategic Marriage dari anggota_dpr_enriched.csv)\n"
        "  4 = Jalankan Agent 4 (bangun relasi Nama–Dapil–Partai–Jabatan–Pendidikan–Pasangan–Keluarga ke Neo4j dari CSV)\n"
        "  5 = Jalankan Agent 5 (tanya jawab ke Neo4j pakai bahasa Indonesia -> Cypher)\n"
        "Masukkan pilihan (1/3/4/5): "
    ).strip()

    if mode == "1":
        # Inisialisasi Neo4j driver global
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        try:
            process_csv_with_agents_1_2(CSV_RAW_PATH, max_rows=1000)
        finally:
            driver.close()

    elif mode == "3":
        # Hanya baca CSV yang sudah enriched, tidak scraping ulang, tidak token ekstraksi per orang
        if not os.path.exists(CSV_ENRICHED_PATH):
            print(f"⚠️ File {CSV_ENRICHED_PATH} tidak ditemukan. Pastikan sudah menjalankan mode 1 sebelumnya.")
        else:
            run_strategic_marriage_agent(CSV_ENRICHED_PATH)

    elif mode == "4":
        # Bangun KG relasi Nama–Dapil–Partai–Jabatan–Pendidikan–Pasangan–Keluarga
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        try:
            # Kalau sudah ada enriched, pakai itu; kalau belum, fallback ke raw
            if os.path.exists(CSV_ENRICHED_PATH):
                csv_path = CSV_ENRICHED_PATH
            else:
                csv_path = CSV_RAW_PATH
            print(f"📄 Menggunakan file CSV: {csv_path}")
            run_relation_kg_agent(csv_path, max_rows=1000)
        finally:
            driver.close()

    elif mode == "5":
        # Mode QA: tanya jawab ke Neo4j dengan bahasa Indonesia
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        try:
            print(
                "\nMode Agent 5 (QA Neo4j).\n"
                "Ketik pertanyaan dalam bahasa Indonesia tentang graf DPR.\n"
                "Contoh: 'siapa anggota DPR dari Lampung I yang punya pasangan dari partai berbeda?'\n"
                "Ketik 'exit' untuk keluar.\n"
            )
            while True:
                q = input("Pertanyaan: ").strip()
                if not q:
                    continue
                if q.lower() in ("exit", "quit", "keluar", "q"):
                    print("Keluar dari mode Agent 5.")
                    break
                run_agent5_qa(q)
        finally:
            driver.close()

    else:
        print("Pilihan tidak dikenal. Jalankan lagi dan pilih 1, 3, 4, atau 5.")
