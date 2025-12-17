Ekstraksi Otomatis dan Analisis Jaringan Kekerabatan Anggota DPR

EAS
Mata Kuliah: Topik dalam Knowledge Graphs

Disusun Oleh:
Moch Rafy Adhipramana Effendy  (6026242012)
Ayunda Kusuma Wardani          (6026242004)

Diampu Oleh:
Nur Aini Rakhmawati, Prof., S.Kom., M.Sc.Eng., Ph.D

Repo ini memaparkan proyek “Ekstraksi Otomatis dan Analisis Jaringan Kekerabatan Anggota DPR” yang bertujuan membangun sistem otomatis untuk mengumpulkan, menstrukturkan, dan menganalisis informasi anggota DPR Indonesia beserta hubungan keluarganya yang tersebar di Wikipedia. Sistem ini memanfaatkan pendekatan multi-agent berbasis Large Language Model (DeepSeek), LangChain, Retrieval-Augmented Generation (RAG), dan basis data graf Neo4j, yang mencakup proses scraping data Wikipedia, ekstraksi informasi ke format CSV, pembangunan Knowledge Graph (meliputi relasi keluarga, partai, daerah pemilihan, jabatan, dan pendidikan), hingga analisis lanjutan menggunakan kueri Cypher dan algoritma PageRank. Hasil analisis menunjukkan kemampuan sistem dalam mengidentifikasi pola dinasti politik, tokoh paling berpengaruh dalam partai, serta dominasi partai di daerah pemilihan tertentu, seperti kasus dinasti politik Ratu Atut Chosiyah di Banten, sehingga membuktikan bahwa pendekatan otomatis berbasis knowledge graph efektif untuk menghasilkan insight politik yang kompleks secara efisien 

Untuk menjalankan repository ini:
1. Git clone Repository ini
2. conda create -n KnowledgeGraphFamily python=3.10.18
3. cd folder hasil git clone
4. jalankan "python -m pip install -r requirements.txt"
5. Isi key untuk deepseek API

