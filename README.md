# 🧠 Intelligent Consulting Copilot

**Intelligent Consulting Copilot** est un outil interne de **knowledge management** et de **génération automatique de livrables** basé sur l’IA.  
Il combine **RAG (Retrieval-Augmented Generation)**, **LLM** et une interface **Streamlit** moderne pour aider les consultants à retrouver rapidement les bonnes informations et produire des résumés, présentations et rapports.

---

## ✨ Fonctionnalités principales

✅ **Knowledge Chat**  
Posez vos questions en langage naturel sur les documents indexés et obtenez des réponses **contextualisées** et **citées**.

✅ **Upload & Index**  
Chargez vos PDF/DOCX/TXT, extrayez et découpez le texte en *chunks*, encodez-les avec **SentenceTransformers MiniLM** et stockez-les dans **ChromaDB**.

✅ **Generate Docs**  
Générez en un clic un **DOCX** ou **PPTX** basé sur la réponse du chatbot ou sur un contenu personnalisé.

✅ **Market Watch** *(placeholder évolutif)*  
Tableau de veille concurrentielle : appels d’offres détectés, news des concurrents, tendances technologiques et rapport PDF hebdomadaire.

---

## 📁 Arborescence du projet

```text
intelligent_copilot/
├─ data/
│   ├─ raw/           # documents bruts uploadés
│   ├─ interim/       # fichiers temporaires (OCR, prétraitement)
│   └─ processed/     # chunks.jsonl et autres sorties
├─ vectors/           # base Chroma persistée (embeddings)
├─ out/               # livrables générés (pptx/docx/pdf)
├─ logs/              # logs d’exécution
├─ scripts/           # scripts utilitaires PowerShell (setup, ingestion, run)
├─ src/
│   ├─ config/        # settings, chemins, logging
│   ├─ utils/         # I/O helpers, texte, timers, métriques
│   ├─ ingestion/     # extraction texte, chunking, pipeline
│   ├─ indexing/      # embeddings, stockage et build index
│   ├─ rag/           # retriever, prompts, LLM et RAGChain LangChain
│   ├─ generation/    # générateurs DOCX, PPTX et PDF
│   └─ ui/            # app.py (interface Streamlit)
└─ tests/             # tests unitaires de base
