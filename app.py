from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for
import chromadb
from chromadb.utils import embedding_functions

app = Flask(__name__)

IMAGE_DIRECTORY = Path(app.static_folder) / "images"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}


def image_url_for_id(doc_id):
    """Return a public static URL for a document image, or None when absent."""
    if not doc_id:
        return None

    # Match by stem so the database does not need to store file extensions.
    # Comparing resolved paths also prevents an ID containing path separators
    # from escaping the static/images directory.
    safe_id = Path(str(doc_id)).name
    requested_stem = Path(safe_id).stem.casefold()
    if not requested_stem:
        return None

    try:
        image_path = next(
            (
                path for path in IMAGE_DIRECTORY.iterdir()
                if path.is_file()
                and path.suffix.casefold() in SUPPORTED_IMAGE_EXTENSIONS
                and path.stem.casefold() == requested_stem
            ),
            None,
        )
    except (FileNotFoundError, OSError):
        return None

    if image_path is None:
        return None

    relative_filename = image_path.relative_to(Path(app.static_folder)).as_posix()
    return url_for("static", filename=relative_filename)

# 1. Подключаемся к базе данных
client = chromadb.PersistentClient(path="./my_grammar_db")

# 2. Явно задаем стандартную легкую функцию эмбеддингов ChromaDB (DefaultEmbeddingFunction)
# Она использует те же веса, но часто отрабатывает стабильнее без сетевых сбоев сторонних ONNX-модулей
default_ef = embedding_functions.DefaultEmbeddingFunction()

# 3. Создаем коллекцию с этой функцией
collection = client.get_or_create_collection(
    name="grammar_images",
    embedding_function=default_ef
)

# 4. Автозаполнение демо-данными, если база пустая
if collection.count() == 0:
    data = [
        {
            "id": "doc_img_01",
            "title": "Verbos Irregulares - Ir y Ser + Verbos en -UIR (Construir, Destruir)",
            "keywords": [
                "verbos muy irregulares",
                "ser",
                "ir",
                "soy",
                "eres",
                "es",
                "somos",
                "sois",
                "son",
                "voy",
                "vas",
                "va",
                "vamos",
                "vais",
                "van",
                "destruir",
                "construir",
                "destruyo",
                "destruyes",
                "destruye",
                "destruimos",
                "destruís",
                "destruyen",
                "construyo",
                "construyes",
                "construye",
                "construimos",
                "construís",
                "construyen",
                "cambio y",
                "huir",
                "sustituir",
                "influir",
                "distribuir",
                "nosotros y vosotros son regulares",
                "gramatica espanol"
            ]
        },
        {
            "id": "doc_img_02",
            "title": "El Presente de Indicativo - Verbos Irregulares (Cambio de vocal: E>IE, E>I, O>UE, U>UE)",
            "keywords": [
                "presente de indicativo",
                "verbos irregulares",
                "cambio de vocal",
                "pensar",
                "pedir",
                "poder",
                "jugar",
                "e>ie",
                "e>i",
                "o>ue",
                "u>ue",
                "pienso",
                "piensas",
                "piensa",
                "pensamos",
                "pensáis",
                "piensan",
                "pido",
                "pides",
                "pide",
                "pedimos",
                "pedís",
                "piden",
                "puedo",
                "puedes",
                "puede",
                "podemos",
                "podéis",
                "pueden",
                "juego",
                "juegas",
                "juega",
                "jugamos",
                "jugáis",
                "juegan",
                "quiero",
                "quieres",
                "quiere",
                "queremos",
                "queréis",
                "quieren",
                "botella de vino",
                "gramatica espanol"
            ]
        },
        {
            "id": "doc_img_03",
            "title": "Verbos Corcho - Cambios en primera persona (C>ZC, G, Tener, Decir, Venir, Oír, Otros)",
            "keywords": [
                "verbos corcho",
                "solo cambia la primera persona del singular yo",
                "c>zc",
                "traductor",
                "traduzco",
                "conozco",
                "parezco",
                "conduzco",
                "g",
                "hacer",
                "hago",
                "caer",
                "caigo",
                "traer",
                "traigo",
                "salir",
                "salgo",
                "tener",
                "decir",
                "venir",
                "oír",
                "tengo",
                "tienes",
                "tiene",
                "tenemos",
                "tenéis",
                "tienen",
                "digo",
                "dices",
                "dice",
                "decimos",
                "decís",
                "dicen",
                "vengo",
                "vienes",
                "viene",
                "venimos",
                "venéis",
                "vienen",
                "oigo",
                "oyes",
                "oye",
                "oimos",
                "ois",
                "oyen",
                "dar",
                "doy",
                "estar",
                "estoy",
                "saber",
                "sé",
                "poner",
                "pongo",
                "pones",
                "pone",
                "ponemos",
                "ponéis",
                "ponen",
                "botella de vino y corcho",
                "gramatica"
            ]
        }
    ]

    collection.add(
        ids=[item["id"] for item in data],
        documents=[", ".join(item["keywords"]) for item in data],
        metadatas=[{"title": item["title"]} for item in data]
    )


# --- РОУТЫ ---

@app.route("/")
def index():
    total_count = collection.count()
    return render_template("index.html", count=total_count)


@app.route("/search")
def search_page():
    return redirect(url_for("index"))


@app.route("/response")
def response_page():
    query = request.args.get("query", "").strip()
    results = search_results(query)
    return render_template("response.html", query=query, results=results)


def search_results(query):
    results = []

    if query:
        raw_results = collection.query(
            query_texts=[query],
            n_results=3
        )

        if raw_results and raw_results['ids'] and raw_results['ids'][0]:
            for doc_id, meta, dist in zip(raw_results['ids'][0], raw_results['metadatas'][0],
                                          raw_results['distances'][0]):
                results.append({
                    "id": doc_id,
                    "title": (meta or {}).get("title") or "Материал без названия",
                    "distance": round(dist, 3),
                    "image_url": image_url_for_id(doc_id),
                })

    return results


if __name__ == "__main__":
    app.run(debug=True)
