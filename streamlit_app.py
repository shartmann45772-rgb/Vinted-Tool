"""
Vinted Preis-Checker - Web-App-Version (für Hugging Face Spaces)
==================================================================

Gleiche Logik wie die Desktop-Version, aber statt eines automatischen
Screenshots lädst du hier ein Bild (Screenshot vom Vinted-Inserat) über
eine Web-Oberfläche hoch — das funktioniert auch vom iPhone/iPad aus,
einfach im Browser aufrufen.

Konfiguration (als "Secret" in den Hugging-Face-Space-Einstellungen setzen):
    GEMINI_API_KEY      -> kostenloser Schlüssel von https://aistudio.google.com/apikey
    VINTED_DOMAIN        -> optional, Standard: www.vinted.de
    ENABLE_VISUAL_MATCH  -> optional, "false" um den Bildabgleich abzuschalten
"""

import io
import json
import os
import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

import requests
import streamlit as st
from PIL import Image

import google.generativeai as genai

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

# Streamlit Cloud speichert Secrets in st.secrets; hier zusätzlich als normale
# Umgebungsvariable verfügbar machen, falls andere Teile des Codes os.environ nutzen.
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
if "VINTED_DOMAIN" in st.secrets:
    os.environ["VINTED_DOMAIN"] = st.secrets["VINTED_DOMAIN"]
if "ENABLE_VISUAL_MATCH" in st.secrets:
    os.environ["ENABLE_VISUAL_MATCH"] = st.secrets["ENABLE_VISUAL_MATCH"]

VINTED_DOMAIN = os.environ.get("VINTED_DOMAIN", "www.vinted.de")
MAX_RESULTS = 100
GEMINI_MODEL = "gemini-flash-lite-latest"

BUYER_PROTECTION_PERCENT = 0.05
BUYER_PROTECTION_FIXED = 0.70

ENABLE_VISUAL_MATCH = os.environ.get("ENABLE_VISUAL_MATCH", "true").lower() != "false"
VISUAL_MATCH_CHUNK_SIZE = 12
VISUAL_MATCH_MAX_ITEMS = 60

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
_model = genai.GenerativeModel(GEMINI_MODEL)


def calculate_buyer_protection(price: float) -> float:
    if not price:
        return 0.0
    return round(price * BUYER_PROTECTION_PERCENT + BUYER_PROTECTION_FIXED, 2)


# ---------------------------------------------------------------------------
# Produktinfo per Gemini Vision extrahieren
# ---------------------------------------------------------------------------

@dataclass
class ProductInfo:
    brand: Optional[str] = None
    model: Optional[str] = None
    size: Optional[str] = None
    color: Optional[str] = None
    material: Optional[str] = None
    condition: Optional[str] = None
    listing_price: Optional[float] = None
    shipping_price: Optional[float] = None
    search_query: Optional[str] = None
    raw: dict = field(default_factory=dict)


EXTRACTION_PROMPT = """Du siehst ein oder mehrere Screenshots DESSELBEN Vinted-Inserats (z.B. einen
vom Produktfoto und einen von der Beschreibung/dem Preis, weil man auf dem Handy oft scrollen muss).
Kombiniere die Informationen aus ALLEN Bildern zu einem Ergebnis.
Extrahiere folgende Felder als reines JSON-Objekt (keine Erklärung, kein Markdown, nur JSON):

{
  "brand": "Markenname, z.B. Nike",
  "model": "Produktbezeichnung/Modellname, z.B. Trackpants",
  "full_title": "Der komplette Anzeigetitel des Inserats, WORTWÖRTLICH wie er auf dem Screenshot steht, inklusive aller Zusätze wie Kollektionen/Kollabos (z.B. 'Nike x Nocta Trackpants', nicht nur 'Nike Trackpants'). Falls kein eindeutiger Titel sichtbar ist, null.",
  "size": "Größe, z.B. M oder 38, falls sichtbar sonst null",
  "color": "Farbe, wie auf Vinted angegeben (z.B. Grau, Schwarz), falls sichtbar sonst null",
  "material": "Material, wie auf Vinted angegeben (z.B. Baumwolle, Polyester), falls sichtbar sonst null",
  "condition": "Zustand, z.B. 'Sehr guter Zustand', falls sichtbar sonst null",
  "listing_price": Preis des Artikels als Zahl (ohne Währungssymbol) oder null,
  "shipping_price": Versandpreis als Zahl oder null
}

WICHTIG bei "full_title": Übernimm JEDEN sichtbaren Zusatzbegriff (Kollektion, Kooperation/Collab,
Sondername) exakt so, wie er geschrieben steht. Vereinfache NICHTS und lass keine Wörter aus dem
sichtbaren Titel weg, auch wenn dir die Marke oder Kollektion unbekannt ist.

Wenn ein Feld nicht erkennbar ist, setze es auf null. Antworte NUR mit dem JSON-Objekt."""


def extract_product_info(images_bytes: list) -> ProductInfo:
    """Nimmt eine LISTE von Screenshots entgegen (z.B. eins vom Produktfoto, eins von der
    Beschreibung/Preis) und kombiniert die Infos daraus in einer einzigen Anfrage."""
    image_parts = [{"mime_type": "image/png", "data": b} for b in images_bytes]
    response = _model.generate_content([EXTRACTION_PROMPT, *image_parts])
    text = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"Konnte Modellantwort nicht parsen: {text!r}")

    full_title = (data.get("full_title") or "").strip()
    brand = (data.get("brand") or "").strip()
    model = (data.get("model") or "").strip()
    color = (data.get("color") or "").strip()
    fallback_query = f"{brand} {model}".strip()

    if full_title and brand and brand.lower() not in full_title.lower():
        search_query = f"{brand} {full_title}".strip()
    else:
        search_query = full_title or fallback_query

    if color and color.lower() not in search_query.lower():
        search_query = f"{search_query} {color}".strip()

    return ProductInfo(
        brand=data.get("brand"),
        model=data.get("model"),
        size=data.get("size"),
        color=data.get("color"),
        material=data.get("material"),
        condition=data.get("condition"),
        listing_price=data.get("listing_price"),
        shipping_price=data.get("shipping_price"),
        search_query=search_query,
        raw=data,
    )


# ---------------------------------------------------------------------------
# Vinted-Suche (inoffizielle interne API)
# ---------------------------------------------------------------------------

class VintedClient:
    def __init__(self, domain: str = VINTED_DOMAIN):
        self.base = f"https://{domain}"
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
            }
        )
        self._warm_up()

    def _warm_up(self):
        self.session.get(self.base, timeout=10)

    def search_items(self, query: str, per_page: int = MAX_RESULTS) -> list:
        url = f"{self.base}/api/v2/catalog/items"
        params = {
            "search_text": query,
            "per_page": per_page,
            "page": 1,
            "order": "price_low_to_high",
        }
        resp = self.session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"Vinted-Suche fehlgeschlagen ({resp.status_code}).")
        return resp.json().get("items", [])


# ---------------------------------------------------------------------------
# Preis-Auswertung
# ---------------------------------------------------------------------------

def filter_matching_items(items: list, info: ProductInfo) -> list:
    if not info.size:
        return items
    size_norm = info.size.strip().lower()
    filtered = [it for it in items if str(it.get("size_title", "")).strip().lower() == size_norm]
    return filtered or items


def compute_average_price(items: list) -> Optional[float]:
    prices = []
    for it in items:
        amount = (it.get("price") or {}).get("amount")
        if amount is not None:
            try:
                prices.append(float(amount))
            except (TypeError, ValueError):
                continue
    return statistics.mean(prices) if prices else None


def compute_likes_stats(items: list) -> dict:
    likes = []
    top_item, top_likes = None, -1
    for it in items:
        count = it.get("favourite_count")
        if count is None:
            continue
        try:
            count = int(count)
        except (TypeError, ValueError):
            continue
        likes.append(count)
        if count > top_likes:
            top_likes, top_item = count, it
    return {
        "avg_likes": statistics.mean(likes) if likes else None,
        "top_likes": top_likes if top_item else None,
        "top_item_title": (top_item or {}).get("title"),
        "top_item_price": ((top_item or {}).get("price") or {}).get("amount"),
    }


def _get_thumbnail_url(item: dict) -> Optional[str]:
    photo = item.get("photo") or {}
    if not photo:
        return None
    return photo.get("url") or (photo.get("thumbnails") or [{}])[0].get("url")


def _fetch_image_bytes(url: str, session: requests.Session) -> Optional[bytes]:
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.content
    except requests.RequestException:
        pass
    return None


VISUAL_MATCH_PROMPT = """Das erste Bild ist ein Referenz-Artikel (der Artikel, dessen Marktwert
der Nutzer herausfinden möchte). Danach folgen mehrere durchnummerierte Vergleichsbilder
(Bild 0, Bild 1, Bild 2, ...) von anderen Vinted-Inseraten der gleichen Produktkategorie/Marke.

Ordne JEDES Vergleichsbild GENAU EINER dieser drei Gruppen zu:

- "same_design": Zeigt eindeutig das GLEICHE Design wie das Referenzbild (gleiches Logo, gleiche
  Grafik, gleiches Muster, gleicher Schnitt, gleiche Sondervariante) — aber es ist erkennbar ein
  ANDERES physisches Exemplar (z.B. andere Aufnahme/anderer Hintergrund/anderer Winkel/anderer
  Zustand). Das sind die eigentlichen Vergleichsartikel.
- "exact_duplicate": Zeigt exakt die GLEICHEN Fotos / das exakt gleiche physische Exemplar wie das
  Referenzbild (z.B. weil es sich um das gleiche Inserat handelt, das der Nutzer gerade selbst
  anschaut oder bereits selbst verkauft). Diese zählen NICHT als unabhängiger Vergleichsartikel.
- Alles andere (anderes Design, andere Marke/Kategorie, unklar): einfach weglassen, nicht auflisten.

WICHTIG: Ein schlichtes Basic-Teil ist NICHT "same_design" zu einem Teil mit auffälligem
Sonder-Logo/Muster, selbst wenn beide die gleiche Marke und Produktart haben. Achte gezielt auf
Logo-Größe/-Platzierung, Muster, Farbverlauf und Schnitt.

Antworte NUR mit einem JSON-Objekt in genau dieser Form (leere Listen falls nichts passt):
{"same_design": [0, 3], "exact_duplicate": [5]}

Kein weiterer Text, nur das JSON-Objekt."""


def visually_filter_items(reference_image_bytes: bytes, items: list, log: list) -> dict:
    if not ENABLE_VISUAL_MATCH or not items:
        return {"same_design": items, "exact_duplicate": []}

    candidates = items[:VISUAL_MATCH_MAX_ITEMS]
    log.append(f"Bildabgleich: vergleiche {len(candidates)} Artikelfotos mit deinem Screenshot...")
    session = requests.Session()
    same_design = []
    exact_duplicate = []

    for chunk_start in range(0, len(candidates), VISUAL_MATCH_CHUNK_SIZE):
        chunk = candidates[chunk_start: chunk_start + VISUAL_MATCH_CHUNK_SIZE]
        parts = [VISUAL_MATCH_PROMPT, {"mime_type": "image/png", "data": reference_image_bytes}]
        chunk_items_with_images = []
        for it in chunk:
            thumb_url = _get_thumbnail_url(it)
            if not thumb_url:
                continue
            img_bytes = _fetch_image_bytes(thumb_url, session)
            if not img_bytes:
                continue
            parts.append({"mime_type": "image/jpeg", "data": img_bytes})
            chunk_items_with_images.append(it)

        if not chunk_items_with_images:
            continue

        try:
            response = _model.generate_content(parts)
            text = re.sub(r"^```json|```$", "", response.text.strip(), flags=re.MULTILINE).strip()
            result = json.loads(text)
            for idx in result.get("same_design", []):
                if isinstance(idx, int) and 0 <= idx < len(chunk_items_with_images):
                    same_design.append(chunk_items_with_images[idx])
            for idx in result.get("exact_duplicate", []):
                if isinstance(idx, int) and 0 <= idx < len(chunk_items_with_images):
                    exact_duplicate.append(chunk_items_with_images[idx])
        except Exception as e:
            log.append(f"Bildabgleich-Warnung: ein Teil übersprungen ({e})")
            continue

    if not same_design and not exact_duplicate:
        return {"same_design": items, "exact_duplicate": []}
    return {"same_design": same_design, "exact_duplicate": exact_duplicate}


def evaluate(info: ProductInfo, reference_image_bytes: bytes, log: list) -> dict:
    vc = VintedClient()
    items = vc.search_items(info.search_query, per_page=MAX_RESULTS)
    matched = filter_matching_items(items, info)

    duplicate_count = 0
    if ENABLE_VISUAL_MATCH:
        visual_result = visually_filter_items(reference_image_bytes, matched, log)
        matched = visual_result["same_design"]
        duplicate_count = len(visual_result["exact_duplicate"])

    avg_price = compute_average_price(matched)
    likes_stats = compute_likes_stats(matched)

    buyer_protection = calculate_buyer_protection(info.listing_price or 0)
    listing_total = (info.listing_price or 0) + (info.shipping_price or 0) + buyer_protection
    margin = margin_pct = None
    if avg_price is not None and listing_total:
        margin = avg_price - listing_total
        margin_pct = (margin / listing_total) * 100

    return {
        "query": info.search_query,
        "matched_count": len(matched),
        "duplicate_count": duplicate_count,
        "avg_price": avg_price,
        "listing_price": info.listing_price,
        "shipping_price": info.shipping_price,
        "buyer_protection": buyer_protection,
        "listing_total": listing_total,
        "margin": margin,
        "margin_pct": margin_pct,
        "avg_likes": likes_stats["avg_likes"],
        "top_likes": likes_stats["top_likes"],
        "top_item_title": likes_stats["top_item_title"],
        "top_item_price": likes_stats["top_item_price"],
    }


# ---------------------------------------------------------------------------
# Web-Oberfläche (Gradio)
# ---------------------------------------------------------------------------

def analyze_screenshot(images: list) -> str:
    if not images:
        return "Bitte zuerst mindestens ein Bild hochladen."

    log = []
    try:
        images_bytes = []
        for image in images:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            images_bytes.append(buf.getvalue())

        log.append(f"Analysiere {len(images_bytes)} Bild(er) mit Gemini...")
        info = extract_product_info(images_bytes)

        if not info.search_query:
            return "Konnte keinen Suchbegriff aus den Bildern erkennen. Sind die Screenshots deutlich genug?"

        log.append(f"Erkannt: {info.brand} / {info.model} | Größe: {info.size} | Farbe: {info.color}")
        log.append(f"Suchbegriff: '{info.search_query}'")
        log.append("Suche auf Vinted...")

        # Das erste hochgeladene Bild dient als Referenzfoto für den Bildabgleich
        # (im Idealfall lädst du das Produktfoto als erstes Bild hoch).
        result = evaluate(info, images_bytes[0], log)

        lines = ["\n".join(log), "\n---\n"]
        lines.append(f"**Treffer verglichen:** {result['matched_count']}")
        if result["duplicate_count"]:
            lines.append(f"({result['duplicate_count']} als dein eigenes/identisches Inserat erkannt und ausgeschlossen)")
        if result["avg_price"] is not None:
            lines.append(f"**Durchschnittspreis vergleichbarer Artikel:** {result['avg_price']:.2f} €")
        else:
            lines.append("Kein Durchschnittspreis ermittelbar.")
        lines.append(f"- Artikelpreis: {result['listing_price'] or 0:.2f} €")
        lines.append(f"- + Versand: {result['shipping_price'] or 0:.2f} €")
        lines.append(f"- + Käuferschutz (geschätzt): {result['buyer_protection']:.2f} €")
        lines.append(f"**Gesamter Einkaufspreis:** {result['listing_total']:.2f} €")
        if result["margin"] is not None:
            sign = "+" if result["margin"] >= 0 else ""
            lines.append(f"**Marge:** {sign}{result['margin']:.2f} € ({sign}{result['margin_pct']:.1f} %)")
        if result["avg_likes"] is not None:
            lines.append(f"Durchschnittliche Likes vergleichbarer Artikel: {result['avg_likes']:.1f}")
        if result["top_likes"] is not None:
            lines.append(
                f"Meiste Likes: {result['top_likes']} "
                f"(\"{result['top_item_title']}\" für {result['top_item_price']} €)"
            )
        return "\n\n".join(lines)

    except Exception as e:
        return f"Fehler: {e}"


st.set_page_config(page_title="Vinted Preis-Checker", page_icon="🛍️")
st.title("🛍️ Vinted Preis-Checker")
st.write(
    "Lade 1-2 Screenshots desselben Vinted-Inserats hoch (z.B. einen vom Produktfoto und "
    "einen von der Beschreibung/dem Preis, falls das auf dem Handy nicht in einen Screenshot "
    "passt). Das Tool erkennt Marke, Modell, Größe, Farbe und Preis, sucht vergleichbare "
    "Artikel auf Vinted und berechnet deine Marge inklusive Versand und Käuferschutz."
)

uploaded_files = st.file_uploader(
    "Screenshot(s) hochladen",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploaded_files:
    images = [Image.open(f) for f in uploaded_files]
    cols = st.columns(len(images))
    for col, img in zip(cols, images):
        col.image(img, width=200)

    if st.button("Analysieren", type="primary"):
        with st.spinner("Analysiere Produkt und suche auf Vinted..."):
            result_text = analyze_screenshot(images)
        st.markdown(result_text)
