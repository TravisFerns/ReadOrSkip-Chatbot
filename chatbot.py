import json
import re
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import difflib

# ---------------------------
# 1) Load intents (ML for intent classification)
# ---------------------------
# expects columns: intent,pattern,response
intents = pd.read_csv("intents.csv")

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(intents["pattern"])
y = intents["intent"]
clf = MultinomialNB()
clf.fit(X, y)

# ---------------------------
# 2) Load books dataset
# ---------------------------
with open("summaries.json", "r", encoding="utf-8") as f:
    books_data = json.load(f)

# Small memory to remember last discussed book
last_book = {"book": None}

# ---------------------------
# 3) Text normalization helpers
# ---------------------------
HELPER_PHRASES = (
    "tell me about", "summary of", "give me the summary of", "short summary of",
    "what is", "what's", "whats", "can you summarize", "verdict on",
    "who is the author of", "author of", "who wrote", "writer of",
    "how many pages in", "total pages of", "length of", "page count of",
    "number of pages in", "how long is",
    "rating of", "goodreads rating of", "average rating of", "how is", "what is the rating of"
)

def _strip_smart_quotes(s: str) -> str:
    # Normalize curly quotes/dashes to basic ASCII so matches succeed
    return (
        s.replace("’", "'").replace("‘", "'")
         .replace("“", '"').replace("”", '"')
         .replace("–", "-").replace("—", "-")
    )

def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = _strip_smart_quotes(s.lower())
    # remove punctuation -> spaces
    s = re.sub(r"[^\w\s]", " ", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_title(s: str) -> str:
    s = normalize_text(s)
    # drop leading articles to improve matching
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s

# Precompute normalized titles for all books
BOOK_TITLE_INDEX = [(normalize_title(b["title"]), b) for b in books_data]

# ---------------------------
# 4) Book finding (robust)
# ---------------------------
def clean_query_from_helpers(message: str) -> str:
    """Remove only helper phrases, avoid extra word stripping."""
    m = message.lower()
    m = _strip_smart_quotes(m)
    for p in HELPER_PHRASES:
        # Remove the helper phrase only when it occurs as a full phrase (with spaces around)
        pattern = r"\b" + re.escape(p) + r"\b"
        m = re.sub(pattern, "", m)
    return re.sub(r"\s+", " ", m).strip()

def find_book_in_message(message: str):
    if not message:
        return None

    msg_norm = normalize_text(message)
    if not msg_norm:
        return None

    # 1) Word-boundary containment of known titles inside the message
    candidates = []
    for nt, book in BOOK_TITLE_INDEX:
        if not nt:
            continue
        pattern = r"\b" + re.escape(nt) + r"\b"
        if re.search(pattern, msg_norm):
            candidates.append((len(nt), book))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # 2) Exact equality against normalized titles
    for nt, book in BOOK_TITLE_INDEX:
        if msg_norm == nt:
            return book

    # 3) Fuzzy match (normal)
    titles_norm = [nt for nt, _ in BOOK_TITLE_INDEX]
    match = difflib.get_close_matches(msg_norm, titles_norm, n=1, cutoff=0.73)
    if match:
        idx = titles_norm.index(match[0])
        return BOOK_TITLE_INDEX[idx][1]

    # 4) Fuzzy match after removing leading articles from user input
    msg_norm_no_article = re.sub(r"^(the|a|an)\s+", "", msg_norm)
    match = difflib.get_close_matches(msg_norm_no_article, titles_norm, n=1, cutoff=0.73)
    if match:
        idx = titles_norm.index(match[0])
        return BOOK_TITLE_INDEX[idx][1]

    return None


def extract_book_from_input(user_input: str):
    """Remove helper phrases first, then find the book in the cleaned text."""
    cleaned = clean_query_from_helpers(user_input)
    return find_book_in_message(cleaned)

# ---------------------------
# 5) Main response function
# ---------------------------
def get_bot_response(user_input: str) -> str:
    # Step A: Predict intent
    X_test = vectorizer.transform([user_input])
    predicted_intent = clf.predict(X_test)[0]

    # Step B: Handle intents
    if predicted_intent == "greeting":
        responses = intents[intents["intent"] == "greeting"]["response"].dropna().tolist()
        return random.choice(responses) if responses else "Hello! How can I help you today?"

    elif predicted_intent == "random_book":
        book = random.choice(books_data)
        last_book["book"] = book  # remember context
        return f"📖 Random pick: {book['title']} — {book['summary']}"

    elif predicted_intent in ["summary", "verdict", "author", "pages", "rating"]:
        # Try to extract the book from the current message
        book = extract_book_from_input(user_input)

        # If not found in current message, use the last discussed book
        if book:
            last_book["book"] = book
        else:
            book = last_book["book"]

        if not book:
            return "Please mention the book title."

        print("DEBUG 📚 Matched book:", book)   # Show full book dict in console

        if predicted_intent == "summary":
            return f"📖 {book.get('title', 'Unknown Title')} — {book.get('summary', 'No summary available')}"

        elif predicted_intent == "verdict":
            return f"✅ Verdict on {book.get('title', 'Unknown Title')}: {book.get('verdict', 'No verdict available')}"

        elif predicted_intent == "author":
            return f"✍️ Author of {book.get('title', 'Unknown Title')} is {book.get('Author', 'Unknown Author')}."

        elif predicted_intent == "pages":
            return f"📄 {book.get('title', 'Unknown Title')} has {book.get('Pages', 'Unknown number of')}."

        elif predicted_intent == "rating":
            return f"⭐ Rating of {book.get('title', 'Unknown Title')}: {book.get('Rating', 'No rating available')}"

        return f"Sorry, I couldn’t find details for '{book.get('title', 'Unknown')}'."

# ---------------------------
# 6) Local REPL (optional)
# ---------------------------
if __name__ == "__main__":
    print("Chatbot is ready! Type 'quit' to exit.")
    while True:
        ui = input("You: ")
        if ui.lower() == "quit":
            break
        print("Bot:", get_bot_response(ui))
