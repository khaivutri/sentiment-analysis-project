import re
import emoji


def intelligent_preprocess(text):
    """Preprocess text for sentiment analysis"""
    # Đảm bảo đầu vào là string
    if not isinstance(text, str):
        return ""

    # 1. Chuyển về chữ thường
    text = text.lower()

    # 2. Xóa URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Xóa @mention
    text = re.sub(r'@\w+', '', text)

    # 4. Xử lý hashtag
    def process_hashtag(match):
        hashtag = match.group(1)
        # Tách CamelCase (#BeautifulDay -> 'Beautiful Day')
        words = re.sub(r'([a-z])([A-Z])', r'\1 \2', hashtag)
        return words.lower()

    text = re.sub(r'#(\w+)', process_hashtag, text)

    # 5. Xóa ký tự đặc biệt (giữ lại chữ và số)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 6. Xóa khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text).strip()

    # 7. Giữ emoji thành token riêng
    text = emoji.demojize(text, delimiters=(" ", " "))

    # 8. Tokenize và ghép lại thành chuỗi
    tokens = text.split()
    return " ".join(tokens)
