"""Convert CFE token sequences into the feature arrays expected by PIS."""


INITIALS = sorted(
    ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h',
     'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w'],
    key=len,
    reverse=True,
)
PUNCTUATION = set('，。？！《》“”‘’、：；')
TONE_IDS = {'无': 5, '空': 6, '平': 1, '上': 2, '去': 3, '入': 4}
RHYTHM_IDS = {'N': 6, 'e': 7, 'n': 5, '1': 1, '2': 2, '3': 3, '4': 4}


def split_pinyin(value):
    for initial in INITIALS:
        if value.startswith(initial) and len(value) > len(initial):
            return initial, value[len(initial):]
    return '', value


def build_pis_features(cfe):
    """Build phone, tone, rhyme, and sentence-order arrays from one CFE result."""

    text = cfe['text']
    pinyin = cfe['pinyin'][:len(text)]
    tone = cfe['tone'][:len(text)]
    rhythm = cfe['rhythm'][:len(text)]
    if not (len(pinyin) == len(tone) == len(rhythm) == len(text)):
        raise ValueError('CFE output cannot be aligned to the source text')

    phones, tones, rhymes, orders = [], [], [], []
    sentence_order = 1
    used = []
    for char, py, tone_symbol, rhythm_symbol in zip(text, pinyin, tone, rhythm):
        if char in PUNCTUATION or py in PUNCTUATION:
            sentence_order += 1
            continue
        if tone_symbol not in TONE_IDS:
            raise ValueError(f'unknown CFE tone {tone_symbol!r}')
        initial, final = split_pinyin(py)
        if not final or not final[-1].isdigit():
            raise ValueError(f'cannot split CFE pinyin {py!r}')
        rhyme_symbol = 'n' if rhythm_symbol == '0' else rhythm_symbol
        if rhyme_symbol not in RHYTHM_IDS:
            rhyme_symbol = 'n'
        if initial:
            phones.append(initial)
            tones.append(TONE_IDS['无'])
            rhymes.append(RHYTHM_IDS['N'])
            orders.append(sentence_order)
        phones.append(final)
        tones.append(TONE_IDS[tone_symbol])
        rhymes.append(RHYTHM_IDS[rhyme_symbol])
        orders.append(sentence_order)
        phones.append('|')
        tones.append(TONE_IDS['空'])
        rhymes.append(RHYTHM_IDS['e'])
        orders.append(sentence_order)
        used.append({'char': char, 'pinyin': py, 'tone': tone_symbol,
                     'rhythm': rhythm_symbol, 'order': sentence_order})

    return phones, tones, rhymes, orders, used
