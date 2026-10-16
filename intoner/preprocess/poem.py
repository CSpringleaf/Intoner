import numpy as np

pingze = '_平上去入无空'
rhythm = '_1234nNe'
# Mappings from symbol to numeric ID and vice versa:
_tone_to_id = {s: i for i, s in enumerate(list(pingze))}
_id_to_tone = {i: s for i, s in enumerate(list(pingze))}
_rhythm_to_id = {s: i for i, s in enumerate(list(rhythm))}
_id_to_rhythm = {i: s for i, s in enumerate(list(rhythm))}

def to_sequence(text,tone,rhythm):
    tone,rhythm = align(text,tone,rhythm)
    tone_sequence = []
    for i in range(len(tone)):
        tone_sequence.append(_tone_to_id[tone[i]])
    rhythm_sequence = []
    for i in range(len(rhythm)):
        rhythm_sequence.append(_rhythm_to_id[rhythm[i]])
    return np.array(tone_sequence),np.array(rhythm_sequence)

def align(text, tone, rhythm):#align tones and rhythms from word level to phoneme level
    rhythm = rhythm.replace('0','n')
    phoneme_list = text.split(' ')
    new_tone = ''
    new_rhythm = ''
    n_word = 0
    for i in phoneme_list:
        if i in '，。？！《》“”‘’、：；':
            continue
        if '1' in i or '2' in i or '3' in i or '4' in i:
            new_tone += tone[n_word]
            new_rhythm += rhythm[n_word]
            n_word += 1
        elif i == '|':
            new_tone += '空'
            new_rhythm += 'e'
        else:
            new_tone += '无'
            new_rhythm += 'N'
    return new_tone, new_rhythm

def number(raw_text,num=None):
    text = raw_text.split(' ')
    if num:
        return np.array([int(num) for i in range(len(text))])
    else:
        pre = 1
        id = []
        for i in text:
            # print(i,pre)
            id.append(pre)
            if i in '，。？！：；、':
                pre += 1
        # print(id)
        return id
