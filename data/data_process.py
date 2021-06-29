import json
from tqdm import tqdm
import glob
import re


# def text_process(text):
#     tokens = [t.text for t in tokenizer.tokenize(text)]
#     new_tokens = [tokens[0]]
#     labels = [0]
#     for i, token in enumerate(tokens[1:]):
#         if token not in punc_dic:
#             new_tokens.append(token)
#             labels.append(0)
#         else:
#             labels[-1] = punc_dic[token]
#
#     return new_tokens, labels


def no_date(s):
    for char in s:
        if (char == '年'):
            return False
        if (char == '月'):
            return False
        if (char == '日'):
            return False
    return True


def filter_simple(text):
    # 过滤掉只有句号的句子
    count = 0
    for punc in punc_dic:
        if punc in text:
            count += 1
            if count >= 1:
                return True
    return False


def clean(text):
    res = []
    for i in range(len(text)):
        if i > 0 and text[i] == text[i - 1] and text[i] in punc_dic:
            continue
        else:
            res.append(text[i])
    return ''.join(res)

f1 = open('cleanwiki.txt', 'w', encoding='utf8')
punc_dic = {'，','。','；','？','！'}
for file_path in tqdm(glob.glob('wiki_zh/*/wiki*')):
    r1 = '[a-zA-Z’"#$%&\'()*+-/:：<=>@★…【】_-—℃％¥℉°（）·「」『』 《》 “”‘’[\\]^_`{|}~]+'
    paras = [json.loads(x)['text'].split('\n') for x in open(file_path, encoding='utf8')]
    for para in paras:
        for line in para:
            cleanline = clean(re.sub(r1,'',line))
            if(len(cleanline)>20) and len(cleanline)<=500 and no_date(cleanline) and filter_simple(line):
                f1.write(cleanline+'\n')
#     break
f1.close()