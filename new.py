from play import *

annots, feats = read_annotations("gap-test.tsv", True)


def tuple_to_int(bool1, bool2):
    if (bool1, bool2) == (False, False):
        return 0
    if (bool1, bool2) == (True, False):
        return 1
    if (bool1, bool2) == (False, True):
        return 2
    if (bool1, bool2) == (True, True):
        return 3


for key in annots:
    annot = annots[key]
    sentence = annot.text
    ord1, ord2 = order(annot.a_offset, annot.pro_offset)
    feats[key].dist_a_pro = len(sentence[ord1:ord2].split())
    ord1, ord2 = order(annot.b_offset, annot.pro_offset)
    feats[key].dist_b_pro = len(sentence[ord1:ord2].split())
    feats[key].label = tuple_to_int(annot.name_a_coref, annot.name_b_coref)

print(annots["test-1"].A)
print(feats["test-1"].name_a_coref)

sentence = annots["test-1"].text

sentence = preprocess(sentence)
for elem in sentence:
    if "His" in elem:
        print("yes")
    else:
        print("No")
print(sentence)


def in_the_same_sent(sent, pro, A, B):
    pro_num = 0
    a_num = 0
    b_num = 0
    sentences = preprocess(sent)
    for idx, sent in enumerate(sentences):
        if pro.lower() in (" ".join(sent)).lower():
            pro_num = idx
        if A in " ".join(sent):
            a_num = idx
        if B in " ".join(sent):
            b_num = idx
    return (pro_num == a_num, pro_num == b_num)


print(annots["test-3"].A)
print(annots["test-3"].B)
print(annots["test-3"].pro)
print(preprocess(annots["test-3"].text))
print(
    in_the_same_sent(
        annots["test-2"].text,
        annots["test-2"].pro,
        annots["test-2"].A,
        annots["test-2"].B,
    )
)
