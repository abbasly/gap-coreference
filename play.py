from __future__ import division, print_function

import argparse
import csv
from collections import defaultdict

import nltk

from constants import GOLD_FIELDNAMES, PRONOUNS, SYSTEM_FIELDNAMES, Gender
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import CategoricalNB
from mixed_naive_bayes import MixedNB

import numpy as np


class Annotation(object):
    """Container class for storing annotations of an example.

    Attributes:
      gender(None): The gender of the annotation. None indicates that gender was
        not determined for the given example.
      name_a_coref(None): bool reflecting whether Name A was recorded as
        coreferential with the target pronoun for this example. None indicates
        that no annotation was found for the given example.
      name_b_coref(None): bool reflecting whether Name B was recorded as
        coreferential with the target pronoun for this example. None indicates
        that no annotation was found for the given example.
    """

    def __init__(self):
        self.text = None
        self.A = None
        self.B = None
        self.pro = None
        self.gender = None
        self.name_a_coref = None
        self.name_b_coref = None
        self.a_offset = None
        self.b_offset = None
        self.pro_offest = None


class Features:
    def __init__(self):
        self.dist_a_pro = None
        self.dist_b_pro = None
        self.pro_a = None
        self.pro_b = None
        self.count_a = None
        self.count_b = None
        self.name_a_coref = None
        self.name_b_coref = None
        self.label = None


def read_annotations(filename, is_gold):
    """Reads coreference annotations for the examples in the given file.

    Args:
      filename: Path to .tsv file to read.
      is_gold: Whether or not we are reading the gold annotations.

    Returns:
      A dict mapping example ID strings to their Annotation representation. If
      reading gold, 'Pronoun' field is used to determine gender.
    """

    def is_true(value):
        # print(value)
        if value.lower() == "true":
            return True
        elif value.lower() == "false":
            return False
        else:
            print("Unexpected label!", value)
            return None

    fieldnames = GOLD_FIELDNAMES if is_gold else SYSTEM_FIELDNAMES

    annotations = defaultdict(Annotation)
    feats = defaultdict(Features)
    with open(filename, "rU") as f:
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter="\t")

        # Skip the header line in the gold data
        if is_gold:
            next(reader, None)

        for row in reader:
            example_id = row["ID"]
            if example_id in annotations:
                print("Multiple annotations for", example_id)
                continue

            annotations[example_id].name_a_coref = is_true(row["A-coref"])
            annotations[example_id].name_b_coref = is_true(row["B-coref"])
            annotations[example_id].A = row["A"]
            annotations[example_id].B = row["B"]
            feats[example_id].name_a_coref = is_true(row["A-coref"])
            feats[example_id].name_b_coref = is_true(row["B-coref"])
            annotations[example_id].a_offset = int(row["A-offset"])
            annotations[example_id].b_offset = int(row["B-offset"])
            annotations[example_id].pro_offset = int(row["Pronoun-offset"])
            annotations[example_id].pro = row["Pronoun"]
            annotations[example_id].text = row["Text"]

            if is_gold:
                gender = PRONOUNS.get(row["Pronoun"].lower(), Gender.UNKNOWN)
                assert gender != Gender.UNKNOWN, row
                annotations[example_id].gender = gender
    return annotations, feats


def preprocess(sents):
    sentences = nltk.sent_tokenize(sents)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # sentences = [nltk.pos_tag(sent) for sent in sentences]
    return sentences


def order(a: int, b: int):
    if a <= b:
        return a, b
    return b, a


def in_the_same_sent(sent, pro, A, B):
    pro_num = 0
    a_num = 0
    b_num = 0
    sentences = preprocess(sent)
    for idx, sent in enumerate(sentences):
        if pro.lower() in ("".join(sent)).lower():
            pro_num = idx
        if A in "".join(sent):
            a_num = idx
        if B in "".join(sent):
            b_num = idx
        if pro_num == a_num:
            x = 0
        else:
            x = 1
        if pro_num == b_num:
            y = 0
        else:
            y = 1
    return (x, y)


def count_occ(text, word):
    return text.count(word)


def retrieve_data(file_name):
    annots, feats = read_annotations(file_name, True)

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
        feats[key].pro_a = in_the_same_sent(annot.text, annot.pro, annot.A, annot.B)[0]
        feats[key].pro_b = in_the_same_sent(annot.text, annot.pro, annot.A, annot.B)[1]
        feats[key].count_a = count_occ(annot.text, annot.A)
        feats[key].count_b = count_occ(annot.text, annot.B)

    with open("test.tsv", "wt") as out_file:
        tsv_writer = csv.writer(out_file, delimiter="\t")
        tsv_writer.writerow(
            [
                "ID",
                "Dist_a_pro",
                "Dist_b_pro",
                "A_coref",
                "B_coref",
                "Count_A",
                "Count_B",
                "Same_with_A",
                "Same_with_B",
                "Label",
            ]
        )

        for key in feats:
            tsv_writer.writerow(
                [
                    key,
                    feats[key].dist_a_pro,
                    feats[key].dist_b_pro,
                    feats[key].name_a_coref,
                    feats[key].name_b_coref,
                    feats[key].count_a,
                    feats[key].count_b,
                    feats[key].pro_a,
                    feats[key].pro_b,
                    feats[key].label,
                ]
            )

    file = open("test.tsv", "r")
    data = csv.reader(file, delimiter="\t")
    table = [row for row in data]
    nparr = np.asarray(table[1:])

    nparr = nparr[:, [1, 2, 5, 6, 7,8,9]]
    nparr = nparr.astype(int)
    X_train = nparr[:, [0, 1, 2, 3,4,5]]
    Y_train = nparr[:, [6]]
    return X_train, Y_train


X_train, Y_train = retrieve_data("gap-development.tsv")
X_test, Y_test = retrieve_data("gap-test.tsv")
print(X_test[:5])
print(len(X_test))
gnb = GaussianNB()

# clf = CategoricalNB()
# clf.fit(X_train, Y_train)

# clf = LogisticRegression(random_state=0).fit(X_train, Y_train)

# clf = MixedNB(categorical_features=[2,3])
# clf.fit(X_train,np.ravel(Y_train))
# y_pred = clf.predict(X_test)

y_pred = gnb.fit(X_train, np.ravel(Y_train)).predict(X_test)
# y_pred = clf.predict(X_test)
print(y_pred)

print(y_pred[50])

num_to_bool = {0: (False, False), 1: (True, False), 2: (False, True), 3: (True, True)}

print(len(y_pred))
with open("output.tsv", "wt") as out_file:
    tsv_writer = csv.writer(out_file, delimiter="\t")
    tsv_writer.writerow(["ID", "A_coref", "B_coref"])
    for idx, elem in enumerate(y_pred):
        id = "test-" + str(idx + 1)
        tsv_writer.writerow(
            [id, num_to_bool[y_pred[idx]][0], num_to_bool[y_pred[idx]][1]]
        )
