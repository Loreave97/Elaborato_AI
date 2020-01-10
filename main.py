from __future__ import print_function
import numpy as np
import numpy.random as ran


def unique_vals(rows, col):
    return set([row[col] for row in rows])


def class_counts(rows):
    counts = {}
    for row in rows:
        label = row[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


class Question:
    def __init__(self, column, value):
        self.column = column
        self.value = value
        self.prob = 0

    def set_prob(self, prob):
        self.prob = prob

    def get_prob(self):
        return self.prob

    def match(self, example):
        val = example[self.column]
        return val == self.value

    def match_classifier(self, example, prob):
        val = example[self.column]
        if val == "NaN":
            p = ran.random()
            if p < prob:
                return True
            else:
                return False
        return val == self.value


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    question.set_prob(len(true_rows) / (len(true_rows) + len(false_rows)))
    return true_rows, false_rows


def gini(rows):
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl ** 2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_question = gain, question
    return best_gain, best_question


class Leaf:
    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)
    if gain < 0.03:
        return Leaf(rows)
    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)
    return Decision_Node(question, true_branch, false_branch)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match_classifier(row, node.question.get_prob()):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs


def function(label, counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = int(counts[lbl] / total * 100)
        if lbl == label:
            if probs[lbl] > 74:
                return int(1)
            return int(0)
    return 0


def delete_data(rows, prob):
    file_deleted = rows
    n_features = len(rows[0]) - 1
    for row in file_deleted:
        for col in range(n_features):
            p = 0.01*prob
            if ran.random() < p:
                row[col] = "NaN"
    return file_deleted


def test(training, testing):
    dt = build_tree(training)
    count_t, count = 0, 0
    for row in testing:
        count_t += function(row[-1], classify(row, dt))
        count += 1
    print("Test con 0% di dati mancanti associa: ", int(count_t / count * 100), "% di dati corretti")

    testing_a = delete_data(testing, 10)
    count_t, count = 0, 0
    for row in testing_a:
        count_t += function(row[-1], classify(row, dt))
        count += 1
    print("Test con 10% di dati mancanti associa: ", int(count_t / count * 100), "% di dati corretti")

    testing_b = delete_data(testing, 20)
    count_t, count = 0, 0
    for row in testing_b:
        count_t += function(row[-1], classify(row, dt))
        count += 1
    print("Test con 20% di dati mancanti associa: ", int(count_t / count * 100), "% di dati corretti")

    testing_c = delete_data(testing, 50)
    count_t, count = 0, 0
    for row in testing_c:
        count_t += function(row[-1], classify(row, dt))
        count += 1
    print("Test con 50% di dati mancanti associa: ", int(count_t / count * 100), "% di dati corretti")


def main():
    training = np.loadtxt(fname="chess_train.txt", dtype="str", delimiter=",")
    testing = np.loadtxt(fname="chess_test.txt", dtype="str", delimiter=",")
    test(training, testing)

    training = np.loadtxt(fname="tic_train.txt", dtype="str", delimiter=",")
    testing = np.loadtxt(fname="tic_test.txt", dtype="str", delimiter=",")
    test(training, testing)

    training = np.loadtxt(fname="nursery_train.txt", dtype="str", delimiter=",")
    testing = np.loadtxt(fname="nursery_test.txt", dtype="str", delimiter=",")
    test(training, testing)


if __name__ == '__main__':
    main()


