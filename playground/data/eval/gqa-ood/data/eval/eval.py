
from collections import defaultdict
from tqdm import tqdm
import argparse
import os.path
import glob
import json
import math




parser = argparse.ArgumentParser()
parser.add_argument('--tier',           default = "val",                     type = str,    help = "Tier, e.g. train, val")
parser.add_argument('--scenes',         default="{tier}_sceneGraphs.json",   type = str,    help = "Scene graphs file name format.")
parser.add_argument('--questions',      default="{tier}_questions.json", type = str,    help = "Questions file name format.")
parser.add_argument('--choices',        default="{tier}_choices.json",       type = str,    help = "Choices file name format.")
parser.add_argument('--predictions',    default="{tier}_predictions.json",   type = str,    help = "Answers file name format.")
parser.add_argument('--attentions',     default="{tier}_attentions.json",    type = str,    help = "Attentions file name format.")
parser.add_argument('--consistency',    action="store_true",        help = "True to compute consistency score (Need to provide answers to questions in val_all_questions.json).")
parser.add_argument('--grounding',      action="store_true",        help = "True to compute grounding score (If model uses attention).")
parser.add_argument('--objectFeatures', action="store_true",        help = "True for object-based attention (False for spatial).")
parser.add_argument('--mapSize',    default = 7,    type = int, help = "Optional, only to get attention score. Images features map size, mapSize * mapSize")
args = parser.parse_args()

print("Please make sure to use our provided visual features as gqadataset.org for better comparability. We provide both spatial and object-based features trained on GQA train set.")
print("In particular please avoid using features from https://github.com/peteanderson80/bottom-up-attention since they were trained on images contained in the GQA validation set and thus may give false scores improvement.\n")

if not args.consistency:
    print("Please consider using --consistency to compute consistency scores for entailed questions.")
    print("If you do so, please provide answers to all questions in val_all_questions.json.\n")

if not args.grounding:
    print("Please consider using --grounding to compute attention scores.")
    print("If you do so, please provide attention maps through --attentions.\n")




def loadFile(name):

    if os.path.isfile(name):
        with open(name) as file:
            data = json.load(file)

    elif os.path.isdir(name.split(".")[0]):
        data = {}
        chunks = glob.glob('{dir}/{dir}_*.{ext}'.format(dir = name.split(".")[0], ext = name.split(".")[1]))
        for chunk in chunks:
            with open(chunk) as file:
                data.update(json.load(file))
    else:
        raise Exception("Can't find {}".format(name))
    return data

print("Loading questions...")
questions = loadFile(args.questions.format(tier = os.path.basename(args.tier)))

print("Loading predictions...")
predictions = loadFile(args.predictions.format(tier = args.tier))
predictions = {p["questionId"]: p["prediction"] for p in predictions}

for qid in questions:
    if (qid not in predictions) and (args.consistency or questions[qid]["isBalanced"]):
        print("no prediction for question {}. Please add prediction for all questions.".format(qid))
        raise Exception("missing predictions")

attentions = None
if args.grounding:
    with open(args.attentions.format(tier = args.tier)) as attentionsFile:
        attentions = json.load(attentionsFile)
        attentions = {a["questionId"]: a["attention"] for a in attentions}

def toScore(b):
    return float(1 if b else 0)

def avg(l):
    if len(l) == 0:
        return 0
    return float(sum(l)) / len(l)

def wavg(l, w):
    if sum(w) == 0:
        return None
    return float(sum(l[i] * w[i] for i in range(len(l)))) / sum(w)

scores = {
    "accuracy": [],
    "binary": [],
    "open": [],
    "validity": [],
    "plausibility": [],
    "consistency": [],
    "accuracyPerStructuralType": defaultdict(list),
    "accuracyPerSemanticType": defaultdict(list),
    "accuracyPerLength": defaultdict(list),
    "accuracyPerSteps": defaultdict(list),
    "grounding": []
}

dist = {
    "gold": defaultdict(lambda: defaultdict(int)),
    "predicted": defaultdict(lambda: defaultdict(int))
}


def getWordsNum(question):
    return len(question["question"].split())


def getStepsNum(question):
    return len([c for c in question["semantic"] if not (any([o in "{}: {}".format(c["operation"], c["argument"])
        for o in ["exist", "query: name", "choose name"]]))])



def toSlice(strSlice):
    sliceLims = (int(n) for n in strSlice.split(':'))
    return apply(slice, sliceLims)



def intsFromSlice(strSlice):
    slice_obj = get_slice_obj(slicearg)
    return(range(slice_obj.start or 0, slice_obj.stop or -1, slice_obj.step or 1))


def belongs(element, group, question):
    if "Common" in question["types"]["detailed"]:
        group = ["color", "material", "shape"]

    return element in group


def updateConsistency(questionId, question, questions):
    inferredQuestions = [eid for eid in question["entailed"] if eid != questionId]

    if correct and len(inferredQuestions) > 0:
        cosnsitencyScores = []
        for eid in inferredQuestions:
            gold = questions[eid]["answer"]
            predicted = predictions[eid]
            score = toScore(predicted == gold)
            cosnsitencyScores.append(score)

        scores["consistency"].append(avg(cosnsitencyScores))




def yrange(c):
    return (c[1], c[3])

def xrange(c):
    return (c[0], c[2])

def length(r):
    if r is None:
        return 0
    return float(r[1] - r[0])

def size(c):
    return length(xrange(c)) * length(yrange(c))

def intersection(r1, r2):
    ir = (max(r1[0], r2[0]), min(r1[1], r2[1]))
    if ir[1] > ir[0]:
        return ir
    return None

def intersectionSize(c1, c2):
    return length(intersection(xrange(c1), xrange(c2))) * length(intersection(yrange(c1), yrange(c2)))

def intersectionRate(c1, c2):
    return float(intersectionSize(c1, c2)) / size(c1)


def getCell(i, j):
    edge = float(1) / args.mapSize
    return (edge * i, edge * j, edge * (i + 1), edge * (j + 1))


def getRegion(sceneGraph, objectId):
    obj = sceneGraph["objects"][objectId]
    x0 = float(obj["x"]) / sceneGraph["width"]
    y0 = float(obj["y"]) / sceneGraph["height"]
    x1 = float(obj["x"] + obj["w"]) / sceneGraph["width"]
    y1 = float(obj["y"] + obj["h"]) / sceneGraph["height"]
    return (x0, y0, x1, y1)

def computeGroundingScore(question, sceneGraph, attentionMap):


    regions = []


    regions += [getRegion(sceneGraph, pointer) for pointer in question["annotations"]["question"].values()]


    regions += [getRegion(sceneGraph, pointer) for pointer in question["annotations"]["fullAnswer"].values()]


    if any(("scene" in c) for c in question["semantic"]):
        regions.append((0, 0, 1, 1))


    if args.objectFeatures:
        cells = [((x0, y0, x1, y1), attention) for x0, y0, x1, y1, attention in cells]
    else:
        cells = [(getCell(i, j), attentionMap[i][j]) for i in range(args.mapSize) for j in range(args.mapSize)]


    scores = []
    for region in regions:
        for cell, attention in cells:
            scores.append(attention * intersectionRate(cell, region))
    return sum(scores)



def chiSquare(goldDist, predictedDist):
    sumScore, sumOverall = 0, 0

    for group in goldDist:
        score, overall = 0, 0

        for ans in goldDist[group]:
            e = goldDist[group][ans]
            o = predictedDist[group].get(ans, 0)
            score += ((float(o - e) ** 2) / e)
            overall += goldDist[group][ans]

        sumScore += score * overall
        sumOverall += overall

    avgScore = float(sumScore) / sumOverall

    return avgScore



for qid, question in tqdm(questions.items()):
    gold = question["answer"]
    predicted = predictions[qid]

    correct = (predicted == gold)
    score = toScore(correct)

    wordsNum = getWordsNum(question)
    stepsNum = getStepsNum(question)


    if question["isBalanced"]:


        scores["accuracy"].append(score)
        scores["accuracyPerLength"][wordsNum].append(score)
        scores["accuracyPerSteps"][stepsNum].append(score)
        scores["accuracyPerStructuralType"][question["types"]["structural"]].append(score)
        scores["accuracyPerSemanticType"][question["types"]["semantic"]].append(score)
        answerType = "open" if question["types"]["structural"] == "query" else "binary"
        scores[answerType].append(score)

        globalGroup = question["groups"]["global"]
        if globalGroup is not None:
            dist["gold"][globalGroup][gold] += 1
            dist["predicted"][globalGroup][predicted] += 1


scores["distribution"] = chiSquare(dist["gold"], dist["predicted"]) / 100


metrics = [
    "binary",
    "open",
    "accuracy",
    "consistency",
    "validity",
    "plausibility",
    "grounding",
    "distribution"
]

detailedMetrics = [
    ("accuracyPerStructuralType", "Accuracy / structural type"),
    ("accuracyPerSemanticType", "Accuracy / semantic type"),
    ("accuracyPerSteps", "Accuracy / steps number"),
    ("accuracyPerLength", "Accuracy / words number")
]

subMetrics = {
    "attr": "attribute",
    "cat": "category",
    "global": "scene",
    "obj": "object",
    "rel": "relation"
}

for k in metrics:
    if isinstance(scores[k], list):
        scores[k] = avg(scores[k]) * 100

for k, _ in detailedMetrics:
    for t in scores[k]:
        scores[k][t] = avg(scores[k][t]) * 100, len(scores[k][t])

print("")
for m in metrics:

    if m == "grounding" and not args.grounding:
        continue
    if m == "consistency" and not args.consistency:
        continue

    print("{title}: {score:.2f}{suffix}".format(title = m.capitalize(), score = scores[m],
        suffix = " (lower is better)" if m == "distribution" else "%"))

for m, mPrintName in detailedMetrics:
    print("")

    print("{}:".format(mPrintName))

    for t in sorted(list(scores[m].keys())):

        tName = t
        if isinstance(scores[k], list):
            tName = subMetrics.get(t, t).capitalize()

        print("  {title}: {score:.2f}{suffix} ({amount} questions)".format(title = tName,
            score = scores[m][t][0], suffix = "%", amount = scores[m][t][1]))
