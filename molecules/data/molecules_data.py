from pathlib import Path
import os

base_path = Path(__file__).absolute().parent

def molecules(n):

    p = os.path.join(base_path, "training%d" % n)
    if not os.path.exists(p):
        raise Exception("Molecules of size %d are not hadled." % n)

    predicates ={}
    with open(os.path.join(base_path, "ontology.nmln")) as f:
        for line in f:
            k,v = line.split(":")
            predicates[k] = int(v)

    res = []
    for filename in os.listdir(p):
        if filename != "ontology":
            res.append(os.path.join(p, filename))

    return res, predicates

