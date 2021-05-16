from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import seaborn as sns
from rdkit.Chem.rdMolDescriptors import CalcNumHBA,CalcNumHBD, CalcTPSA, CalcNumRings
import mme
from molecules.utils import MoleculesHandler
from molecules.data.molecules_data import molecules

def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions

    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
        Discrete probability distributions.
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def print_molecules(file_name, molecules, num=20, mol_per_line=5, random=False):
     MOLS = molecules
     if random:
        MOLS = np.take(MOLS,np.random.choice(range(len(MOLS)), size=num), axis=0).tolist()
     else:
        MOLS = MOLS[:num]
     imgsvg = Draw.MolsToGridImage(MOLS, molsPerRow=mol_per_line, subImgSize=(200, 200), useSVG=True)
     with open("{}.svg".format(file_name),"w") as f:
         f.write(imgsvg)
     from svglib.svglib import svg2rlg
     from reportlab.graphics import renderPDF                
     drawing = svg2rlg("{}.svg".format(file_name))
     renderPDF.drawToFile(drawing, "{}.pdf".format(file_name))

def show_molecules(MOLS):
    imgs = []
    for i in range(len(MOLS) // mol_per_line):
        img = Draw.MolsToImage(MOLS[i * mol_per_line:(i + 1) * mol_per_line], subImgSize=(200, 200))
        imgs.append(img)
    imgs_comb = np.vstack((np.asarray(i) for i in imgs))
    imgs_comb = Image.fromarray(imgs_comb)
    imgs_comb.save("gen.pdf")
    imgs_comb.show()


def computeProb(l):
    h, b = scipy.histogram(l)
    pmf_unnorm = h + 1
    pmf_unnorm = pmf_unnorm.astype(np.float64)
    h = pmf_unnorm / sum(pmf_unnorm)
    return h


def computeKLOf(f, true_mols, gen_mols):

    count_true = f(true_mols)
    count_gen = f(gen_mols)
    p_gen = computeProb(count_gen)
    q_true = computeProb(count_true)
    return kl(p_gen, q_true)



def numAtoms(MOLS):

    num_atoms = []
    for mol in MOLS:
        num_atoms.append(mol.GetNumAtoms())
    return num_atoms

def numBonds(MOLS):

    num_bonds = []
    for mol in MOLS:
        num_bonds.append(mol.GetNumBonds())
    return num_bonds

def avgNodeDegree(MOLS):

    num_bonds = []
    for mol in MOLS:
        avg = []
        for a in mol.GetAtoms():
            avg.append(a.GetDegree())
        num_bonds.append(float(sum(avg))/(float(len(avg))+1e-12))
    return num_bonds

def plotHistComparison(f,MOLS_gen, MOLS_true, bins, bw, ax=None, title=None):
    g = f(MOLS_gen)
    t = f(MOLS_true)

    # sns.distplot(g, hist=False, kde=True,
    #              bins=int(180 / 5), color='darkblue', label="gen")

    # bins = np.arange(0,20,0.5)
    # bins = None
    # sns.histplot(g, hist=True, kde=False,
    #              kde_kws={ "bw" : bw},
    #              hist_kws={"histtype": "step", "linewidth": 2, "alpha":0.8},
    #              color='blue',
    #              bins=bins,
    #              label="generated", ax=ax)

    # sns.distplot(t, hist=True, kde=False,
    #              kde_kws={"bw" : bw},
    #              hist_kws={"histtype": "step", 'linestyle':'--', "linewidth": 2, "alpha":0.8},
    #              color='red',
    #              bins=bins,
    #              label="training", ax=ax)

    sns.histplot(g,
                 element="step",
                 stat="density",
                 linewidth= 2,
                 alpha=0.1,
                 color='blue',
                 bins=bins,
                 label="generated", ax=ax)

    sns.histplot(t,
                 element="step",
                 stat="density",
                 linewidth= 2,
                 linestyle = '--',
                 alpha=0.1,
                 color='red',
                 bins=bins,
                 label="training", ax=ax)

    # plt.hist(g,bins =b,alpha=0.5,label="gen")
    # plt.hist(t,bins =b,alpha=0.5,label="true")
    # plt.legend(loc="upper left")
    title = f.__name__ if title is None else title
    if ax is None:
        plt.title(title)
        plt.show()
    else:
        # ax.legend()
        ax.set_title(title)


def pick_last_n(folder, n):
    MOLS_gen_smiles = [Chem.CanonSmiles(str(line).strip()) for line in open("%s/generated_smiles.csv" % folder)]
    MOLS_gen = [Chem.MolFromSmiles(sml) for sml in MOLS_gen_smiles]
    MOLS_gen = MOLS_gen[-n:]
    return MOLS_gen, MOLS_gen_smiles

def pick_most_frequent_n(folder, n):
    from collections import Counter
    MOLS_gen_smiles = [Chem.CanonSmiles(str(line).strip()) for line in open("%s/generated_smiles.csv" % folder)]
    MOLS_gen_smiles = Counter(MOLS_gen_smiles)
    MOLS_gen_smiles = [sml[0] for sml in MOLS_gen_smiles.most_common(n*10)]
    MOLS_gen = [Chem.MolFromSmiles(sml) for sml in MOLS_gen_smiles]
    return MOLS_gen, MOLS_gen_smiles

def how_many_are_known(MOLS_gen_smiles, MOLS_true_set):
    not_in_train = []
    for n_m_f in [100, 500, 1000, 2000, 10000]:
        m_g = MOLS_gen_smiles[:n_m_f]
        c = 0
        for i in m_g:
            if i in MOLS_true_set:
                c += 1
            else:
                not_in_train.append(i)
    print("Known (most frequent %d):" % n_m_f, float(c) / n_m_f)




# SOME VARS
skip = 1
n_molecules = 1000
mol_per_line = 10
MAX_NUM_ATOMS = 8
# folder  = 'molecules'
# folder  = 'molecules2'
folder  = 'previous_results/9'


# CREATING ONTOLOGY
filenames, predicates = molecules(MAX_NUM_ATOMS)
"""Knowledge Representation"""
constants = [str(i) for i in range(MAX_NUM_ATOMS)]
d = mme.Domain(name="atoms", constants=constants, num_constants=len(constants))
predicates = [mme.Predicate(p, domains=[d for _ in range(a)]) for p, a in predicates.items()]
o = mme.Ontology(domains=[d], predicates=predicates)
handler = MoleculesHandler(MAX_NUM_ATOMS, o)



"""Data Loading. FOL descriptions are serialized into flat tensors"""
Y = []
for filename in filenames:
    y = o.file_to_linearState(filename)
    Y.append(y)


MOLS_true = handler.fromLin2Mol(Y)
MOLS_true_smiles = [Chem.CanonSmiles(line.strip()) for line in open("smiles8.txt")]
MOLS_true_set = set(MOLS_true_smiles)
# print_molecules(file_name="true", molecules=MOLS_true, num=15, mol_per_line=5, random=True)
y = Y


# MOLS_gen, MOLS_gen_smiles = pick_last_n(folder,len(MOLS_true))
MOLS_gen, MOLS_gen_smiles = pick_most_frequent_n(folder,len(MOLS_true))
MOLS_gen_set = set(MOLS_gen_smiles)
# print_molecules(file_name="gen", molecules=MOLS_gen, num=20, mol_per_line=5)
# show_molecules(MOLS_gen)


not_in_train = [Chem.MolFromSmiles(i) for i in MOLS_gen_smiles if i not in MOLS_true_set]
with open("100_not_in_train.txt", "w") as f: f.writelines([i+"\n" for i in MOLS_gen_smiles if i not in MOLS_true_set])
print_molecules("100_not_in_train", not_in_train, num=100, mol_per_line=10)
exit()
# show_molecules(MOLS_true[:n_molecules])
# for i, mol in enumerate(MOLS_true):
#     Draw.MolToFile(mol, "printed/mol%d.png" % i)
# exit()


fig = plt.figure(figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

def calcNumHBD(x):
    return [CalcNumHBD(a) for a in x]

def calcNumRings(x):
    return [CalcNumRings(a) for a in x]

def calcNumHBA(x):
    return  [CalcNumHBA(a) for a in x]

def calcTPSA(x):
      return[CalcTPSA(a) for a in x]


fs = [#(numAtoms,np.arange(0,12,1),0.5),
      (numBonds,np.arange(0,20,1),0.5, "Number of Bonds"),
      (avgNodeDegree,np.arange(0,5,0.5),0.5, "Average Node Degree"),
    (calcNumRings, np.arange(0, 4, 1), 0.5, "Number of Rings"),
    (calcNumHBD,np.arange(0,7,1),0.5, "HBD"),
      (calcNumHBA,np.arange(0,10,1),0.5, "HBA"),
      (calcTPSA,np.arange(0,100,10),5., "TPSA")
      ]
axs=[]
for i,f in enumerate(fs):
    # print("KL divergence distribution" + str(computeKLOf(numAtoms,MOLS_true, MOLS_gen)))
    ax = plt.subplot(2, 3, i+1)
    plotHistComparison(f[0], MOLS_gen=MOLS_gen, MOLS_true=MOLS_true, bins=f[1], bw=f[2],ax=ax, title=f[3])
    axs.append(ax)


handles, labels = ax.get_legend_handles_labels()
plt.figlegend(handles, labels, loc = 'lower center', ncol=5, labelspacing=0.)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# plt.tight_layout(pad=2.5)
plt.savefig("stats.pdf")


plt.show()




