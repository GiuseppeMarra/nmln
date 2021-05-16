from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdchem
import mme
import os
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
from rdkit.Chem import rdchem
import matplotlib.pyplot as plt
from PIL import Image
import scipy
import seaborn as sns




mol_per_line = 5







class MoleculesHandler():



    def __init__(self, num_atoms, ontology):
        self.ontology = ontology
        self.num_atoms = num_atoms
        self.atom_types = {p.name for p in self.ontology.predicates.values() if len(p.domains)==1}
        self.bond_types = {p.name for p in self.ontology.predicates.values() if len(p.domains)==2 and p.name != 'skipBond'}

        self.fol_to_RDKIT = {p.name: p.name.upper() for p in self.ontology.predicates.values()}
        self.fol_to_RDKIT["cl"] = "Cl"
        self.RDKIT_to_fol = {v:k for k,v in self.fol_to_RDKIT.items()}
    def smile2Fol(self, smile):

        mol = Chem.MolFromSmiles(smile)

        to_write = []
        if "aromatic" not in self.bond_types:
            Chem.Kekulize(mol, clearAromaticFlags=True)

        for a in mol.GetAtoms():
            sym = self.RDKIT_to_fol[str(a.GetSymbol())]

            aid = a.GetIdx()
            to_write.append("%s(%d)" % (sym, aid))

            if len(to_write) == 0: continue
            for b in mol.GetBonds():
                sid = b.GetBeginAtom().GetIdx()
                eid = b.GetEndAtom().GetIdx()
                type = self.RDKIT_to_fol[str(b.GetBondType())]
                to_write.append("%s(%d,%d)" % (type, sid, eid))
                to_write.append("%s(%d,%d)" % (type, eid, sid))
            print(to_write)



    def fromLin2Mol(self, Y):
        MOLS = []
        for linear in Y:
            d = self.ontology.linear_to_fol_dictionary(linear)
            try:
                mol = self.fromFol2Mol(d)
            except Exception as e:
                print(e)
            if mol is not None:
                MOLS.append(mol)
        return MOLS

    def fromFol2Mol(self, d, sanitize=True):

        mol = Chem.RWMol()
        node_to_idx = {}


        for i in range(self.num_atoms):

            S = None
            for s in self.atom_types:
                if d[s][i] == 1:
                    S = s
                    break
            a = Chem.Atom(self.fol_to_RDKIT[S])

            idx = mol.AddAtom(a)
            node_to_idx[i] = idx

        for i in range(self.num_atoms):
            for j in range(i, self.num_atoms):
                if i not in node_to_idx or j not in node_to_idx:
                    continue
                for b in self.bond_types:
                    if d[b][i, j] == 1:
                        try:
                            ifirst = node_to_idx[i]
                            isecond = node_to_idx[j]
                            bond_type = rdchem.BondType.names[self.fol_to_RDKIT[b]]
                            mol.AddBond(ifirst, isecond, bond_type)
                        except:
                            pass
                        break


        mol = mol.GetMol()
        if sanitize:
            mol = AdjustAromaticNs(mol)
            mol.UpdatePropertyCache(False)
            for a in mol.GetAtoms():
                a.UpdatePropertyCache()
            Chem.Kekulize(mol)
            Chem.SanitizeMol(mol)

        return mol

    def get_train_molecules(self, folder, n_molecules=None, get_linear = False):

        Y = []
        for filename in os.listdir(folder):
            y = self.ontology.FOL2LinearState(os.path.join(folder, filename))
            Y.append(y)

        print(len(Y))
        if n_molecules is None:
            n_molecules = len(Y)

        from random import shuffle
        shuffle(Y)
        Y = Y[:n_molecules]

        MOLS = []
        MOLS_lin = []


        for linear in Y:
            d = self.ontology.linear2Dict(linear)
            mol = self.fromFol2Mol(d)
            if mol is not None:
                MOLS.append(mol)
                MOLS_lin.append(linear[0].tolist())
            if len(MOLS)==n_molecules:
                break
        if not  get_linear:
            return MOLS
        else:
            return MOLS, MOLS_lin

    def get_molecules(self, n_molecules, file, skip=1, get_linear = False):

        MOLS = []
        MOLS_lin = []
        with open(file) as f:
            lines = f.readlines()

        lines = lines[-200000:-1]

        # from random import shuffle
        lines = lines[::-skip]
        count = 0
        for line in lines:
            linear = [int(float(s)) for s in line.split(",")]
            d = self.ontology.linear2Dict(linear)
            mol = self.fromFol2Mol(d)
            if mol is not None:
                MOLS.append(mol)
                MOLS_lin.append(linear)
            if len(MOLS)==n_molecules:
                break
            count+=1
        print("Valid samples: %f" % (len(MOLS) / float(count)))
        if not get_linear:
            return MOLS
        else:
            return MOLS, MOLS_lin

    def get_linear_molecules_generated(self, file, n_molecules=None):


        MOLS = []

        with open(file) as f:
            lines = f.readlines()

        if n_molecules is None:
            n_molecules = len(lines)

        from random import shuffle
        shuffle(lines)
        for line in lines:
            linear = [int(float(s)) for s in line.split(",")]
            MOLS.append(linear)
            if len(MOLS)==n_molecules:
                break
        return MOLS

    def get_linear_molecules_train(self, folder, n_molecules=None):

        Y = []
        for filename in os.listdir(folder):
            y = self.ontology.FOL2LinearState(os.path.join(folder, filename))
            Y.append(y)

        if n_molecules is None:
            n_molecules = len(Y)

        from random import shuffle
        shuffle(Y)
        Y = Y[:n_molecules]
        return Y

    def print_molecules(self, file_name, molecules, num=20, mol_per_line=5):
        MOLS = molecules
        MOLS = np.take(MOLS, np.random.choice(range(len(MOLS)), size=num), axis=0).tolist()
        imgsvg = Draw.MolsToGridImage(MOLS, molsPerRow=mol_per_line, subImgSize=(200, 200), useSVG=True)
        with open("{}.svg".format(file_name), "w") as f:
            f.write(imgsvg)
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
        drawing = svg2rlg("{}.svg".format(file_name))
        renderPDF.drawToFile(drawing, "{}.pdf".format(file_name))

    def show_molecules(self, MOLS):
        imgs = []
        for i in range(len(MOLS) // mol_per_line):
            img = Draw.MolsToImage(MOLS[i * mol_per_line:(i + 1) * mol_per_line], subImgSize=(200, 200))
            imgs.append(img)
        imgs_comb = np.vstack((np.asarray(i) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save("gen.pdf")
        imgs_comb.show()

    def show_molecule(self, mol):
        # img = Draw.MolsToImage(mol)
        Draw.MolToImage(mol)
        plt.show()

    def save_molecule(self, mol, i, q):
        # img = Draw.MolsToImage(mol)
        print("saving_molecule")
        img = Draw.MolToImage(mol)
        try:
            img.save("molecules/mol%d/%d.jpg" % (i, q))
        except Exception as e:
            print(e)

    def plotHistComparison(self, f, MOLS_gen, MOLS_true, bins, bw, ax=None, title=None):
        g = f(MOLS_gen)
        t = f(MOLS_true)

        # sns.distplot(g, hist=False, kde=True,
        #              bins=int(180 / 5), color='darkblue', label="gen")

        # bins = np.arange(0,20,0.5)
        # bins = None
        sns.distplot(g, hist=True, kde=False,
                     kde_kws={"bw": bw},
                     hist_kws={"histtype": "step", "linewidth": 2, "alpha": 0.8},
                     color='blue',
                     bins=bins,
                     label="generated", ax=ax)

        sns.distplot(t, hist=True, kde=False,
                     kde_kws={"bw": bw},
                     hist_kws={"histtype": "step", 'linestyle': '--', "linewidth": 2, "alpha": 0.8},
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





def _FragIndicesToMol(oMol, indices):
    em = Chem.EditableMol(Chem.Mol())

    newIndices = {}
    for i, idx in enumerate(indices):
        em.AddAtom(oMol.GetAtomWithIdx(idx))
        newIndices[idx] = i

    for i, idx in enumerate(indices):
        at = oMol.GetAtomWithIdx(idx)
        for bond in at.GetBonds():
            if bond.GetBeginAtomIdx() == idx:
                oidx = bond.GetEndAtomIdx()
            else:
                oidx = bond.GetBeginAtomIdx()
            # make sure every bond only gets added once:
            if oidx < idx:
                continue
            em.AddBond(newIndices[idx], newIndices[oidx], bond.GetBondType())
    res = em.GetMol()
    res.ClearComputedProps()
    Chem.GetSymmSSSR(res)
    res.UpdatePropertyCache(False)
    res._idxMap = newIndices
    return res


def _recursivelyModifyNs(mol, matches, indices=None):
    if indices is None:
        indices = []
    res = None
    while len(matches) and res is None:
        tIndices = indices[:]
        nextIdx = matches.pop(0)
        tIndices.append(nextIdx)
        nm = Chem.Mol(mol)
        nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
        nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
        cp = Chem.Mol(nm)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            res, indices = _recursivelyModifyNs(nm, matches, indices=tIndices)
        else:
            indices = tIndices
            res = cp
    return res, indices


def AdjustAromaticNs(m, nitrogenPattern='[n&D2&H0;r5,r6]'):
    """
       default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
       to fix: O=c1ccncc1
    """
    Chem.GetSymmSSSR(m)
    m.UpdatePropertyCache(False)

    # break non-ring bonds linking rings:
    em = Chem.EditableMol(m)
    linkers = m.GetSubstructMatches(Chem.MolFromSmarts('[r]!@[r]'))
    plsFix = set()
    for a, b in linkers:
        em.RemoveBond(a, b)
        plsFix.add(a)
        plsFix.add(b)
    nm = em.GetMol()
    for at in plsFix:
        at = nm.GetAtomWithIdx(at)
        if at.GetIsAromatic() and at.GetAtomicNum() == 7:
            at.SetNumExplicitHs(1)
            at.SetNoImplicit(True)

    # build molecules from the fragments:
    fragLists = Chem.GetMolFrags(nm)
    frags = [_FragIndicesToMol(nm, x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    ok = True
    for i, frag in enumerate(frags):
        cp = Chem.Mol(frag)
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            matches = [x[0] for x in frag.GetSubstructMatches(Chem.MolFromSmarts(nitrogenPattern))]
            lres, indices = _recursivelyModifyNs(frag, matches)
            if not lres:
                # print 'frag %d failed (%s)'%(i,str(fragLists[i]))
                ok = False
                break
            else:
                revMap = {}
                for k, v in frag._idxMap.iteritems():
                    revMap[v] = k
                for idx in indices:
                    oatom = m.GetAtomWithIdx(revMap[idx])
                    oatom.SetNoImplicit(True)
                    oatom.SetNumExplicitHs(1)
    return m


def fix_mol(m):
    if m is None:
        return None
    try:
        m.UpdatePropertyCache(False)
        cp = Chem.Mol(m.ToBinary())
        Chem.SanitizeMol(cp)
        m = cp
        print
        'fine:', Chem.MolToSmiles(m)
        return m
    except ValueError:
        print
        'adjust'
        nm = AdjustAromaticNs(m)
        if nm is not None:
            Chem.SanitizeMol(nm)
            print
            'fixed:', Chem.MolToSmiles(nm)
        else:
            print
            'still broken'
        return nm

def computeProb(l):
    h, b = scipy.histogram(l)
    pmf_unnorm = h + 1
    pmf_unnorm = pmf_unnorm.astype(np.float64)
    h = pmf_unnorm / sum(pmf_unnorm)
    return h


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
    sns.distplot(g, hist=True, kde=False,
                 kde_kws={ "bw" : bw},
                 hist_kws={"histtype": "step", "linewidth": 2, "alpha":0.8},
                 color='blue',
                 bins=bins,
                 label="generated", ax=ax)

    sns.distplot(t, hist=True, kde=False,
                 kde_kws={"bw" : bw},
                 hist_kws={"histtype": "step", 'linestyle':'--', "linewidth": 2, "alpha":0.8},
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



# Removing RDKit logs for the moment
from rdkit import RDLogger,Chem

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)



class MoleculesLogger():


    def __init__(self, log_file, ontology, handler: MoleculesHandler, verbose = False, buffer_size = 20):

        self.mols = []
        self.handler = handler
        self.ontology = ontology
        self.buffer_size = buffer_size
        self.log_file = log_file
        self.verbose = verbose

    def __call__(self, molecules):

        molecules = np.reshape(molecules, [-1, self.ontology.linear_size()])
        for t in molecules:
            d = self.ontology.linear_to_fol_dictionary(t)
            try:
                mol = self.handler.fromFol2Mol(d)
                self.mols.append(mol)
                # handler.show_molecule(mol)
            except Exception as e:
                if self.verbose:
                    print(e)

        if len(self.mols) > self.buffer_size:
            with open(self.log_file, "a") as f_sml:
                for q, mol in enumerate(self.mols):
                    try:
                        sml = Chem.MolToSmiles(mol)
                        if self.verbose: print(sml)
                        f_sml.write(sml)
                        f_sml.write("\n")
                    except Exception as e:
                        if self.verbose: print(e)
                    self.mols = []






