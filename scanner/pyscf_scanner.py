import pyscf


class QED_CCSD_GradScanner(pyscf.lib.GradScanner):
    def __init__(self, egf, mol):
        self.egf = egf
        self.mol = mol
        self.verbose = mol.verbose
        self.stdout = mol.stdout

    @property
    def e_tot(self):
        return self.e_tot

    @e_tot.setter
    def e_tot(self, x):
        self.e_tot = x

    @property
    def converged(self):
        return True

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, pyscf.gto.MoleBase):
            assert mol_or_geom.__class__ == pyscf.gto.Mole
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        e, g = self.egf(mol)

        return e, g.T.ravel()
