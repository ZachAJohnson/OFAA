



model = AverageAtomFactory.create_model('ThomasFermi', 13, 28, 1*eV, 3, 30, name='Aluminum', Npoints=3000, ignore_vxc=True)
model.solve(verbose=True)

model = AverageAtomFactory.create_model('NeutralPseudoAtom', 13, 28, 1*eV, 3, 30, name='Aluminum', Npoints=3000, Zstar_init=3.8, TFW=False, ignore_vxc=False)
model.solve(verbose=True)
