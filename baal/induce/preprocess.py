def get_trees(infile, use_tqdm=False):
    # trying things WITH root
    #fix = lambda y: [x.replace("\n","").replace("ROOT","").strip() for x in y]
    fix = lambda y: [x.replace("\n","").strip() for x in y]
    with open(infile) as fp:
        capacitor = []
        is_consuming = True
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(enumerate(fp), desc=' lines')
        else:
            it = enumerate(fp)
        for i, line in it:

            # catching shitty noise
            if line[0]!="(" and not capacitor:
                continue

            if line[0]=="(" and capacitor:
                yield "".join(fix(capacitor))
                capacitor = []

            capacitor.append(line)
    if len(capacitor)>0:
        yield "".join(fix(capacitor))
    raise StopIteration