import numpy as np
from pathlib import Path
import logging


def project(origin_xyz, others_xyz, selected_indices):
    """
    projects others_xyz to MNI coordiantes given anchors in origin_xyz
    :param origin_xyz: anchors given as nx3 np array (n >= 4)
    :param others_xyz: optodes to project given as mx3 np array (m>=1)
    :param selected_indices: which indices to select from origin_xyz as anchors given as np array (len must be at least 4)
                             order matters! selection is based on this order:
                             ["nosebridge", "inion", "rightear", "leftear",
                              "fp1", "fp2", "fz", "f3",
                              "f4", "f7", "f8", "cz",
                              "c3", "c4", "t3", "t4",
                              "pz", "p3", "p4", "t5",
                              "t6", "o1", "o2"]
    :return: otherH - others transformed to MNI of ideal head (head surface)
             otherC - others transformed to MNI of ideal head  (cortical surface)
             otherHSD - transformation standard deviation per axis, point manner (for otherH).
                        Last channel is SD across all axes (root sum of squares).
             otherCSD - transformation standard deviation per axis, point manner (for otherC).
                        Last channel is SD across all axes (root sum of squares).
    """
    path_wo_ext = "resource/MNI_templates/DMNIHAve"
    dmnihavePath = Path(path_wo_ext + ".csv")
    if Path(path_wo_ext+".npy").is_file():
        DMNIHAve = np.load(path_wo_ext+".npy", allow_pickle=True)
    else:
        DMNIHAve = np.genfromtxt(dmnihavePath, delimiter=',')
        np.save(path_wo_ext, DMNIHAve)

    size = len(selected_indices)

    assert size >= 4

    D = origin_xyz  # our anchors
    DD = DMNIHAve[selected_indices, :]  # contains locations of anchors in original experiment
    DDD = others_xyz  # our sensors


    # ==================== AffineEstimation4 ====================== (l 225)

    # find affine transformation with ideal brain (not used anywhere..)
    listOri = np.c_[D, np.ones(size)]
    DD[:, 3] = 1
    listDist = DD
    W = np.linalg.lstsq(listOri, listDist, rcond=None)[0]  # affine transformation matrix
    listCur = np.matmul(listOri, W)

    # ------------ Transformation to reference brains -------------- ( l 271)
    # find affine transformation with every brain in the 17 templates
    refN = 17
    refBList = np.empty((refN, 2), dtype=object)

    # load 17 brain templates from disk
    DMS = []
    for i in range(1, refN+1):
        path_wo_ext = "resource/MNI_templates/DMNI{:0>4d}".format(i)
        if Path(path_wo_ext+".npy").is_file():
            DMS.append(np.load(path_wo_ext+".npy", allow_pickle=True))
        else:
            csv_path = Path(path_wo_ext+".csv")
            DM = np.genfromtxt(csv_path, delimiter=',')
            np.save(path_wo_ext, DM)
            DMS.append(DM)
    # find affine transformation between our anchors and all brains
    for i in range(1, refN+1):
        DM = DMS[i-1][selected_indices, :]
        refDist = np.c_[DM, np.ones(size)]
        WW = np.linalg.lstsq(listOri, refDist, rcond=None)[0]
        refBListCur = np.matmul(listOri, WW)
        refBList[i-1, 0] = refBListCur
        refBList[i-1, 1] = WW

    # ---------- Transforming given head surface points stored in others to the ideal brain and each ref brain ----- (l 310)

    DDDD = np.c_[DDD, np.ones(DDD.shape[0])]
    othersRefList = np.empty((1, refN), dtype=object)

    for i in range(refN):
        WR = refBList[i, 1]
        othersRef = np.matmul(DDDD, WR)
        othersRefList[0, i] = othersRef


    # --------------Restore data across reference brains----------------  (l 335)

    pointN = DDD.shape[0]
    pListOverPoint = np.empty((1, pointN), dtype=object)

    for i in range(pointN):                             # for each point from others
        pList = np.ones((refN, 3))

        for j in range(refN):                           # iterate over all the reference brains (17)
            pList[j, :] = othersRefList[0, j][i, 0:3]

        pListOverPoint[0, i] = pList


    # --------- Finding the representative point on the head surface ------- (l 361)

    otherH = np.ones((pointN, 3))
    otherHMean = np.ones((pointN, 3))
    otherHVar = np.ones((pointN, 4))
    otherHSD = np.ones((pointN, 4))

    path_wo_ext = "resource/MNI_templates/xyzallHEM"
    if Path(path_wo_ext+".npy").is_file():
        XYZ = np.load(path_wo_ext+".npy", allow_pickle=True)
    else:
        xallhemPath = Path("resource/MNI_templates/xallHEM.csv")
        yallhemPath = Path("resource/MNI_templates/yallHEM.csv")
        zallhemPath = Path("resource/MNI_templates/zallHEM.csv")
        xallHEM = np.genfromtxt(xallhemPath, delimiter=',')
        yallHEM = np.genfromtxt(yallhemPath, delimiter=',')
        zallHEM = np.genfromtxt(zallhemPath, delimiter=',')
        XYZ = np.column_stack((xallHEM, yallHEM, zallHEM))
        np.save(path_wo_ext, XYZ)
    top = 3

    for i in range(pointN):
        AA = np.mean(pListOverPoint[0, i], axis=0)
        otherHMean[i, :] = AA

        # ----- Back projection ----- (l 707)
        PP = np.broadcast_to(AA, XYZ.shape)
        D = np.linalg.norm(XYZ - PP, axis=1)
        # preD = XYZ - PP
        # preD2 = preD*preD
        # preD3 = np.sum(preD2, axis=1)
        # D = preD3 ** 0.5
        IDtop = np.argpartition(D, top)[:top] # sort by lowest norm
        # ID = np.argsort(D)
        # IDtop = ID[0: top]
        XYZtop = XYZ[IDtop, :]
        closest = np.mean(XYZtop, axis=0)
        # -------- End of back projection ----------

        otherH[i, :] = closest

        # ---- Variance calculation ---- (line 739)

        AAA = pListOverPoint[0, i]
        AV = closest
        N = AAA.shape[0]
        subMat = np.ones(AAA.shape)
        for j in range(N):
            subMat[j, :] = AV

        dispEach = AAA - subMat
        dispEachSq = dispEach * dispEach
        XYZSS = np.sum(dispEachSq, axis=0)
        RSS = np.sum(XYZSS)

        XYZVar = XYZSS / (N-1)
        RSSVar = RSS / (N-1)

        VV = np.append(XYZVar, RSSVar)
        otherHVar[i, :] = VV
        otherHSD[i, :] = np.sqrt(VV)


    # --------- Calculating errors on cortical surface -------------

    otherRefCList = np.empty((1, refN), dtype=object)
    for i in range(refN):
        projectionListC = np.ones((pointN, 3))
        my_str = "resource/MNI_templates/xyzall{}.npy".format(str(i+1))
        if Path(my_str).is_file():
            XYZ = np.load(my_str, allow_pickle=True)
        else:
            pathX = Path("resource/MNI_templates/xallM0" + str(i+1) + ".csv")
            pathY = Path("resource/MNI_templates/yallM0" + str(i+1) + ".csv")
            pathZ = Path("resource/MNI_templates/zallM0" + str(i+1) + ".csv")
            xall = np.genfromtxt(pathX, delimiter=',')
            yall = np.genfromtxt(pathY, delimiter=',')
            zall = np.genfromtxt(pathZ, delimiter=',')
            XYZ = np.column_stack((xall, yall, zall))
            np.save("resource/MNI_templates/xyzall{}".format(str(i+1)), XYZ)
        logging.info("Projection to MNI in progress [" + str(round(100/17*(i+1))) + "%]")

        for j in range(pointN):
            P = othersRefList[0, i][j, 0:3]
            PP = np.broadcast_to(P, XYZ.shape)
            D = np.linalg.norm(XYZ - PP, axis=1)
            # PP = np.ones(XYZ.shape)
            # PP[:, 0], PP[:, 1], PP[:, 2] = P[0], P[1], P[2]
            # PreD = XYZ - PP
            # PreD2 = PreD*PreD
            # PreD3 = np.sum(PreD2, axis=1)
            # D = PreD3 ** 0.5
            top = round(XYZ.shape[0] * 0.05)
            ID = np.argsort(D)
            IDtop = ID[0:top]
            XYZtop = XYZ[IDtop, :]

            Nclose = 200
            IDclose = ID[0:Nclose]
            XYZclose = XYZ[IDclose, :]
            PNear = np.mean(XYZclose, axis=0)

            # Line between P and PNear
            # NXYZtop = XYZtop.shape[0]
            PVec = P - PNear
            A, B, C = PVec[0], PVec[1], PVec[2]

            t = (A * (XYZtop[:, 0] - P[0]) + B * (XYZtop[:, 1] - P[1]) + C * (XYZtop[:, 2] - P[2])) / (
                        A * A + B * B + C * C)
            H = np.array([A * t + P[0], B * t + P[1], C * t + P[2]]).T

            # H = np.ones(XYZtop.shape)
            # for k in range(NXYZtop):
            #     xc, yc, zc = XYZtop[k,0], XYZtop[k, 1], XYZtop[k, 2]
            #     t = (A * (xc - P[0]) + B * (yc - P[1]) + C * (zc - P[2])) / (A * A + B * B + C * C)
            #     H[k, :] = np.column_stack((A * t + P[0], B * t + P[1], C *t + P[2]))

            # Find deviation between points in XYZclose and H (l 841)
            # PreDH = XYZtop - H
            # PreDH2 = PreDH * PreDH
            # PreDH3 = np.sum(PreDH2, axis=1)
            # DH = PreDH3 ** 0.5
            DH = np.linalg.norm(XYZtop - H, axis=1)
            # Extend the line P-Pnear to a rod (l 851)
            det = 0
            rodR = 0
            while det == 0:
                rodR += 1
                Iless2 = np.where(DH <= rodR)
                rod = XYZtop[Iless2, :][0]
                det = np.sum(rod**2)

            # Find brain surface points on the vicinity of P (l 862)
            PPB = np.broadcast_to(P, rod.shape)
            VicD = np.linalg.norm(rod - PPB, axis=1)
            # PPB = np.ones(rod.shape)
            # PPB[:, 0], PPB[:, 1], PPB[:, 2] = P[0], P[1], P[2]
            # PreVicD = rod - PPB
            # PreVicD2 = PreVicD*PreVicD
            # PreVicD3 = np.sum(PreVicD2, axis=1)
            # VicD = PreVicD3**0.5
            NVic = 3
            if rod.shape[0] < NVic:
                NVic = rod.shape[0]
            IVicD = np.argsort(VicD)
            NIVicD = IVicD[0:NVic]
            Vic = rod[NIVicD, :]
            CP = np.mean(Vic, axis=0)
            # --- End of projection BS -----
            projectionListC[j, :] = CP

        otherRefCList[0, i] = projectionListC

    # ----- Restore data across reference brains -----
    CPListOverPoint = np.empty((1, pointN), dtype=object)

    for i in range(pointN):
        CPlist = np.ones((refN, 3))

        for j in range(refN):
            CPlist[j, :] = otherRefCList[0, j][i, :]

        CPListOverPoint[0, i] = CPlist


    # ------ Finding the representive point on the head surface -----

    otherC = np.ones(otherH.shape)
    otherCVar = np.ones((pointN, 4))
    otherCSD = np.ones((pointN, 4))
    my_str = "resource/MNI_templates/xyzallBEM.npy"
    if Path(my_str).is_file():
        XYZ = np.load(my_str, allow_pickle=True)
    else:
        xallbemPath = Path("resource/MNI_templates/xallBEM.csv")
        yallbemPath = Path("resource/MNI_templates/yallBEM.csv")
        zallbemPath = Path("resource/MNI_templates/zallBEM.csv")
        xallBEM = np.genfromtxt(xallbemPath, delimiter=',')
        yallBEM = np.genfromtxt(yallbemPath, delimiter=',')
        zallBEM = np.genfromtxt(zallbemPath, delimiter=',')
        XYZ = np.column_stack((xallBEM, yallBEM, zallBEM))
        np.save("resource/MNI_templates/xyzallBEM", XYZ)
    top = 3

    for i in range(pointN):
        AA = np.mean(CPListOverPoint[0, i], axis=0)
        PP = np.ones(XYZ.shape)
        PP[:, 0], PP[:, 1], PP[:, 2] = AA[0], AA[1], AA[2]
        PreD = XYZ - PP
        PreD2 = PreD * PreD
        PreD3 = np.sum(PreD2, axis=1)
        D = PreD3 ** 0.5
        ID = np.argsort(D)
        IDtop = ID[0:top]
        XYZtop = XYZ[IDtop, :]
        closest = np.mean(XYZtop, axis=0)
        BB = closest
        otherC[i, :] = BB

        # --- variance calculation ---
        AA = CPListOverPoint[0, i]
        AV = BB
        N = AA.shape[0]
        subMat = np.ones(AA.shape)
        for j in range(N):
            subMat[j, :] = AV

        DispEach = AA - subMat
        DispEachSq = DispEach*DispEach
        XYZSS = np.sum(DispEachSq, axis=0)
        RSS = np.sum(XYZSS)

        XYZVar = XYZSS / (N - 1)
        RSSVar = RSS / (N - 1)
        VV = np.append(XYZVar, RSSVar)
        otherCVar[i, :] = VV
        otherCSD[i, :] = np.sqrt(VV)


    SSwsH = otherHVar * (refN - 1)
    SSwsC = otherCVar * (refN - 1)

    # ---- what he have by now ----
    # otherH - given head surface points transformed to the MNI ideal head (within-subject hat)
    # otherC - given cortical surface points transformed to the MNI ideal brain (within-subject hat)
    # otherHSD, otherCSD -  transformation SD for given head surface points, point manner
    # SSwsH, SSwsC - transformation SD for given cortical surface points, point manner
    # ----------------------
    return otherH, otherC, otherHSD, otherCSD
