import numpy as np
from pathlib import Path
import logging


def project(origin_xyz, others_xyz, selected_indices):
    """
    projects others_xyz to MNI coordiantes given anchors in origin_xyz
    :param origin_xyz: anchors
    :param others_xyz: optodes to project
    :return: otherH - given head surface points transformed to the MNI ideal head (within-subject hat)
             otherC - given cortical surface points transformed to the MNI ideal brain (within-subject hat)
             otherHSD, otherCSD -  transformation SD for given head surface points, point manner
    """
    # --------------------------- CONFIG ----------------------------------

    # originPath = "resource\\MNI_templates\\origin_sample.csv"  # Provide origin path
    # othersPath = "resource\\MNI_templates\\others_sample.csv"  # Provide others path

    # --------------------- Load and process the data ----------------------

    dmnihavePath = Path("resource/MNI_templates/DMNIHAve.csv")
    xallhemPath = Path("resource/MNI_templates/xallHEM.csv")
    yallhemPath = Path("resource/MNI_templates/yallHEM.csv")
    zallhemPath = Path("resource/MNI_templates/zallHEM.csv")
    xallbemPath = Path("resource/MNI_templates/xallBEM.csv")
    yallbemPath = Path("resource/MNI_templates/yallBEM.csv")
    zallbemPath = Path("resource/MNI_templates/zallBEM.csv")

    # try:
    #     originXYZ = (pd.DataFrame(pd.read_csv(originPath), columns=['X', 'Y', 'Z'])).to_numpy()
    # except:
    #     print("Origin file not found!")
    #     exit(1)
    # try:
    #     othersXYZ = (pd.DataFrame(pd.read_csv(othersPath, header=None))).to_numpy()
    # except:
    #     print("Others file not found!")
    #     exit(1)

    originXYZ = origin_xyz
    othersXYZ = others_xyz

    DMNIHAve = np.genfromtxt(dmnihavePath, delimiter=',')
    xallHEM = np.genfromtxt(xallhemPath, delimiter=',')
    yallHEM = np.genfromtxt(yallhemPath, delimiter=',')
    zallHEM = np.genfromtxt(zallhemPath, delimiter=',')
    xallBEM = np.genfromtxt(xallbemPath, delimiter=',')
    yallBEM = np.genfromtxt(yallbemPath, delimiter=',')
    zallBEM = np.genfromtxt(zallbemPath, delimiter=',')


    # othersXYZ = np.delete(othersXYZ, 0, 1)
    # selectedBoolean = np.invert(np.isnan(originXYZ).any(axis=1))

    # for i in range(len(selectedBoolean)):
    #     if selectedBoolean[i]:
    #         selectedIndexes.append(i)

    selectedIndexes = selected_indices
    size = len(selectedIndexes)

    # if size < 4:
    #     print("Please provide at least 4 values in the origin file")
    #     exit(1)

    D = originXYZ
    DD = DMNIHAve[selectedIndexes, :]
    DDD = othersXYZ


    # ==================== AffineEstimation4 ====================== (l 225)

    listOri = np.c_[D, np.ones(size)]
    DD[:, 3] = 1
    listDist = DD
    W = np.linalg.lstsq(listOri, listDist, rcond=None)[0]  # transformation matrix
    listCur = np.matmul(listOri, W)

    # ------------ Transformation to reference brains -------------- ( l 271)

    refN = 17
    refBList = np.empty((refN, 2), dtype=object)

    for i in range(1, refN+1):
        path = Path("resource/MNI_templates/DMNI{:0>4d}.csv".format(i))
        DM = np.genfromtxt(path, delimiter=',')
        DM = DM[selectedIndexes, :]
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
    XYZ = np.column_stack((xallHEM, yallHEM, zallHEM))
    top = 3

    for i in range(pointN):
        AA = np.mean(pListOverPoint[0, i], axis=0)
        otherHMean[i, :] = AA

        # ----- Back projection ----- (l 707)
        PP = np.ones(XYZ.shape)
        PP[:, 0], PP[:, 1], PP[:, 2] = AA[0], AA[1], AA[2]
        preD = XYZ - PP
        preD2 = preD*preD
        preD3 = np.sum(preD2, axis=1)
        D = preD3 ** 0.5
        ID = np.argsort(D)      # sort and show indices

        IDtop = ID[0: top]
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
        pathX = Path("resource/MNI_templates/xallM0" + str(i+1) + ".csv")
        pathY = Path("resource/MNI_templates/yallM0" + str(i+1) + ".csv")
        pathZ = Path("resource/MNI_templates/zallM0" + str(i+1) + ".csv")
        xall = np.genfromtxt(pathX, delimiter=',')
        yall = np.genfromtxt(pathY, delimiter=',')
        zall = np.genfromtxt(pathZ, delimiter=',')
        XYZ = np.column_stack((xall, yall, zall))
        logging.info("Projection to MNI in progress [" + str(round(100/17*(i+1))) + "%]")

        for j in range(pointN):
            P = othersRefList[0, i][j, 0:3]
            PP = np.ones(XYZ.shape)
            PP[:, 0], PP[:, 1], PP[:, 2] = P[0], P[1], P[2]
            PreD = XYZ - PP
            PreD2 = PreD*PreD
            PreD3 = np.sum(PreD2, axis=1)
            D = PreD3 ** 0.5
            ID = np.argsort(D)
            top = round(XYZ.shape[0] * 0.05)
            IDtop = ID[0:top]
            XYZtop = XYZ[IDtop, :]

            Nclose = 200
            IDclose = ID[0:Nclose]
            XYZclose = XYZ[IDclose, :]
            PNear = np.mean(XYZclose, axis=0)

            # Line between P and PNear
            NXYZtop = XYZtop.shape[0]
            PVec = P - PNear
            A, B, C = PVec[0], PVec[1], PVec[2]
            H = np.ones(XYZtop.shape)

            for k in range(NXYZtop):
                xc, yc, zc = XYZtop[k,0], XYZtop[k, 1], XYZtop[k, 2]
                t = (A * (xc - P[0]) + B * (yc - P[1]) + C * (zc - P[2])) / (A * A + B * B + C * C)
                H[k, :] = np.column_stack((A * t + P[0], B * t + P[1], C *t + P[2]))

            # Find deviation between points in XYZclose and H (l 841)
            PreDH = XYZtop - H
            PreDH2 = PreDH * PreDH
            PreDH3 = np.sum(PreDH2, axis=1)
            DH = PreDH3 ** 0.5

            # Extend the line P-Pnear to a rod (l 851)
            det = 0
            rodR = 0
            while det == 0:
                rodR += 1
                Iless2 = np.where(DH <= rodR)
                rod = XYZtop[Iless2, :][0]
                tmp = rod*rod
                det = np.sum(sum(np.sum(tmp, axis=0)))

            # Find brain surface points on the vicinity of P (l 862)
            PPB = np.ones(rod.shape)
            PPB[:, 0], PPB[:, 1], PPB[:, 2] = P[0], P[1], P[2]
            PreVicD = rod - PPB
            PreVicD2 = PreVicD*PreVicD
            PreVicD3 = np.sum(PreVicD2, axis=1)
            VicD = PreVicD3**0.5
            IVicD = np.argsort(VicD)
            NVic = 3
            if rod.shape[0] < NVic:
                NVic = rod.shape[0]
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
    XYZ = np.column_stack((xallBEM, yallBEM, zallBEM))
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
