import sys, pickle, numpy as np, cv2
sys.path.insert(0, "/mnt/14BE47C2BE479ADE/Code/ear_stuff/Ear_3D_Lifter")
from ear3d.face import show
from lap import E_, edge_ratio, RUN
L = pickle.load(open("lifts.pkl", "rb"))
sc = [np.nanpercentile(np.abs(np.log(edge_ratio(o["current"], G))), 90) for h, o, G in L]
for j in np.argsort(sc)[[-1, -2, len(sc)//2]]:
    h, o, G = L[j]; z = np.load(RUN / f"{h}.npz")
    for k in ("current", "lap 3"):
        show(z["V"], z["F"], [(G, (0, 255, 0)), (o[k], (0, 0, 255))], f"{h}_{k.replace(' ','')}.jpg", yaws=(-40, 0, 40), centre=G[1]-np.array([0,0,60]))
    print(h, round(sc[j], 3))
