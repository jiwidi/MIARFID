#!/usr/bin/octave -qf

T = [0 0 0 .576; 0 0 1 .008; 0 1 0 .144; 0 1 1 .072; 1 0 0 .064; 1 0 1 .012; 1 1 0 .016; 1 1 1 .108];

indexDolor = 1
indexCaires = 2
indexHueco = 3


pDolor=sum(T(find(T(:,indexDolor)==1),end))
pCaires=sum(T(find(T(:,indexCaires)==1),end))
pHueco=sum(T(find(T(:,indexHueco)==1),end))

Pdolorycaries=sum(T(find(T(:,indexDolor)==1 & T(:,indexCaires)==1),end))
Pdolordadocaries = Pdolorycaries/pCaires


Pcariesdadodolor = pCaires * (Pdolordadocaries/pDolor)
