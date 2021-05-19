#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <time.h>

#include "cfg.h"
#include "error.h"

/**********************************************************/
void Analyse (int prod, int Li, int Ld, TGram G, int *index,
	       TrellisShort F1, TrellisShort F2, TrellisShort S, 
	       char **VNT, char **VT) {

  Rule2 *gre;
  Rule1 *reg;
  int l, pi, pd;

  if ((Ld - Li) == 0)
    if ((gre = Search_b(prod, index[Li], G.b)) != NULL)
      fprintf(stdout,"(<%s> %s)",VNT[prod],VT[index[Li]]);
    else
      Warning(2,"");
  else {
    l = S[Li][Ld][prod];
    pi = F1[Li][Ld][prod];
    pd = F2[Li][Ld][prod];
    if ((reg = Search_a(prod, pi, pd, G.a)) != NULL) {
      fprintf(stdout,"(<%s> ",VNT[prod]);
      Analyse(pi, Li, l, G, index, F1, F2, S, VNT, VT);
      fprintf(stdout," ");
      Analyse(pd, l+1, Ld, G, index, F1, F2, S, VNT, VT);
      fprintf(stdout,")");
    }
    else
      Warning(2,"");
  }
}

/**************************/
void syntax(char *command) {

  fprintf(stdout,"\nUsage: %s [-v [-s]] -g g0 -m fic [-l]",command);
  fprintf(stdout,"[-e TaTre]\n\n");
  fprintf(stdout,"-v:\t\tviterbi probability. (inside by default).\n");
  fprintf(stdout,"-s:\t\tshow structure if viterbi.\n");
  fprintf(stdout,"-g g0:\t\tinput grammar.\n");
  fprintf(stdout,"-m fic:\t\tsample file.\n");
  fprintf(stdout,"-e TaTre:\ttrellis size (%d by default).\n",TATREL);
  fprintf(stdout,"-l:\t\tshow only normalized likelihood (not by default).\n\n");
  exit(0);
}

/************************************************************************/
/************************************************************************/
/************************************************************************/

int main(int argc,char *argv[]) {
  extern char *optarg;
  extern int opterr;
  TGram G;
  char gram1[MAXCAD], cad1[MAXCAD], fmu[MAXCAD], **VNT, **VT;
  int option, vite, lsam, TaTrel, *index, CCat, ptree, l, pd, pi, nL, longTot;
  Trellis e;
  TrellisShort F1, F2, S;
  FILE *ficm;
  TTree tree;
  double VeroTotal;

  strcpy(gram1,"");
  strcpy(fmu,"");
  vite = 0;
  TaTrel = TATREL;
  ptree = 0;
  nL = 0;

  if (argc == 1) syntax(argv[0]);
  while ((option=getopt(argc,argv,"hvg:m:e:b:sl")) != EOF ) {
    switch(option) {
    case 'v': vite = 1;break;
    case 'e': TaTrel = atoi(optarg);break;
    case 'g': strcpy(gram1,optarg);break;
    case 'm': strcpy(fmu,optarg);break;
    case 's': ptree = 1; break;
    case 'l': nL = 1; break;
    case 'h': syntax(argv[0]);
    default: syntax(argv[0]);
    }
  }
  if (vite == 0)
    ptree = 0;

  ReadGram(&G, gram1);

  Memory(&e, G.NoTer, TaTrel);
  if (vite == 1) {
    MemoryShort(&F1, G.NoTer, TaTrel);
    MemoryShort(&F2, G.NoTer, TaTrel);
    MemoryShort(&S, G.NoTer, TaTrel);
    if (ptree == 1) {
      WriteHashTab(G.SNT, &VNT);
      WriteHashTab(G.ST, &VT);
    }
  }

  tree.maxNodes=4*TaTrel;
  tree.l = (int *) malloc(tree.maxNodes * sizeof(int));
  tree.r = (int *) malloc(tree.maxNodes * sizeof(int));

  if ((index = (int*) malloc(TaTrel*sizeof(int))) == NULL) 
    Error(1,"");
  
  if ((ficm = fopen(fmu, "r")) == NULL) Error(0, fmu);

  CCat = 0;
  VeroTotal = 0.0; longTot = 0;
  while ((fgets(cad1,MAXCAD,ficm) != NULL)) {
    CCat++;
    cad1[strlen(cad1) - 1] = '\0';
    if (strlen(cad1) >= MAXCAD) fprintf(stderr,"Warning: string number %d too long.\n",CCat);
    tree.lastNode = 0;
    if ((lsam = IndexSpan(cad1, G, index, &tree, TaTrel)) != 0) {
      if (vite == 0)
        InnerSpan(e, G, index, lsam, tree);
      else
        ViterbiSpan(G, lsam, index, e, F1, F2, S, tree);
      VeroTotal += log(e[0][lsam - 1][0]);

      longTot += lsam;
      if (nL == 0) {
	fprintf(stdout,"%e\n",e[0][lsam - 1][0]);
	fflush(stdout);
	if (ptree == 1) {
	  if (lsam == 1)
	    fprintf(stdout,"(<%s> %s)\n",VNT[0],VT[index[0]]);
	  else {
	    l = S[0][lsam - 1][0];
	    pi = F1[0][lsam - 1][0];
	    pd = F2[0][lsam - 1][0];
	    fprintf(stdout,"(<%s> ",VNT[0]);
	    Analyse(pi, 0, l, G, index, F1, F2, S, VNT, VT);
	    fprintf(stdout," ");
	    Analyse(pd, l+1,lsam-1, G, index, F1, F2, S, VNT, VT);
	    fprintf(stdout,")\n");
	  }
	}
      }
    }
  }
  if (nL == 1)
    fprintf(stdout,"%e %e\n",VeroTotal/longTot,exp(VeroTotal));
  return(0);

}
