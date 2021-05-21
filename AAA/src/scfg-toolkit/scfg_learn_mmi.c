#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <float.h>


#include "cfg.h"
#include "error.h"

double VeroTotalVit;
// DBL_MIN is 2.2250738585072014e-308;
/******************************************/
void InitializeAcum(TGram G, double **Deno) {
  int i;
  Rule1 *reg1;
  Rule2 *gre1;

  if ((*Deno) != NULL)
    free(*Deno);
  if (((*Deno) = (double *) malloc(G.NoTer * sizeof(double))) == NULL)
    Error(1,"");

  for (i=0; i<G.NoTer; i++) {
    for (reg1 = G.a[i]; reg1 != NULL; reg1 = reg1->next)
      reg1->frec = 0.0;
    for (gre1 = G.b[i]; gre1 != NULL; gre1 = gre1->next)
      gre1->frec = 0.0;
    (*Deno)[i] = 0.0;
  }
}

/*******************************************************/
void Accumulate(Trellis e, Trellis f, TGram G, double *Deno,
	      int *index, int lon, TTree tree) {

  int  i, j, k, l, n;
  double Prob, Acum;
  Rule1 *reg1;
  Rule2 *gre1;

  Prob = e[0][lon - 1][0];

  for(k=0; k<G.NoTer; k++) {
    reg1 = G.a[k];
    while (reg1 != NULL) {
      Acum = 0.0;
      for (n=0; n<tree.lastNode; n=n+tree.l[n]+1)
	for (i=1; i<tree.l[n]; i++)
	  for (j=0; j<tree.l[n] - i; j++)
	    for (l=n+1+j; l<n+1+j+i; l++)
	      Acum = Acum +
		reg1->pro *
		e[tree.l[n+1+j]][tree.r[l]][reg1->con1] *
		e[tree.l[l+1]][tree.r[n+1+j+i]][reg1->con2] *
		f[tree.l[n+1+j]][tree.r[n+1+j+i]][k];
      reg1->frec = reg1->frec + Acum / Prob;
      reg1 = reg1->next;
    }
    gre1 = G.b[k];
    while (gre1 != NULL) {
      Acum = 0.0;
      for (j=0; j<lon; j++)
        if (index[j] == gre1->con)
          Acum = Acum + e[j][j][k] * f[j][j][k];
      gre1->frec = gre1->frec + Acum / Prob;
      gre1 = gre1->next;
    }

    Acum = 0.0;

    for (n=0; n<tree.lastNode; n=n+tree.l[n]+1)
      for (i=0; i<tree.l[n]-1; i++)
	for (j=0; j<tree.l[n] - i; j++)
	  Acum = Acum +
	    f[tree.l[n+1+j]][tree.r[n+1+j+i]][k] *
	    e[tree.l[n+1+j]][tree.r[n+1+j+i]][k];
    Acum = Acum + f[0][lon-1][k] * e[0][lon-1][k];
    Deno[k] = Deno[k] + Acum / Prob;
  }
}

/*******************************************************/
void AccumulateSubstract(Trellis e, Trellis f, TGram G, double *Deno,
			 int *index, int lon, TTree tree, float Hval) {

  int  i, j, k, l, n;
  double Prob, Acum;
  Rule1 *reg1;
  Rule2 *gre1;

  Prob = e[0][lon - 1][0];

  for(k=0; k<G.NoTer; k++) {
    reg1 = G.a[k];
    while (reg1 != NULL) {
      Acum = 0.0;
      for (n=0; n<tree.lastNode; n=n+tree.l[n]+1)
	for (i=1; i<tree.l[n]; i++)
	  for (j=0; j<tree.l[n] - i; j++)
	    for (l=n+1+j; l<n+1+j+i; l++)
	      Acum = Acum +
		reg1->pro *
		e[tree.l[n+1+j]][tree.r[l]][reg1->con1] *
		e[tree.l[l+1]][tree.r[n+1+j+i]][reg1->con2] *
		f[tree.l[n+1+j]][tree.r[n+1+j+i]][k];
      reg1->frec = reg1->frec - Hval * Acum / Prob;
      reg1 = reg1->next;
    }
    gre1 = G.b[k];
    while (gre1 != NULL) {
      Acum = 0.0;
      for (j=0; j<lon; j++)
        if (index[j] == gre1->con)
          Acum = Acum + e[j][j][k] * f[j][j][k];
      gre1->frec = gre1->frec - Hval * Acum / Prob;
      gre1 = gre1->next;
    }

    Acum = 0.0;

    for (n=0; n<tree.lastNode; n=n+tree.l[n]+1)
      for (i=0; i<tree.l[n]-1; i++)
	for (j=0; j<tree.l[n] - i; j++)
	  Acum = Acum +
	    f[tree.l[n+1+j]][tree.r[n+1+j+i]][k] *
	    e[tree.l[n+1+j]][tree.r[n+1+j+i]][k];
    Acum = Acum + f[0][lon-1][k] * e[0][lon-1][k];
    Deno[k] = Deno[k] - Hval * Acum / Prob;
  }
}

/***************************************************************/
void OuterSpan(Trellis f, Trellis e, TGram G, int lon, TTree tree) {
  Rule1 *reg1;
  int i, j, k, l, *ll = NULL, *rr = NULL, n;

  /* First, the list of nodes of the tree is reordered in order
     to be processed adequately. */
  ll = (int *) malloc(tree.lastNode * sizeof(int));
  rr = (int *) malloc(tree.lastNode * sizeof(int));

  for (i=0; i<tree.lastNode; i=i+tree.l[i]+1) {
    ll[tree.lastNode-tree.l[i]-i-1] = tree.l[i];
    for (j=0; j<tree.l[i]; j++) {
      ll[tree.lastNode-tree.l[i]-i+j] = tree.l[i+1+j];
      rr[tree.lastNode-tree.l[i]-i+j] = tree.r[i+1+j];
    }
  }

  for (k=0; k<G.NoTer; k++)
    f[0][lon-1][k] = 0.0;
  f[0][lon - 1][0] = 1.0;

  for (n=0; n<tree.lastNode-1; n=n+ll[n]+1)
    for (i=ll[n]-2; i>=0; i--)
      for (j=0; j<ll[n] - i; j++) {
	for (k=0; k<G.NoTer; k++)
	  f[ll[n+1+j]][rr[n+1+j+i]][k] = 0.0;
	for (k=0; k<G.NoTer; k++) {
	  reg1 = G.a[k];
	  while (reg1 != NULL) {
	    for (l=n+1; l<n+1+j; l++)
	      f[ll[n+1+j]][rr[n+1+j+i]][reg1->con2] =
		f[ll[n+1+j]][rr[n+1+j+i]][reg1->con2] +
		f[ll[l]][rr[n+1+j+i]][k] *
		reg1->pro *
		e[ll[l]][rr[n+1+j-1]][reg1->con1];
	    for (l=n+1+j+i+1; l<n+1+ll[n]; l++)
	      f[ll[n+1+j]][rr[n+1+j+i]][reg1->con1] =
		f[ll[n+1+j]][rr[n+1+j+i]][reg1->con1] +
		f[ll[n+1+j]][rr[l]][k] *
		reg1->pro *
		e[ll[n+1+j+i+1]][rr[l]][reg1->con2];
	    reg1 = reg1->next;
	  }
	}
      }

  free(ll);
  free(rr);
}

/**************************************/
void  Reestimate(TGram G, double *Deno) {
  int i;
  Rule1 *reg1;
  Rule2 *gre1;

  double epsiV = 0.5;

  double rMax = DBL_MIN;

  for (i=0; i<G.NoTer; i++) {
    reg1 = G.a[i];
    while (reg1 != NULL) {
      if (-reg1->frec / reg1->pro > rMax)
	rMax = -reg1->frec / reg1->pro;
      reg1 = reg1->next;
    }
    gre1 = G.b[i];
    while (gre1 != NULL) {
      if (-gre1->frec / gre1->pro > rMax)
	rMax = -gre1->frec / gre1->pro;
      gre1 = gre1->next;
    }
  }
  rMax+=epsiV;
  for (i=0; i<G.NoTer; i++) {
    reg1 = G.a[i];
    while (reg1 != NULL) {
      reg1->pro = (reg1->frec + rMax * reg1->pro) / (Deno[i] + rMax);
      reg1 = reg1->next;
    }
    gre1 = G.b[i];
    while (gre1 != NULL) {
      gre1->pro = (gre1->frec + rMax * gre1->pro) / (Deno[i] + rMax);
      gre1 = gre1->next;
    }
  }
}

/*********************************************************************/
void Analyse (int prod, int Li, int Ld, TGram G, int *index, Trellis Q,
	      TrellisShort F1, TrellisShort F2, TrellisShort S,
	      double *Deno) {

  Rule2 *gre;
  Rule1 *reg;
  int l, pi, pd;

  if ((Ld - Li) == 0)
    if ((gre = Search_b(prod, index[Li], G.b)) != NULL) {
      gre->frec = gre->frec + 1;
      Deno[prod] = Deno[prod] + 1.0;
    }
    else
      Warning(2, "");
  else {
    l = S[Li][Ld][prod];
    pi = F1[Li][Ld][prod];
    pd = F2[Li][Ld][prod];
    if ((reg = Search_a(prod, pi, pd, G.a)) != NULL) {
      reg->frec = reg->frec + 1.0;
      Deno[prod] = Deno[prod] + 1.0;
      Analyse(pi, Li, l, G, index, Q, F1, F2, S, Deno);
      Analyse(pd, l+1, Ld, G, index, Q, F1, F2, S, Deno);
    }
    else
      Warning(2, "");
  }
}

/****************************************************************/
void Parsing (TGram G, int lsam, int *index, double *Deno, Trellis Q,
	      TrellisShort F1, TrellisShort F2, TrellisShort S,
	      TTree tree) {
  int l, pi, pd;
  Rule1 *reg;
  Rule2 *gre;

  ViterbiSpan(G, lsam, index, Q, F1, F2, S, tree);

  if (Q[0][lsam - 1][0] == 0.0)
    Warning(0, "");
  else
    if (lsam == 1)
      if ((gre = Search_b(0, index[0], G.b)) != NULL) {
        Deno[0] = Deno[0] + 1.0;
        gre->frec = gre->frec + 1.0;
      }
      else
        Warning(2, "");
    else {
      l = S[0][lsam - 1][0];
      pi = F1[0][lsam - 1][0];
      pd = F2[0][lsam - 1][0];
      if ((reg = Search_a(0, pi, pd, G.a)) != NULL) {
        reg->frec = reg->frec + 1.0;
        Deno[0] = Deno[0] + 1.0;
        Analyse(pi, 0, l, G, index, Q, F1, F2, S, Deno);
        Analyse(pd, l+1, lsam-1, G, index, Q, F1, F2, S, Deno);
      }
      else
        Warning(2, "");
    }
}

/**************************/
void Syntax(char *command) {

  fprintf(stdout,"\nUsage: %s -g gram1 [-f gram2] -p samples [-n samples] [-i ite]",command);
  fprintf(stdout," [{-s|-S} smooth] [-l] [-e TaTre] [-v] [-a] [-H hi]\n\n");
  fprintf(stdout,"-a:\t\tkeep null rules (no, by default).\n");
  fprintf(stdout,"-g gram1:\tinitial grammar.\n");
  fprintf(stdout,"-f gram2:\tfinal grammar (gram1, by default).\n");
  fprintf(stdout,"-p sample:\tpositive sample file.\n");
  fprintf(stdout,"-n sample:\tnegative sample file.\n");
  fprintf(stdout,"-i ite:\t\titeration number (1, by default).\n");
  fprintf(stdout,"-s|-S smooth:\tsmooth value (0, by default).\n");
  fprintf(stdout,"\t\t-s: smooth after each iteration.\n");
  fprintf(stdout,"\t\t-S: smooth after last iteration.\n");
  fprintf(stdout,"-l:\t\tshow normalized log likelihood with the input grammar.\n");
  fprintf(stdout,"-e TaTre:\ttrellis size (%d, by default).\n",TATREL);
  fprintf(stdout,"-a:\t\tkeep null rules (no, by default).\n");
  fprintf(stdout,"-H h:\t\th value (0.0, by default).\n\n");
  exit(0);
}

/************************************************************************/
/************************************************************************/
/************************************************************************/

int main(int argc,char *argv[]) {
  extern char *optarg;
  extern int opterr;
  TGram G;
  Rules1 a_null;
  Rules2 b_null;
  char gram1[MAXCAD], gram2[MAXCAD];
  int  i, ii, option, ite, lsam, all, smooth,  *index,
    SmoCadaIte, SacaVero, TaTrel, CCat=0,  vite=0;
  double *Deno=NULL, Hval;
  char cad1[MAXCAD], fmu[MAXCAD], cad2[MAXCAD], fmuNeg[MAXCAD];
  Trellis e, f;
  TrellisShort F1, F2, S;
  FILE *ficm;
  double VeroTotal;
  TTree tree;

  smooth = 0.0;
  SmoCadaIte = true;
  ite = 1;
  strcpy(gram1,"");
  strcpy(gram2,"");
  strcpy(fmu,"");
  strcpy(fmuNeg,"");
  all = false;
  VeroTotal = 0.0;
  SacaVero = false;
  TaTrel = TATREL;
  Hval = 0.0;

  if (argc == 1)
    Syntax(argv[0]);
  while ((option=getopt(argc,argv,"lahg:f:p:i:s:S:e:b:H:vn:")) != EOF ) {
    switch(option) {
    case 'g': strcpy(gram1,optarg);strcpy(gram2,gram1); break;
    case 'f': strcpy(gram2,optarg); break;
    case 'p': strcpy(fmu,optarg); break;
    case 'n': strcpy(fmuNeg,optarg); break;
    case 'i': ite = atoi(optarg); break;
    case 'v': vite = 1; break;
    case 'e': TaTrel = atoi(optarg); break;
    case 's': smooth = atof(optarg); SmoCadaIte = true; break;
    case 'H': Hval = atof(optarg); break;
    case 'S': smooth = atof(optarg); SmoCadaIte = false; break;
    case 'a': all = true; break;
    case 'l': SacaVero = true; break;
    case 'h': Syntax(argv[0]); break;
    default: Syntax(argv[0]);
    }
  }

  ReadGram(&G, gram1);

  MemNullRules(&a_null, &b_null, G.NoTer);
  if (smooth > 0.0)
    Smooth(G, smooth);

  if ((index = (int*) malloc(TaTrel*sizeof(int))) == NULL)
    Error(1,"");

  Memory(&e, G.NoTer, TaTrel);
  if (vite == 0)
    Memory(&f, G.NoTer, TaTrel);
  else {
    MemoryShort(&F1, G.NoTer, TaTrel);
    MemoryShort(&F2, G.NoTer, TaTrel);
    MemoryShort(&S, G.NoTer, TaTrel);
  }

  tree.maxNodes=5*TaTrel;
  tree.l = (int *) malloc(tree.maxNodes * sizeof(int));
  tree.r = (int *) malloc(tree.maxNodes * sizeof(int));

  for (i=0; i<ite; i++) {

    if ((ficm = fopen(fmu,"r")) == NULL) Error(0, fmu);

    VeroTotal = 0.0;
    VeroTotalVit = 0.0;

    InitializeAcum(G, &Deno);

    while ((fgets(cad1, MAXCAD, ficm) != NULL)) {
      CCat++;
      if (CCat % 10 == 0) fprintf(stderr,".");
      if (CCat % 1000 == 0) fprintf(stderr,"%5dK",CCat/1000);
      if (CCat % 100 == 0) fprintf(stderr,"\n");
      fflush(stderr);
      cad1[strlen(cad1) - 1] = '\0';
      strcpy(cad2,cad1);
      if (strlen(cad1) >= MAXCAD)
	fprintf(stderr,"Warning: string number %d too long.\n",CCat);

      tree.lastNode = 0;
      if ((lsam = IndexSpan(cad1, G, index, &tree, TaTrel)) != 0) {
	if (vite == 0) {
	  InnerSpan(e, G, index, lsam, tree);
	  OuterSpan(f, e, G, lsam, tree);

	  if (e[0][lsam - 1][0] == 0.0)
	    Warning(0, cad1);
	  else {
	    Accumulate(e, f, G, Deno, index, lsam, tree);
	    VeroTotal += log(e[0][lsam - 1][0]);
	  }
	  for (ii=0; ii<strlen(cad2); ii++)
	    if (cad2[ii] == '[' || cad2[ii] == ']')
	      cad2[ii] = ' ';
	  tree.lastNode = 0;
	  lsam = IndexSpan(cad2, G, index, &tree, TaTrel);
	  InnerSpan(e, G, index, lsam, tree);
	  OuterSpan(f, e, G, lsam, tree);
	  AccumulateSubstract(e, f, G, Deno, index, lsam, tree, Hval);
	}
	else {
	  Parsing(G, lsam, index, Deno, e, F1, F2, S, tree);
	  if (e[0][lsam - 1][0] == 0.0)
	    Warning(0, cad1);
	  else
	    VeroTotalVit += log(e[0][lsam - 1][0]);
	}
      }
      else
	Warning(1,cad1);
    }
    /*
    fclose(ficm);
    if (strcmp(fmuNeg,"") != 0 && strcmp(fmuNeg,"") != 0) {
      if ((ficm = fopen(fmuNeg,"r")) == NULL) Error(0, fmuNeg);
      while (fgets(cad1, MAXCAD, ficm) != NULL) {
	CCat++;
	if (CCat % 10 == 0) fprintf(stderr,".");
	if (CCat % 1000 == 0) fprintf(stderr,"%5dK",CCat/1000);
	fflush(stderr);
	if (CCat % 100 == 0) fprintf(stderr,"\n");
	fflush(stderr);
	cad1[strlen(cad1) - 1] = '\0';
	if (strlen(cad1) >= MAXCAD)
	  fprintf(stderr,"Warning: string number %d too long.\n",CCat);

	tree.lastNode = 0;
	if ((lsam = IndexSpan(cad1, G, index, &tree, TaTrel)) != 0) {
	  InnerSpan(e, G, index, lsam, tree);
	  OuterSpan(f, e, G, lsam, tree);
	  AccumulateSubstract(e, f, G, Deno, index, lsam, tree, Hval);
	}
	else
	  Warning(1,cad1);
      }
    }
    */
    if (vite == 0)
      Reestimate(G, Deno);
    if ((smooth > 0.0) && (SmoCadaIte == true))
      Smooth(G, smooth);
    else
      DeleteNullRules(&G, a_null, b_null);

    if (SacaVero == true)
      fprintf(stdout,"%e\n",VeroTotalVit - Hval * VeroTotal);
    fclose(ficm);
    WriteGram(G, gram2, all, argc, argv);
  }

  if (all == true)
    InsertNullRules(&G, a_null, b_null);
  if (smooth > 0.0)
    if (SmoCadaIte == false)
      Smooth(G, smooth);

  return(0);
}
