#include "utils.h"

typedef double ***Trellis;
typedef short ***TrellisShort;

typedef struct sRule{
  int con1,con2;
  double pro,frec,estSuf;
  struct sRule *next;
} Rule1;

typedef struct rRule{
  int con;
  double pro,frec,estSuf;
  struct rRule *next;
} Rule2;

typedef Rule1 **Rules1;

typedef Rule2 **Rules2;

typedef struct {
  Rules1 a;
  Rules2 b;
  int Ter, NoTer;
  THashTable SNT, ST;
} TGram;

typedef struct {
  int maxNodes, lastNode;
  int *l, *r;
} TTree;

/**************************************************************/
extern void ReadGram(TGram *G, char *NameFic);
extern void WriteGram(TGram G, char *NameFic, int all, int argc, char *argv[]);
extern void Memory(Trellis *e, int NoTer, int TaTrel);
extern void MemoryShort(TrellisShort *e, int NoTer, int TaTrel);
extern void InnerSpan(Trellis e, TGram G, int *index, int lsam, TTree tree);
extern int IndexSpan(char * cade1, TGram G, int * index, TTree * t, int size);
extern Rule2 *Search_b(int ant, int con, Rules2 b);
extern Rule1 *Search_a(int ant, int con1, int con2, Rules1 a);
extern void ViterbiSpan (TGram G, int lsam, int *index, Trellis Q, 
			 TrellisShort F1, TrellisShort F2, 
			 TrellisShort S, TTree tree);
extern void Smooth(TGram G, double smooth);
extern void DeleteNullRules (TGram *G, Rules1 a_nul, Rules2 b_nul);
extern void MemNullRules (Rules1 *a, Rules2 *b, int NoTer);
extern void InsertNullRules (TGram *G, Rules1 a_nul, Rules2 b_nul);











