
CFLAGS= -ansi -Wall -pedantic -O3

OBJECTS= error.o ht.o utils.o cfg.o 

all: scfg_cgr scfg_gstr scfg_prob scfg_learn_mmi genFig

clean:
	rm -if *.o
	rm -if *~*

scfg_cgr: $(OBJECTS) scfg_cgr.c 
	$(CC) scfg_cgr.c $(OBJECTS) -o scfg_cgr $(CFLAGS) -lm

scfg_gstr: $(OBJECTS) scfg_gstr.c 
	$(CC) scfg_gstr.c $(OBJECTS) -o scfg_gstr $(CFLAGS) -lm

scfg_learn_mmi: $(OBJECTS) scfg_learn_mmi.c 
	$(CC) scfg_learn_mmi.c $(OBJECTS) -o scfg_learn_mmi $(CFLAGS) -lm

scfg_prob: $(OBJECTS) scfg_prob.c 
	$(CC) scfg_prob.c $(OBJECTS) -o scfg_prob $(CFLAGS) -lm

genFig: genFig.c
	$(CC) -o genFig genFig.c $(CFLAGS) -lm



cfg.o: cfg.c cfg.h utils.h error.h ht.h
	$(CC) -c cfg.c  $(CFLAGS) 

utils.o: utils.c utils.h error.h ht.h
	$(CC) -c utils.c $(CFLAGS) 

error.o: error.c error.h
	$(CC) -c error.c $(CFLAGS) 

ht.o: ht.c ht.h
	$(CC) -c ht.c $(CFLAGS) 


