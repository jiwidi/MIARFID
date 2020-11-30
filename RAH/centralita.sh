HParse telefono.gram telefono.wdnet 
mkdir mfc
HCopy -T 1 -C param.conf -S parametriza.scp
python2.7 hacermlf.py frases training > entrenamiento.mlf 
python2.7 hacermlf.py frasestest test > evaluacion.mlf

HLEd -l '*' -d telefono.dic -i transcripcion.mlf transcribe.led entrenamiento.mlf 

ls mfc/training*.mfc > entrena.scp

mkdir hmm0
HCompV -C entrena.conf -S entrena.scp -m -M hmm0 proto
python2.7 crea_hmmdefs.py fonemas hmm0/proto > hmm0/hmmdefs
mkdir hmm1 
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm0/hmmdefs -M hmm1 fonemas
mkdir hmm2
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm1/hmmdefs -M hmm2 fonemas
mkdir hmm3 
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm2/hmmdefs -M hmm3 fonemas
mkdir hmm4 
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm3/hmmdefs -M hmm4 fonemas
mkdir hmm5 
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm4/hmmdefs -M hmm5 fonemas

ls mfc/test*.mfc > evalua.scp 

#HVite -T 1 -C entrena.conf -H hmm5/hmmdefs -S evalua.scp -l "mfc" -i salida.mlf -w telefono.wdnet telefono.dic fonemas



HVite -T 1 -C entrena.conf -H hmm5/hmmdefs -S evalua.scp -l “mfc” -i salida.mlf -w telefono.wdnet -t 1250.0 telefono.dic fonemas

HVite -T 1 -C entrena.conf -H hmm5/hmmdefs -S evalua.scp -l “mfc” -i salida.mlf -w telefono.wdnet -t 250.0 telefono.dic fonemas

python2.7 crea_diccionario.py telefono.wdnet > palabras
HResults -I evaluacion.mlf palabras salida.mlf

mkdir hmm6
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm5/hmmdefs -M hmm6 fonemas
mkdir hmm7
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm6/hmmdefs -M hmm7 fonemas
mkdir hmm8
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm7/hmmdefs -M hmm8 fonemas
mkdir hmm9
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm8/hmmdefs -M hmm9 fonemas
mkdir hmm10
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm9/hmmdefs -M hmm10 fonemas
mkdir hmm11
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm10/hmmdefs -M hmm11 fonemas
mkdir hmm12
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm11/hmmdefs -M hmm12 fonemas
mkdir hmm13
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm12/hmmdefs -M hmm13 fonemas
mkdir hmm14
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm13/hmmdefs -M hmm14 fonemas

HVite -T 1 -C entrena.conf -H hmm14/hmmdefs -S evalua.scp -l "mfc" -i salida.mlf -w telefono.wdnet -t 1250.0 telefono.dic fonemas

HResults -I evaluacion.mlf palabras salida.mlf

#–
echo "MU 4 {*.state[2-4].mix}" > Mixtures.conf
mkdir hmm15
HHEd -H hmm14/hmmdefs -M hmm15 Mixtures.conf fonemas
mkdir hmm16
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm15/hmmdefs -M hmm16 fonemas
mkdir hmm17
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm16/hmmdefs -M hmm17 fonemas
mkdir hmm18
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm17/hmmdefs -M hmm18 fonemas
mkdir hmm19
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm18/hmmdefs -M hmm19 fonemas
mkdir hmm20
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm19/hmmdefs -M hmm20 fonemas
mkdir hmm21
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm20/hmmdefs -M hmm21 fonemas
mkdir hmm22
HERest -T 1 -C entrena.conf -I transcripcion.mlf -S entrena.scp -H hmm21/hmmdefs -M hmm22 fonemas
HVite -T 1 -C entrena.conf -H hmm22/hmmdefs -S evalua.scp -l "mfc" -i salida.mlf -t 1250 -w telefono.wdnet telefono.dic fonemas 

HResults -I evaluacion.mlf palabras salida.mlf
