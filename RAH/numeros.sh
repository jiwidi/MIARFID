mkdir mfc
HCopy -C param.conf -S param.scp
mkdir hmm0
HCompV -C param2.conf -S entrena.scp -m -M hmm0 proto
python2.7 crea_hmmdefs.py palabras hmm0/proto > hmm0/hmmdefs
mkdir hmm1
HERest -T 1 -C param2.conf -I numeros.mlf -S entrena.scp -H hmm0/hmmdefs -M hmm1 palabras
mkdir hmm2
HERest -T 1 -C param2.conf -I numeros.mlf -S entrena.scp -H hmm1/hmmdefs -M hmm2 palabras
mkdir hmm3
HERest -T 1 -C param2.conf -I numeros.mlf -S entrena.scp -H hmm2/hmmdefs -M hmm3 palabras
mkdir hmm4
HERest -T 1 -C param2.conf -I numeros.mlf -S entrena.scp -H hmm3/hmmdefs -M hmm4 palabras



HParse numeros.gram numeros.wdnet
HVite -T 1 -C param2.conf -H hmm4/hmmdefs -S evalua.scp -l "mfc" -i salida.mlf -w numeros.wdnet numeros.dic palabras
HResults -I numeros.mlf palabras salida.mlf 

echo "EX" > haz_fonemas.led
HLEd -l '*' -d numeros_fonemas.dic -i numeros_fonemas.mlf haz_fonemas.led numeros.mlf

mkdir fhmm0
HCompV -C param2.conf -S entrena.scp -m -M fhmm0 protofon

python2.7 crea_hmmdefs.py fonemas fhmm0/protofon > fhmm0/fhmmdefs
mkdir fhmm1
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm0/fhmmdefs -M fhmm1 fonemas
mkdir fhmm2
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm1/fhmmdefs -M fhmm2 fonemas
mkdir fhmm3
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm2/fhmmdefs -M fhmm3 fonemas
mkdir fhmm4
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm3/fhmmdefs -M fhmm4 fonemas



HVite -T 1 -C param2.conf -H fhmm4/fhmmdefs -S evalua.scp -l "mfc"  -i fsalida.mlf -w numeros.wdnet numeros_fonemas.dic fonemas
HResults -I numeros.mlf palabras fsalida.mlf 

#Mixturas
echo "MU   2 {*.state[2-4].mix}" > HHEdIncrementMixtures.conf 
mkdir fhmm5
HHEd -H fhmm4/fhmmdefs -M fhmm5 HHEdIncrementMixtures.conf fonemas
mkdir fhmm6
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm5/fhmmdefs -M fhmm6 fonemas
mkdir fhmm7
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm6/fhmmdefs -M fhmm7 fonemas
mkdir fhmm8
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm7/fhmmdefs -M fhmm8 fonemas
mkdir fhmm9
HERest -T 1 -C param2.conf -I numeros_fonemas.mlf -S entrena.scp -H fhmm8/fhmmdefs -M fhmm9 fonemas

HVite -T 1 -C param2.conf -H fhmm9/fhmmdefs -S evalua.scp -l "mfc" -i fsalida_mix.mlf -w numeros.wdnet numeros_fonemas.dic fonemas

HResults -I numeros.mlf palabras fsalida_mix.mlf


#
HParse numerosconectados.gram numerosconectados.wdnet

HVite -T 1 -C param2.conf -H fhmm9/fhmmdefs -S evaluacontinuotodo.scp -l "mfc" -i fsalidacont.mlf -w numerosconectados.wdnet numeros_fonemas_conectados.dic fonemas

HResults -f -p -I evaluaconect.mlf palabras fsalidacont.mlf

ls mfc/nums0[0-3][0-9].mfc > entrenaconect.scp
ls mfc/nums04[0-9].mfc > evaluaconect.scp

HLEd -l "*" -d numeros_fonemas_conectados.dic -i entrenaconect.mlf haz_fonemas.led entrena_fonemas.mlf

mkdir fchmm1
mkdir fchmm2

HERest -T 1 -C param2.conf -I entrenaconect.mlf -S entrenaconect.scp -H fhmm0/fhmmdefs -M fchmm1 fonemas 

HERest -T 1 -C param2.conf -I entrenaconect.mlf -S entrenaconect.scp -H fchmm1/fhmmdefs -M fchmm2 fonemas

HVite -T 1 -C param2.conf -H fchmm2/fhmmdefs -S evaluaconect.scp -l "mfc" -i fsalida_fonemascont.mlf -w numerosconectados.wdnet numeros_fonemas_conectados.dic fonemas

HResults -f -p -I evaluaconect.mlf palabras fsalida_fonemascont.mlf

mkdir fcchmm6
HERest -T 1 -C param2.conf -I entrenaconect.mlf -S entrenaconect.scp -H fhmm5/fhmmdefs -M fcchmm6 fonemas
mkdir fcchmm7
HERest -T 1 -C param2.conf -I entrenaconect.mlf -S entrenaconect.scp -H fcchmm6/fhmmdefs -M fcchmm7 fonemas
mkdir fcchmm8
HERest -T 1 -C param2.conf -I entrenaconect.mlf -S entrenaconect.scp -H fcchmm7/fhmmdefs -M fcchmm8 fonemas
mkdir fcchmm9
HERest -T 1 -C param2.conf -I entrenaconect.mlf -S entrenaconect.scp -H fcchmm8/fhmmdefs -M fcchmm9 fonemas
mkdir fcchmm10
HERest -T 1 -C param2.conf -I entrenaconect.mlf -S entrenaconect.scp -H fcchmm9/fhmmdefs -M fcchmm10 fonemas

HVite -T 1 -C param2.conf -H fcchmm10/fhmmdefs -S evaluaconect.scp -l "mfc" -i fsalida_fonemascont_mix.mlf -w numerosconectados.wdnet numeros_fonemas_conectados.dic fonemas

HResults -I evaluaconect.mlf palabras fsalida_fonemascont_mix.mlf
