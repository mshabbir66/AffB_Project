function MFCCs = ExtractMFCC(soundSignal,fs)

wavwrite(soundSignal,fs,'temp.wav');
eval(['!HCopy -A -D -C analysis.conf ','temp.wav',' temp.mfcc']);
MFCCs = htkread('temp.mfcc');

end