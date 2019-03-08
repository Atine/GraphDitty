import crema
import sys
import warnings
import scipy.io as sio
import os
import subprocess
from multiprocessing import Pool as PPool
from SongStructure import FFMPEG_BINARY
from SalamiExperiments import AUDIO_DIR
model = crema.models.chord.ChordModel()


def compute_crema(num):
    filename = "{}/{}.mp3".format(AUDIO_DIR, num)
    print("Doing crema on %s" % filename)
    matfilename = "%s_crema.mat" % filename
    subprocess.call([FFMPEG_BINARY,
                     "-i", filename,
                     "-ar", "44100",
                     "-ac", "1",
                     "%s.wav" % filename,
                     "-hide_banner"])

    _, y44100 = sio.wavfile.read("%s.wav" % filename)
    y44100 = y44100/2.0**15
    os.remove("%s.wav" % filename)

    data = model.outputs(y=y44100, sr=44100)
    sio.savemat(matfilename, data)


def compute_all_crema(NThreads=12):
    """
    Precompute all crema features in the SALAMI dataset
    """
    # Disable inconsistent hierarchy warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
    songnums = [int(s.replace('.mp3', '')) for s in os.listdir(AUDIO_DIR)]
    if NThreads > -1:
        parpool = PPool(NThreads)
        parpool.map(compute_crema, (songnums))
    else:
        for num in songnums:
            compute_crema(num)


if __name__ == '__main__':
    compute_all_crema(-1)
