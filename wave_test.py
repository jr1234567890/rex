
# from pydub import AudioSegment
# from pydub.playback import play

# #audio track setup
# roarsong = AudioSegment.from_wav("trex_roar.wav")
# frogsong = AudioSegment.from_wav("frog.wav")
# barneysong = AudioSegment.from_wav("barney.wav")

# play(frogsong)


import simpleaudio as sa
from time import sleep

wave_obj = sa.WaveObject.from_wave_file("trex_roar.wav")
play_obj = wave_obj.play()

sleep (1)
play_obj.stop()
#
# 
# 
# play_obj.wait_done()