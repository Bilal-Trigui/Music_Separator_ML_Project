#importing libraries
import os
import subprocess as sp

#convert file from mp4 to wav
def mp4_to_wav(mp4Path, wavPath):
    command = [
        "ffmpeg",
        "-y",              # overwrite output
        "-i", mp4Path,
        wavPath
    ]
    sp.run(command, check=True)

#traverses through mp4 files directory
def batch_convert(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp4"):
            mp4_path = os.path.join(input_directory, filename)
            wav_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}.wav")
            mp4_to_wav(mp4_path, wav_path)

#function calls          
if __name__ == "__main__":
    inputdir  = "/media/bilal/HardDrive2/musdb18/train"
    outputdir = "/media/bilal/HardDrive2/musdb18/wav_train"
    batch_convert(inputdir, outputdir)
