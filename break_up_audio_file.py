import argparse
import logging
import sys
import os

from pydub import AudioSegment

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%m-%d %H:%M:%S", stream=sys.stdout)

logger = logging.getLogger("break_up_audio_file")
logger.setLevel(logging.INFO)

def main():
    
    parser = argparse.ArgumentParser(prog="break_up_audio_file", 
                                     usage="./%(prog)s.py -f audio_file.mp3 -o output_dir -t time_chunk_in_secs", 
                                     description="Breaks up an audio file into multiple files with each file being as long as specified")
    parser.add_argument("-f", "--file", help="audio file path", type=str)
    parser.add_argument("-o", "--output_dir", help="output directory where audio chunks are to be saved", type=str)
    parser.add_argument("-t", "--time", help="Time chunks into which audio has to be broken into", type=int)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()
    audio_file_path = args.file
    output_dir = args.output_dir
    time_chunk = args.time
    
    os.makedirs(output_dir, exist_ok=True)
    create_chunks(audio_file_path, output_dir, time_chunk)
    
    
def create_chunks(audio_file_path, output_dir, time_chunk):
    """
    Break down large audio file into small segments and save them in an output folder
    params:
    audio_file_path(str): Audio file path
    output_dir(str): output directory where all small files needs to be saved
    time_chunk(int): time in seconds into which the audio file needs to be broken into
    """
    logger.info(f"audio file - {audio_file_path}, output_dir - {output_dir}, time_chunk - {time_chunk}")
    audio = AudioSegment.from_mp3(audio_file_path)
    audio_len = len(audio)/1000
    if audio_len <= time_chunk:
        logger.error("Audio uploaded is smaller than the chunks you want to break it into")
        sys.exit(1)
    chunks = [audio[i:i+(time_chunk*1000)] for i in range(0, len(audio), time_chunk*1000)]
    for i, chunk in enumerate(chunks):
        chunk.export(f"{output_dir}/chunk-{i}.mp3", format="mp3")
        
if __name__=="__main__":
    main()