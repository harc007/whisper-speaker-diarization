import argparse
import logging
import sys
import os
import glob
import whisper
from pyannote.core import Segment
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
import numpy as np
import torch
from sklearn.cluster import KMeans

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s - %(message)s", datefmt="%m-%d %H:%M:%S", stream=sys.stdout)
logger = logging.getLogger("diarize_and_transcribe")
logger.setLevel(logging.INFO)

WHISPER_MODEL = whisper.load_model("medium")
AUDIO = Audio()
EMBEDDING_MODEL = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb", device=torch.device('cpu'))


def get_clustering_files(input_folder, no_files_for_diarize):
    if os.path.isdir(input_folder):
        if len(glob.glob(os.path.join(input_folder, "chunk-*.mp3"))) > no_files_for_diarize:
            logger.info(f"We have enough files to work with")
            return True, [os.path.join(input_folder, f"chunk-{i}.mp3") for i in range(no_files_for_diarize)]
        else:
            logger.info("We dont have sufficient files in the folder given. Exiting.")
            return False, []
    else:
        logger.info(f"No folder - {input_folder}. Exiting.")
        return False, []
    
    
def get_segments_from_whisper(files_for_clust):
    whisper_segments = {}
    for file in files_for_clust:
        logger.info(f"Transcribing file {file}")
        result = WHISPER_MODEL.transcribe(file, fp16=False)
        whisper_segments[file] = result['segments']
    return whisper_segments

def get_embedding_length(whisper_segments):
    count = 0
    for seg in list(whisper_segments.values()):
        count += len(seg)
    return count

def get_speaker_embeddings(whisper_segments, crop_max):
    embeddings = np.zeros(shape=(get_embedding_length(whisper_segments), 192))
    seg_idx = 0
    for file, segments in whisper_segments.items():
        logger.info(f"Getting speaker embeddings for file {file}")
        for i, segment in enumerate(segments):
            clip = Segment(segment['start'], min(segment['end'], crop_max))
            waveform, sample_rate = Audio.crop(file, clip)
            embs = EMBEDDING_MODEL(waveform[0][None, None, :])
            embeddings[seg_idx] = embs
            seg_idx += 1
    return np.nan_to_num(embeddings)

def perform_clustering_and_infer(no_speakers, embeddings, input_folder, speaker_names, crop_max):
    clg = KMeans(no_speakers).fit(embeddings)
    label_to_idx_map = get_first_occurence_of_multiples(clg.labels_)
    files_for_preds = [os.path.join(input_folder, f"chunk-{i}.mp3") for i in range(len(os.listdir(input_folder)))]
    complete_whisper_segments = get_segments_from_whisper(files_for_preds)
    complete_embeddings = get_speaker_embeddings(complete_whisper_segments, crop_max)
    complete_labels = clg.predict(complete_embeddings)
    old_speaker, final_text = '', ''
    idx = 0
    for file, segments in complete_whisper_segments.items():
        for chunk in segments:
            speaker = speaker_names[label_to_idx_map[complete_labels[idx]]]
            idx += 1
            if len(old_speaker) == 0:
                final_text += f"{speaker} - {chunk['text']}"
            else:
                if old_speaker != speaker:
                    final_text += f"\n\n{speaker} - {chunk['text']}"
                else:
                    final_text += f" {chunk['text']}"
    return final_text


def get_first_occurence_of_multiples(xs, n=3):
    ti = {}
    i = 0
    idx_count = 0
    while i < len(xs) - (n-1):
        if all([xs[i]==xs[i+p] for p in range(1, n)]) and xs[i] not in ti:
            ti[xs[i]] = idx_count
            i += n
            idx_count += 1
        else:
            i += 1
    return ti


def main():
    logger.info("In main")
    parser = argparse.ArgumentParser(prog="diarize_and_transcribe", 
                                     usage="./%(prog)s.py -i input_folder -f no_files_for_diarize -s no_speakers -n speaker_names", 
                                     description="""Take first few lines and get speaker embeddings and cluster basd on no_speakers. 
                                                    Transcribe along with speakers for all the audio files and save them as txt file""")
    parser.add_argument("-i", "--input_folder", help="input folder with split files", type=str)
    parser.add_argument("-f", "--no_files_for_diarize", help="number of files to train clustering on", type=int)
    parser.add_argument("-s", "--no_speakers", help="number of speakers", type=int)
    parser.add_argument("-n", "--speaker_names", help="speaker names comma separated", type=str)
    parser.add_argument("-t", "--time", help="time chunks into which audio has to be broken into", type=int)
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr())
        sys.exit(1)
        
    args = parser.parse_args()
    input_folder = args.input_folder
    no_files_for_diarize = args.no_files_for_diarize
    no_speakers = args.no_speakers
    speaker_names = args.speaker_names.split(',')
    crop_max = args.time
    
    logger.info(f"""input folder - {input_folder}, files to diarize - {no_files_for_diarize}, speaker count - {no_speakers} 
                    speaker names - {speaker_names}, crop max - {crop_max}""")
    is_files_present, files_for_clust = get_clustering_files(input_folder, no_files_for_diarize)
    if not is_files_present:
        sys.exit(1)
        
    logger.info("Files are present and we are now going to load whisper model and get the segments")
    whisper_segments = get_segments_from_whisper(files_for_clust)
    logger.info(f"Sample segment - {whisper_segments[files_for_clust[0][0]]}")
    
    logger.info("Getting speaker embeddings")
    speaker_embeddings = get_speaker_embeddings(whisper_segments, crop_max)
    logger.info(f"Obtained speaker embeddings with shape {speaker_embeddings.shape}")
    
    logger.info("Clustering to identify speakers")
    final_text = perform_clustering_and_infer(no_speakers, speaker_embeddings, input_folder, speaker_names, crop_max)
    logger.info(f"Final Text - \n\n {final_text}")
    
if __name__ == '__main__':
    main()