import os
import sys
sys.path.append('..')
import argparse
import io

def preprocess(filename):

    raw_text = io.open(filename, 'r', encoding='utf-8').read().lower().replace('\n', ' ').strip()
    clean_txt = filename[:-4] + "_clean.txt"
    if not os.path.exists(clean_txt):
        with io.open(clean_txt,'w+', encoding='ASCII', errors='ignore' ) as f:
            f.write(raw_text)
        return clean_txt

    # formatted_txt = io.open(clean_txt, 'r', encoding='utf-8').read()
    return clean_txt

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--txt_file', type=str, required=True, help="Text to be pre-processed before training") 
    config = parser.parse_args()
    assert os.path.splitext(config.txt_file)[1] == ".txt"

    preprocess(filename=config.txt_file)