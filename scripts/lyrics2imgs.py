import argparse
import os
from scripts.txt2img import txt2img
from scripts.update_embeddings import update_embed

# suppress warning: Some weights of the model checkpoint at openai/clip-vit-large-patch14 were not used when initializing CLIPTextModel
from transformers import logging
logging.set_verbosity_error()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lyrics",
        type=str,
        required=True,
        help="path to lyrics",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="path to prompt",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action="store_true",
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference-aesthetic.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--aesthetic_steps",
        type=int,
        help="number of steps for the aesthetic personalization",
        default=10,
    )
    parser.add_argument(
        "--aesthetic_lr",
        type=float,
        help="learning rate for the aesthetic personalization",
        default=0.0001,
    )
    parser.add_argument(
        "--aesthetic_embedding",
        type=str,
        help="aesthetic embedding file",
        default="aesthetic_embeddings/sac_8plus.pt",
    )

    args = parser.parse_args()

    return args


def get_lyrics(lyrics_file):
    lines = []
    with open(lyrics_file) as f:
        for line in f:
            if line[0] == '#':
                continue
            line = line.strip('\n')
            lines.append(line)

    # merge 2 line into 1
    ret = []
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            line = lines[i] + '. ' + lines[i+1] + '.'
        else:
            line = lines[i] + '.'
        ret.append(line)

    return ret


def main():
    args = get_args()
    lyrics = get_lyrics(args.lyrics)
    song = args.lyrics.split('/')[-1].strip('.txt')
    args.outdir = f'outputs/{song}'
    os.makedirs(args.outdir, exist_ok=True)
    assert len(os.listdir(args.outdir)) == 0, "There should not be images in the output directory."

    for line in lyrics:
        # generate image
        print('>', line)
        print('[lyrics2imgs] Using embed:', args.aesthetic_embedding)
        args.prompt = line
        txt2img(args)

        # update embedding
        '''
        new_embed_path, _ = update_embed(args.aesthetic_embedding, args.outdir, song)
        args.aesthetic_embedding = new_embed_path
        '''
    print(f"[lyrics2imgs] Your samples are ready and waiting for you here: '{args.outdir}'. Enjoy!")

if __name__ == '__main__':
    main()
