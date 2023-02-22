# Modify from gen_aesthetic_embeddings.py
import clip
import argparse
import glob
from PIL import Image
import torch
import tqdm


def update_embed(embed_path, img_dir, name, save=True, context=3):
    '''
    Args:
        - embed_path: path to embedding, e.g. "aesthetic_embeddings/sac_8plus.pt"
        - img_dir: path to all the previous images
        - name: name of the new update embedding
        - save: Save updated embedding to file, default: True
        - context: use the latest `context` of images to update embedding. Default: 3
    '''

    image_paths = glob.glob(f"{img_dir}/*")
    image_paths.sort()
    st = len(image_paths)-1
    end = max(st-context, 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)

    style_emb = torch.load(embed_path).to(device)
    scale = 0.85
    assert 0 <= scale <= 1

    with torch.no_grad():
        embs = []
        for i in range(st, end-1, -1):
            path = image_paths[i]
            print("[embed] Using image from", path)
            image = preprocess(Image.open(path)).unsqueeze(0).to(device)
            emb = model.encode_image(image)
            embs.append(emb.cpu())

        assert len(embs) > 0
        embs = torch.cat(embs, dim=0).mean(dim=0).to(device)

        concat_embed = scale*style_emb + (1-scale)*embs
        #concat_embed = torch.stack(concat_embed, dim=0).mean(dim=0) #, keepdim=True)
        path = f"aesthetic_embeddings/{name}.pt"
        if save:
            # The generated embedding will be located here
            print("[embed] Update embedding and save at", path)
            torch.save(concat_embed, path)

    return path, concat_embed


if __name__ == '__main__':
    embed_path = "aesthetic_embeddings/sac_8plus.pt"
    img_dir = "outputs/txt2img-samples/samples"
    name = "test"
    path, concat = update_embed(embed_path, img_dir, name, save=False)
    print(concat.shape)
