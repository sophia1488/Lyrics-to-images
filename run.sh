export CUDA_VISIBLE_DEVICES=1
#lyrics="I walked through the door with you, the air was cold. But something about it felt like home somehow."
#lyrics="And I, left my scarf there at your sister's house. And you've still got it in your drawer even now"
lyrics="Seven A.M., the usual morning lineup. Start on the chores and sweep 'til the floor's all clean"
ckpt="../Universal-Guided-Diffusion/sd-v1-4.ckpt"

step=20
lyrics_file="lyrics/until_I_found_u_story.txt"
# Whole song
python scripts/lyrics2imgs.py --lyrics $lyrics_file --plms --seed 332 --aesthetic_steps $step --aesthetic_embedding aesthetic_embeddings/sac_8plus.pt --ckpt $ckpt

# Single line
#python scripts/txt2img.py --prompt "$lyrics" --plms --seed 332 --aesthetic_steps $step --aesthetic_embedding aesthetic_embeddings/sac_8plus.pt --ckpt $ckpt
