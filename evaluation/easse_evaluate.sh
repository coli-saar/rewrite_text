# --refs_sents_paths /home/AK/skrjanec/rewrite_text/experiments/2/checkpoints/test.src-tgt.tgt --orig_sents_path /home/AK/skrjanec/rewrite_text/experiments/2/checkpoints/test.src-tgt.src --sys_sents_path /home/AK/skrjanec/rewrite_text/experiments/2/checkpoints/generation2.out

#ORIGINAL_SOURCE = $1
#REF = $2
#HYPOTHESES = $3

# evalate with EASSE
easse evaluate -m bleu,sari,fkgl,bertscore -t custom -tok moses --refs_sents_paths $2 --orig_sents_path $1 --sys_sents_path $3

